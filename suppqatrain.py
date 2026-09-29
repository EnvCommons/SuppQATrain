from __future__ import annotations

import os

import pandas as pd
import openai
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1
from openreward.toolsets import WebToolset
from openreward.tools.search import describe_fetch, describe_search

# This environment searches the live web through Tavily, always. See
# SuppQATrain.search_backend for the pin that routes every tool call.
SEARCH_BACKEND = "tavily"

# Anything in this process that resolves a backend without an explicit pin —
# a bare Search()/Fetch(), a WebToolset built against some other object — reads
# this rather than falling back to backsearch.
os.environ["OPENREWARD_SEARCH_BACKEND"] = SEARCH_BACKEND

# WebToolset resolves its tool *descriptions* once, at its own import time, and
# openreward.environments already pulls that module in — so by the time this
# file runs, the descriptions are whatever the process environment said, which
# for backsearch promises results "on or before the configured cutoff date".
# That is untrue of live-web Tavily, and a model told its sources are bounded
# will not hedge about post-cutoff information. Re-resolve them against the
# pinned backend, which is what the SDK itself does at the bottom of that module.
WebToolset.web_search.__doc__ = describe_search(SEARCH_BACKEND)
WebToolset.web_fetch.__doc__ = describe_fetch(SEARCH_BACKEND)


# ============= Data Loading (module-level) =============

from pathlib import Path

if Path("/orwd_data/").exists():
    DATA_PATH = Path("/orwd_data")
else:
    DATA_PATH = Path(__file__).parent

train_df = pd.read_parquet(DATA_PATH / "train.parquet")

# Create task specs (public) and answers (backend only)
TASKS_BY_SPLIT = {"train": []}
ANSWERS = {}

for idx, row in train_df.iterrows():
    task_id = f"suppqatrain_train_{idx}"

    # Public task spec (no answer or key_passage)
    TASKS_BY_SPLIT["train"].append({
        "id": task_id,
        "question": row["question"],
        "source_doi": row.get("source_doi", ""),
        "domain": row.get("domain", ""),
        "supp_type": row.get("supp_type", ""),
    })

    # Private answer storage
    ANSWERS[task_id] = {
        "answer": row["answer"],
    }


# ============= Pydantic Models for Tool Inputs =============
class SubmitAnswerInput(BaseModel):
    answer: str


# ============= Environment Class =============
class SuppQATrain(Environment):
    """
    SuppQATrain: A scientific QA environment focused on supplementary materials,
    with web search and LLM-based semantic grading.
    """

    # web_search / web_fetch come from the SDK rather than being hand-rolled here.
    #
    # The toolset owns the error split too: an unfetchable page stays tool output
    # the agent can act on, while a missing key or exhausted quota raises so the
    # rollout ends with a blank reward rather than a score that reads as a bad answer.
    toolsets = [WebToolset]

    # Tavily, always. WebToolset reads this hook on every tool call and an explicit
    # value beats OPENREWARD_SEARCH_BACKEND, so the backend cannot be swapped out
    # from under the environment by process configuration. Questions here are drawn
    # from supplementary materials of published papers, which the agent has to reach
    # on the live web; the default backdated corpus does not carry them.
    search_backend = SEARCH_BACKEND

    # Search hits keep their snippets, as the prompt promises. Off in the SDK by
    # default, which would force a fetch per candidate just to triage results.
    web_include_snippets = True

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        # Extract task info
        self.task_id = str(task_spec["id"])
        self.question = str(task_spec["question"])
        self.source_doi = str(task_spec.get("source_doi", ""))
        self.domain = str(task_spec.get("domain", ""))
        self.supp_type = str(task_spec.get("supp_type", ""))

        # Graded submissions this session. Only the first is rewarded: the
        # feedback prints the reference answer in full, so an uncapped tool
        # would let the agent read it and resubmit.
        self.submitted = 0

        # Validate API keys from secrets (no env var fallback)
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key required in secrets parameter. "
                "Pass secrets={'openai_api_key': 'your-key'} when creating session."
            )

        # Read live by WebToolset on every tool call, so the search backend takes its
        # credentials from the session rather than the server process. Tavily wants
        # `tavily_api_key`. No up-front check: the toolset also falls back to the
        # server process environment (TAVILY_API_KEY), and a genuinely missing key
        # raises SearchBackendUnavailable at call time, which discards the rollout
        # rather than scoring it 0.0.
        self.search_secrets = secrets

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)

        # Load answer from backend storage
        answer_data = ANSWERS.get(self.task_id)
        if not answer_data:
            raise ValueError(f"Task {self.task_id} not found in dataset")

        self.answer = str(answer_data["answer"])

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in TASKS_BY_SPLIT:
            raise ValueError(
                f"Unknown split: {split}. Available splits: {list(TASKS_BY_SPLIT.keys())}"
            )
        return TASKS_BY_SPLIT[split]

    async def get_prompt(self) -> list[TextBlock]:
        prompt_text = f"""{self.question}

When you have your answer, submit it using the submit_answer tool."""

        return [TextBlock(text=prompt_text)]

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your final answer to the scientific question.
        This tool will grade your answer against the reference answer and end the episode.
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="An answer has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        grader_result = await self._grade_answer(params.answer)

        reward = grader_result["reward"]
        is_correct = grader_result["is_correct"]

        # Only the verdict is shown: the grader's justification is written
        # with the reference answer in view and can restate it.
        display_text = "CORRECT" if is_correct else "INCORRECT"

        # Incremented only after grading succeeds, so a grader failure leaves
        # the attempt retryable.
        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "task_id": self.task_id,
                "submitted_answer": params.answer,
                "is_correct": is_correct,
                "domain": self.domain,
                "supp_type": self.supp_type,
            },
            reward=reward,
            finished=True,
        )

    async def _grade_answer(self, predicted_answer: str) -> dict:
        """
        Grade the answer using gpt-5-mini LLM grader.
        Compares submitted answer against reference answer for semantic equivalence.
        """
        if not predicted_answer or len(predicted_answer.strip()) == 0:
            return {
                "is_correct": False,
                "justification": "Empty or whitespace-only answer provided.",
                "reward": 0.0,
            }

        grader_prompt = f"""You are an expert scientific evaluator. Determine if the predicted answer is semantically equivalent to the reference answer.

Question: {self.question}

Reference Answer: {self.answer}

Predicted Answer: {predicted_answer}

Instructions:
1. Check if the predicted answer is semantically equivalent to the reference answer
2. Consider synonyms, abbreviations, and equivalent scientific terminology
3. Ignore minor formatting differences
4. Do NOT require exact word-for-word matches
5. For numerical answers, allow minor rounding differences
6. For sequence answers (DNA, RNA, protein), require exact match
7. Provide a brief justification (2-3 sentences)
8. End your response with EXACTLY one of these labels on a new line:
   - "CORRECT" if semantically equivalent
   - "INCORRECT" if not equivalent

Format:
[Your justification here]

CORRECT or INCORRECT"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}],
        )

        grading_response = response.choices[0].message.content or ""

        upper_response = grading_response.upper()
        is_correct = "CORRECT" in upper_response and "INCORRECT" not in upper_response

        reward = 1.0 if is_correct else 0.0

        return {
            "is_correct": is_correct,
            "justification": grading_response,
            "reward": reward,
        }
