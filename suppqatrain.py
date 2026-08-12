from __future__ import annotations

import pandas as pd
import openai
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from openreward.toolsets import WebToolset


# ============= Data Loading (module-level) =============

import os
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
    # Which provider answers is process configuration (OPENREWARD_SEARCH_BACKEND,
    # default "backsearch"), so changing search provider needs no change here.
    #
    # The toolset owns the error split too: an unfetchable page stays tool output
    # the agent can act on, while a missing key or exhausted quota raises so the
    # rollout ends with a blank reward rather than a score that reads as a bad answer.
    toolsets = [WebToolset]

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

        # Validate API keys from secrets (no env var fallback)
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key required in secrets parameter. "
                "Pass secrets={'openai_api_key': 'your-key'} when creating session."
            )

        # Read live by WebToolset on every tool call, so the search backend takes its
        # credentials from the session rather than the server process. The configured
        # backend picks the key it needs: `api_key` for backsearch, `tavily_api_key`
        # for tavily. No up-front check — which key is required depends on the backend.
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
        grader_result = await self._grade_answer(params.answer)

        reward = grader_result["reward"]
        is_correct = grader_result["is_correct"]
        justification = grader_result["justification"]

        result_text = "CORRECT" if is_correct else "INCORRECT"

        display_text = f"""{result_text}

Evaluation:
{justification}

Reference Answer: {self.answer}
"""

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "task_id": self.task_id,
                "submitted_answer": params.answer,
                "reference_answer": self.answer,
                "is_correct": is_correct,
                "justification": justification,
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

        try:
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
        except Exception as e:
            return {
                "is_correct": False,
                "justification": f"Grading failed due to error: {str(e)}",
                "reward": 0.0,
            }
