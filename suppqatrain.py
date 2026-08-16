from __future__ import annotations

import pandas as pd
import openai
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool


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
class WebSearchInput(BaseModel):
    query: str


class FetchUrlInput(BaseModel):
    url: str
    page: int = Field(default=1, description="Page number to retrieve (1-indexed). Each page contains ~10,000 characters.")


class SubmitAnswerInput(BaseModel):
    answer: str


# ============= Environment Class =============
class SuppQATrain(Environment):
    """
    SuppQATrain: A scientific QA environment focused on supplementary materials,
    with web search and LLM-based semantic grading.
    """

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

        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "Tavily API key required in secrets parameter. "
                "Pass secrets={'tavily_api_key': 'your-key'} when creating session."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

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
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web using Tavily. Returns search results with titles, URLs, and snippets.
        Use fetch_url tool to get full content from specific URLs if needed.
        """
        try:
            response = await self.tavily_client.search(
                query=params.query,
                search_depth="basic",
                max_results=5,
            )

            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(text="No search results found.")],
                    metadata={"query": params.query, "results": []},
                    reward=0.0,
                    finished=False,
                )

            display_parts = [f"Search results for: {params.query}\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("content", "")
                display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

            display_text = "\n".join(display_parts)

            return ToolOutput(
                blocks=[TextBlock(text=display_text)],
                metadata={
                    "query": params.query,
                    "results": results,
                    "count": len(results),
                },
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Web search failed: {str(e)}")],
                metadata={"query": params.query, "error": str(e)},
                reward=0.0,
                finished=False,
            )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the text content from a specific URL using Tavily's extract method.
        Content is paginated - use the page parameter to retrieve additional pages.
        """
        PAGE_SIZE = 10000  # Characters per page

        try:
            response = await self.tavily_client.extract(urls=[params.url])

            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(text=f"No content extracted from {params.url}")],
                    metadata={"url": params.url, "results": []},
                    reward=0.0,
                    finished=False,
                )

            result = results[0]
            raw_content = result.get("raw_content", "")
            total_length = len(raw_content)

            # Calculate total pages
            total_pages = max(1, (total_length + PAGE_SIZE - 1) // PAGE_SIZE)
            page = max(1, min(params.page, total_pages))  # Clamp to valid range

            # Extract the requested page
            start_idx = (page - 1) * PAGE_SIZE
            end_idx = min(start_idx + PAGE_SIZE, total_length)
            page_content = raw_content[start_idx:end_idx]

            # Build display with pagination info
            if total_pages == 1:
                display_text = f"Content from {params.url}:\n\n{page_content}"
            else:
                display_text = f"Content from {params.url} (Page {page}/{total_pages}):\n\n{page_content}"
                if page < total_pages:
                    display_text += f"\n\n[Use fetch_url with page={page + 1} to see more content]"

            return ToolOutput(
                blocks=[TextBlock(text=display_text)],
                metadata={
                    "url": params.url,
                    "page": page,
                    "total_pages": total_pages,
                    "total_length": total_length,
                    "page_start": start_idx,
                    "page_end": end_idx,
                },
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Failed to fetch URL: {str(e)}")],
                metadata={"url": params.url, "error": str(e)},
                reward=0.0,
                finished=False,
            )

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
