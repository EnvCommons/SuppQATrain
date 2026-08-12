# SuppQATrain

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/SuppQATrain)

## Description

SuppQATrain is an environment for evaluating question answering over supplementary materials of scientific papers, based on FutureHouse's [SuppQA](https://github.com/Future-House/LAB-Bench) subtask within LAB-Bench. Each question asks about a specific verifiable fact found exclusively in a paper's supplementary data rather than the main text. Questions are designed to be specific enough that they can only be answered from a single source, and answers require locating and reading the correct supplementary material.

## Capabilities

- Question answering from supplementary materials of scientific papers
- Web search and information retrieval from academic literature
- Multi-step research: searching, reading papers, locating supplementary data, and extracting precise facts
- Verifiable factual recall from supplementary tables, methods, and data sections

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There is one split: **train** with 999 tasks spanning 10 scientific domains:

| Domain | Count |
|--------|-------|
| Chemistry / Materials science | 100 |
| Computational biology / Bioinformatics | 100 |
| Computer science / AI | 100 |
| Earth science / Geology | 100 |
| Ecology / Environmental science | 100 |
| Engineering / Applied science | 99 |
| Medicine / Clinical research | 100 |
| Molecular biology / Genomics | 100 |
| Neuroscience | 100 |
| Physics / Astronomy | 100 |

Each task provides a question and metadata (source DOI, domain, supplementary type). The agent prompt contains only the question; the agent must find the answer through web search and supplementary material retrieval.

## Reward Structure

Reward is sparse and binary, emitted only when the agent calls `submit_answer` (which ends the episode). The `web_search` and `web_fetch` tools always return reward 0.0 and do not end the episode.

On submission, the agent's answer is evaluated by an LLM grader (gpt-5-mini) that checks semantic equivalence against the reference answer. The grader accounts for synonyms, abbreviations, equivalent scientific terminology, and minor formatting or rounding differences. For biological sequence answers (DNA, RNA, protein), exact match is required. Empty or whitespace-only submissions receive reward 0.0 without invoking the grader.

- **1.0**: Submitted answer is semantically equivalent to the reference answer
- **0.0**: Submitted answer is incorrect, missing, or empty

## Data

Data consists of a single Parquet file (`train.parquet`) containing 999 QA pairs generated from supplementary materials of papers published in trusted journals (Nature, Nature Communications, Scientific Reports, PLOS, Science, PNAS, and others). Each row contains a question, answer, source DOI, key passage from supplementary material, domain, and supplementary type. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web. Returns up to 5 results with titles, URLs, and snippets. |
| `web_fetch` | Fetch full text content from a specific URL (truncated at 8000 characters). |
| `submit_answer` | Submit your final answer for LLM grading. Ends the episode. |

Search and fetch come from the OpenReward SDK's `WebToolset`, so the provider is configuration on the environment server rather than code here:

| `OPENREWARD_SEARCH_BACKEND` | Backend | Needs |
|---|---|---|
| unset (default) | `backsearch` — GR's backdated corpus, bounded to an `as_of` cutoff | `OPENREWARD_API_KEY`, or `api_key` in session secrets |
| `tavily` | Tavily — live web | `TAVILY_API_KEY`, or `tavily_api_key` in session secrets |

## Time Horizon

Multi-turn. Agents can perform multiple web searches and URL fetches before submitting a final answer.

## Environment Difficulty

[To be determined]

## Other Environment Requirements

- OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.
- Search credentials — whichever the configured backend needs: `api_key` for the default backsearch backend, or `tavily_api_key` when the server runs with `OPENREWARD_SEARCH_BACKEND=tavily`. Both fall back to the server process environment (`OPENREWARD_API_KEY` / `TAVILY_API_KEY`).

## Safety

Agents interact with the web via the SDK's search tools, over whichever backend the server is configured with. While the environment is designed for retrieving scientific papers, agents can in principle search for or fetch arbitrary URLs. The environment does not restrict search queries or target domains. No sandbox or file system access is provided, so agents cannot persist data or execute code.

The questions themselves concern published scientific literature and do not involve sensitive, hazardous, or dual-use information beyond what is already publicly available in peer-reviewed journals.

## Citations

```bibtex
@article{laurent2024labbench,
  title     = {LAB-Bench: Measuring Capabilities of Language Models for Biology Research},
  author    = {Laurent, Jon M. and Janizek, Joseph D. and Ruzo, Michael and Hinks, Michaela M. and Hammerling, Michael J. and Narayanan, Siddharth and Ponnapati, Manvitha and White, Andrew D. and Rodriques, Samuel G.},
  year      = {2024},
  eprint    = {2407.10362},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

```bibtex
@dataset{GRSuppQATrain,
  author    = {General Reasoning Inc. Team},
  title     = {SuppQATrain},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/SuppQATrain}
}
```
