"""
SuppQATrain Dataset Generator

Generates scientific QA pairs from SUPPLEMENTARY MATERIALS of published papers by:
1. Searching for papers with accessible supplementary data across 10 domains
2. Extracting content (including supplementary info) via Tavily
3. Using gpt-4.1 to generate verifiable QA pairs from supplementary content only
4. Validating and deduplicating results

Usage:
    export OPENAI_API_KEY="sk-..."
    export TAVILY_API_KEY="tvly-..."
    python generate_dataset.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

# ============= Configuration =============

TARGET_PER_DOMAIN = 100  # 100 per domain = 1000 total
OUTPUT_PARQUET = Path(__file__).parent / "train.parquet"
PROGRESS_PARQUET = Path(__file__).parent / "suppqatrain_progress.parquet"
REFERENCE_FILE = Path(__file__).parent / "reference.txt"

DOMAINS = [
    "Molecular biology / Genomics",
    "Neuroscience",
    "Computational biology / Bioinformatics",
    "Ecology / Environmental science",
    "Medicine / Clinical research",
    "Physics / Astronomy",
    "Computer science / AI",
    "Chemistry / Materials science",
    "Earth science / Geology",
    "Engineering / Applied science",
]

# Search queries per domain — targeting papers with accessible supplementary materials
DOMAIN_SEARCH_QUERIES = {
    "Molecular biology / Genomics": [
        "site:nature.com/articles/s41467 supplementary table primer genomics 2024",
        "site:nature.com/articles/s41467 supplementary table sequencing 2024",
        "site:nature.com/articles/s41467 supplementary data CRISPR 2024",
        "site:nature.com/articles/s41467 supplementary methods gene regulation 2024",
        "site:nature.com/articles/s41467 supplementary table antibody reagents 2024",
        "site:journals.plos.org supporting information table primer PCR 2024",
        "site:journals.plos.org S1 Table genomics sequencing 2024",
        "site:journals.plos.org supporting information oligonucleotides 2024",
        "site:nature.com/articles/s41598 supplementary table gene expression 2024",
        "site:nature.com/articles/s41598 supplementary data RNA-seq 2024",
        "site:nature.com supplementary table epigenetics methylation 2024",
        "site:nature.com supplementary data proteomics mass spectrometry 2024",
        "site:journals.plos.org supporting information CRISPR sgRNA 2024",
        "site:nature.com/articles/s41467 supplementary table single cell 2023",
        "site:journals.plos.org S1 Table primers gene expression 2023",
        "site:nature.com supplementary data genome assembly 2024",
        "site:nature.com supplementary table chromatin immunoprecipitation 2024",
        "site:journals.plos.org supporting information cloning primers 2024",
        "site:nature.com/articles/s41467 supplementary table siRNA knockdown 2024",
        "site:nature.com supplementary methods whole genome sequencing 2024",
    ],
    "Neuroscience": [
        "site:nature.com/articles/s41467 supplementary table neuroscience brain 2024",
        "site:nature.com/articles/s41467 supplementary data neural circuits 2024",
        "site:nature.com/articles/s41467 supplementary methods electrophysiology 2024",
        "site:nature.com/articles/s41467 supplementary table mouse brain 2024",
        "site:nature.com/articles/s41593 supplementary table 2024",
        "site:nature.com/articles/s41593 supplementary methods 2024",
        "site:journals.plos.org supporting information table neuroscience 2024",
        "site:journals.plos.org S1 Table brain imaging 2024",
        "site:nature.com/articles/s41598 supplementary table cognitive 2024",
        "site:nature.com supplementary data hippocampus 2024",
        "site:nature.com supplementary table optogenetics viral vectors 2024",
        "site:nature.com supplementary data dopamine reward 2024",
        "site:journals.plos.org supporting information primers neuron 2024",
        "site:nature.com supplementary table synapse plasticity 2024",
        "site:nature.com/articles/s41467 supplementary methods calcium imaging 2023",
        "site:nature.com supplementary table glia astrocyte 2024",
        "site:nature.com supplementary data EEG oscillations 2024",
        "site:journals.plos.org supporting information behavioral neuroscience 2024",
        "site:nature.com supplementary table neurodegeneration biomarker 2024",
        "site:nature.com supplementary methods patch clamp 2024",
    ],
    "Computational biology / Bioinformatics": [
        "site:journals.plos.org/ploscompbiol supporting information table 2024",
        "site:journals.plos.org/ploscompbiol S1 Table parameters 2024",
        "site:journals.plos.org/ploscompbiol supporting information simulation 2024",
        "site:nature.com/articles/s41467 supplementary table machine learning biology 2024",
        "site:nature.com/articles/s41467 supplementary data algorithm benchmark 2024",
        "site:nature.com supplementary table network analysis 2024",
        "site:nature.com supplementary methods bioinformatics pipeline 2024",
        "site:journals.plos.org/ploscompbiol S1 Table software parameters 2024",
        "site:journals.plos.org/ploscompbiol supporting information benchmark 2024",
        "site:nature.com supplementary table protein structure prediction 2024",
        "site:nature.com supplementary data phylogenetic analysis 2024",
        "site:journals.plos.org supporting information table computational 2023",
        "site:nature.com/articles/s41598 supplementary table bioinformatics 2024",
        "site:journals.plos.org/ploscompbiol S1 Table model performance 2024",
        "site:nature.com supplementary table systems biology 2024",
        "site:journals.plos.org supporting information dataset statistics 2024",
        "site:nature.com supplementary methods sequence alignment 2024",
        "site:journals.plos.org/ploscompbiol supplementary table CRISPR analysis 2024",
        "site:nature.com supplementary data drug target prediction 2024",
        "site:journals.plos.org supporting information hyperparameters 2024",
    ],
    "Ecology / Environmental science": [
        "site:nature.com/articles/s41467 supplementary table ecology biodiversity 2024",
        "site:nature.com/articles/s41467 supplementary data species conservation 2024",
        "site:nature.com/articles/s41467 supplementary table climate change 2024",
        "site:journals.plos.org supporting information table ecology 2024",
        "site:journals.plos.org S1 Table species abundance 2024",
        "site:nature.com/articles/s41598 supplementary table environmental 2024",
        "site:nature.com supplementary data microplastics pollution 2024",
        "site:nature.com supplementary table coral reef ocean 2024",
        "site:journals.plos.org supporting information sampling sites 2024",
        "site:nature.com supplementary data carbon sequestration 2024",
        "site:nature.com supplementary table marine ecology fisheries 2024",
        "site:journals.plos.org S1 Table coordinates sampling 2024",
        "site:nature.com supplementary methods field survey 2024",
        "site:nature.com supplementary table pollinator decline 2024",
        "site:journals.plos.org supporting information invasive species 2024",
        "site:nature.com supplementary data deforestation tropical 2024",
        "site:nature.com/articles/s41559 supplementary table 2024",
        "site:nature.com supplementary table bird migration phenology 2024",
        "site:journals.plos.org supporting information water quality 2024",
        "site:nature.com supplementary data permafrost methane 2024",
    ],
    "Medicine / Clinical research": [
        "site:nature.com/articles/s41467 supplementary table clinical trial 2024",
        "site:nature.com/articles/s41467 supplementary data patient cohort 2024",
        "site:nature.com/articles/s41467 supplementary table cancer treatment 2024",
        "site:nature.com/articles/s41591 supplementary table 2024",
        "site:nature.com/articles/s41598 supplementary table clinical 2024",
        "site:journals.plos.org supporting information table clinical 2024",
        "site:journals.plos.org S1 Table patient demographics 2024",
        "site:nature.com supplementary data immunotherapy response 2024",
        "site:nature.com supplementary table survival analysis 2024",
        "site:nature.com supplementary methods diagnostic assay 2024",
        "site:journals.plos.org supporting information hazard ratio 2024",
        "site:nature.com supplementary table biomarker panel 2024",
        "site:nature.com supplementary data vaccine efficacy 2024",
        "site:journals.plos.org S1 Table epidemiology cohort 2024",
        "site:nature.com supplementary table gene therapy clinical 2024",
        "site:nature.com supplementary data drug resistance 2024",
        "site:journals.plos.org supporting information regression model 2024",
        "site:nature.com supplementary table multivariate analysis 2024",
        "site:nature.com/articles/s41591 supplementary methods 2024",
        "site:nature.com supplementary data adverse events 2024",
    ],
    "Physics / Astronomy": [
        "site:nature.com/articles/s41467 supplementary table physics 2024",
        "site:nature.com/articles/s41567 supplementary table 2024",
        "site:nature.com/articles/s41567 supplementary methods 2024",
        "site:nature.com/articles/s41586 supplementary table quantum 2024",
        "site:nature.com supplementary data superconductor 2024",
        "site:nature.com supplementary table photonics laser 2024",
        "site:nature.com supplementary methods device fabrication 2024",
        "site:nature.com/articles/s41550 supplementary table 2024",
        "site:nature.com supplementary data exoplanet JWST 2024",
        "site:nature.com supplementary table qubit performance 2024",
        "site:nature.com supplementary methods material characterization 2024",
        "site:nature.com supplementary table spin quantum 2024",
        "site:nature.com/articles/s41467 supplementary data gravitational waves 2024",
        "site:nature.com supplementary table neutron star observation 2024",
        "site:nature.com supplementary methods cryogenic measurement 2024",
        "site:nature.com supplementary table dark matter detection 2024",
        "site:nature.com supplementary data Bose-Einstein condensate 2024",
        "site:nature.com/articles/s41598 supplementary table astrophysics 2024",
        "site:nature.com supplementary methods optical setup 2024",
        "site:nature.com supplementary data plasma fusion 2024",
    ],
    "Computer science / AI": [
        "site:nature.com/articles/s41467 supplementary table machine learning 2024",
        "site:nature.com supplementary data deep learning benchmark 2024",
        "site:nature.com supplementary table neural network hyperparameters 2024",
        "site:nature.com/articles/s42256 supplementary table 2024",
        "site:nature.com/articles/s42256 supplementary methods 2024",
        "site:journals.plos.org supporting information table classification 2024",
        "site:journals.plos.org S1 Table model accuracy 2024",
        "site:nature.com supplementary methods training details 2024",
        "site:nature.com supplementary table dataset statistics 2024",
        "site:nature.com supplementary data AI protein structure 2024",
        "site:journals.plos.org supporting information hyperparameter tuning 2024",
        "site:nature.com supplementary table reinforcement learning 2024",
        "site:nature.com supplementary methods computational resources 2024",
        "site:nature.com/articles/s41598 supplementary table computer science 2024",
        "site:journals.plos.org S1 Table NLP text mining 2024",
        "site:nature.com supplementary data graph neural network 2024",
        "site:nature.com supplementary table model comparison 2024",
        "site:journals.plos.org supporting information feature importance 2024",
        "site:nature.com supplementary table ablation study 2024",
        "site:nature.com supplementary methods data preprocessing 2024",
    ],
    "Chemistry / Materials science": [
        "site:nature.com/articles/s41467 supplementary table chemistry synthesis 2024",
        "site:nature.com/articles/s41467 supplementary data materials 2024",
        "site:nature.com supplementary table catalysis reaction conditions 2024",
        "site:nature.com supplementary methods characterization XRD 2024",
        "site:nature.com/articles/s41563 supplementary table 2024",
        "site:nature.com/articles/s41563 supplementary methods 2024",
        "site:nature.com supplementary table battery electrolyte 2024",
        "site:nature.com supplementary data perovskite solar 2024",
        "site:nature.com supplementary table polymer nanoparticle 2024",
        "site:nature.com supplementary methods synthesis procedure 2024",
        "site:nature.com/articles/s41929 supplementary table 2024",
        "site:nature.com supplementary table MOF metal organic framework 2024",
        "site:nature.com supplementary data electrochemistry 2024",
        "site:journals.plos.org supporting information table chemistry 2024",
        "site:nature.com supplementary table crystal structure 2024",
        "site:nature.com supplementary methods spectroscopy NMR 2024",
        "site:nature.com supplementary table alloy composition 2024",
        "site:nature.com supplementary data semiconductor quantum dot 2024",
        "site:nature.com supplementary table photocatalysis 2024",
        "site:nature.com supplementary methods thin film deposition 2024",
    ],
    "Earth science / Geology": [
        "site:nature.com/articles/s41467 supplementary table geoscience 2024",
        "site:nature.com/articles/s41467 supplementary data earthquake 2024",
        "site:nature.com/articles/s41561 supplementary table 2024",
        "site:nature.com/articles/s41561 supplementary methods 2024",
        "site:nature.com supplementary table volcanic eruption 2024",
        "site:nature.com supplementary data ice sheet glacier 2024",
        "site:nature.com supplementary table sample locations coordinates 2024",
        "site:nature.com supplementary methods dating isotope 2024",
        "site:nature.com/articles/s41598 supplementary table geology 2024",
        "site:nature.com supplementary data ocean circulation 2024",
        "site:nature.com supplementary table mineral composition 2024",
        "site:nature.com supplementary data paleoclimate proxy 2024",
        "site:journals.plos.org supporting information table geomorphology 2024",
        "site:nature.com supplementary table groundwater aquifer 2024",
        "site:nature.com supplementary methods seismic analysis 2024",
        "site:nature.com supplementary data sediment stratigraphy 2024",
        "site:nature.com supplementary table sea level rise 2024",
        "site:nature.com supplementary methods remote sensing 2024",
        "site:nature.com supplementary data atmospheric chemistry 2024",
        "site:nature.com supplementary table geothermal 2024",
    ],
    "Engineering / Applied science": [
        "site:nature.com/articles/s41467 supplementary table engineering 2024",
        "site:nature.com/articles/s41467 supplementary data biomedical device 2024",
        "site:nature.com supplementary table microfluidics fabrication 2024",
        "site:nature.com supplementary methods 3D printing 2024",
        "site:nature.com supplementary table wearable sensor 2024",
        "site:nature.com supplementary data robotics actuator 2024",
        "site:nature.com/articles/s41598 supplementary table engineering 2024",
        "site:nature.com supplementary methods characterization testing 2024",
        "site:nature.com supplementary table tissue engineering scaffold 2024",
        "site:journals.plos.org supporting information table biomedical 2024",
        "site:nature.com supplementary data water purification membrane 2024",
        "site:nature.com supplementary table energy harvesting 2024",
        "site:nature.com supplementary methods device fabrication process 2024",
        "site:nature.com supplementary table brain computer interface 2024",
        "site:nature.com supplementary data flexible electronics 2024",
        "site:journals.plos.org supporting information structural testing 2024",
        "site:nature.com supplementary table fuel cell hydrogen 2024",
        "site:nature.com supplementary methods optical fiber 2024",
        "site:nature.com supplementary table MEMS sensor 2024",
        "site:nature.com supplementary data drone remote sensing 2024",
    ],
}

# Trusted DOI prefixes
TRUSTED_DOI_PREFIXES = [
    "10.1038",   # Nature
    "10.1126",   # Science
    "10.1073",   # PNAS
    "10.1016",   # Elsevier / Cell Press
    "10.1128",   # ASM
    "10.1109",   # IEEE
    "10.1371",   # PLOS
    "10.48550",  # arXiv
    "10.1021",   # ACS
    "10.1002",   # Wiley
    "10.1146",   # Annual Reviews
    "10.1093",   # Oxford University Press
    "10.1103",   # APS (Physical Review)
    "10.1098",   # Royal Society
    "10.7554",   # eLife
    "10.1186",   # BMC / Springer
    "10.3389",   # Frontiers
    "10.1080",   # Taylor & Francis
    "10.1177",   # SAGE
    "10.1111",   # Wiley
    "10.1039",   # RSC
    "10.1088",   # IOP
    "10.1007",   # Springer
    "10.1029",   # AGU
]


# ============= Pydantic Models =============

class QAPair(BaseModel):
    question: str = Field(..., description="A verifiable factual question about supplementary material")
    answer: str = Field(..., description="A short, precise answer (under 100 characters)")
    source_doi: str = Field(..., description="The DOI of the source paper")
    key_passage: str = Field(..., description="The exact passage from supplementary material supporting the answer")
    domain: str = Field(..., description="The scientific domain")
    supp_type: str = Field(..., description="Type of supplementary material: table, methods, data, or text")


# ============= Reference Examples =============

def load_reference_examples() -> str:
    with open(REFERENCE_FILE, "r") as f:
        examples = json.load(f)

    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"""Example {i}:
Question: {ex['question']}
Answer: {ex['answer']}
Source DOI: {ex['source_doi']}
Key passage: {ex['key_passage']}
Domain: {ex['domain']}""")

    return "\n\n".join(parts)


# ============= Extraction Helpers =============

DOI_PATTERN = re.compile(r'(10\.\d{4,9}/[^\s,;"\'>]+)')

# Supplementary content markers
SUPP_MARKERS = [
    "supplementary table",
    "supplementary data",
    "supplementary methods",
    "supplementary information",
    "supplementary material",
    "supporting information",
    "s1 table", "s2 table", "s3 table", "s4 table", "s5 table",
    "s1 data", "s2 data",
    "s1 methods",
    "table s1", "table s2", "table s3", "table s4", "table s5",
    "additional file",
    "extended data table",
]


def has_supplementary_content(text: str) -> bool:
    """Check if text contains identifiable supplementary material content."""
    lower_text = text.lower()
    return any(marker in lower_text for marker in SUPP_MARKERS)


def extract_dois_from_text(text: str) -> list[str]:
    """Extract DOI strings from text content."""
    matches = DOI_PATTERN.findall(text)
    cleaned = []
    for m in matches:
        m = m.rstrip(".")
        if any(m.startswith(prefix) for prefix in TRUSTED_DOI_PREFIXES):
            cleaned.append(m)
    return list(set(cleaned))


def is_trusted_doi(doi: str) -> bool:
    """Check if a DOI is from a trusted publisher."""
    return any(doi.startswith(prefix) for prefix in TRUSTED_DOI_PREFIXES)


# ============= Core Pipeline =============

async def search_papers_for_domain(
    tavily_client: AsyncTavilyClient,
    domain: str,
    num_papers: int,
) -> list[dict]:
    """Search for papers with supplementary materials in a specific domain."""
    queries = DOMAIN_SEARCH_QUERIES.get(domain, [])
    all_results = []
    seen_urls = set()

    for query in queries:
        if len(all_results) >= num_papers * 4:  # Extra candidates for filtering
            break
        try:
            response = await tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=5,
            )
            results = response.get("results", [])
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "domain": domain,
                    })
            await asyncio.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"  Search error for '{query}': {e}")
            continue

    return all_results[:num_papers * 4]


def extract_supplementary_context(full_content: str) -> str | None:
    """
    Extract content relevant to supplementary materials from full article text.

    Strategy: Many papers discuss supplementary results inline in the main text
    (e.g., "as shown in Supplementary Table S1, the F-value was 71.76").
    We extract:
    1. Article header (title, abstract) for paper identification
    2. All paragraphs/passages that reference supplementary materials
    3. The supplementary information section itself if present
    """
    lower = full_content.lower()

    # Check if content references supplementary materials at all
    if not has_supplementary_content(full_content):
        return None

    # Part 1: Article header for paper identification
    article_header = full_content[:3000]

    # Part 2: Extract all passages that reference supplementary data
    # Find all positions where supplementary content is mentioned
    supp_passages = []
    search_terms = [
        "supplementary table", "supplementary method", "supplementary data",
        "supplementary information", "supplementary note",
        "supporting information", "s1 table", "s2 table", "s3 table",
        "s4 table", "s5 table", "s1 data", "s2 data",
        "table s1", "table s2", "table s3", "table s4", "table s5",
        "extended data table", "additional file",
    ]

    seen_ranges = []
    for term in search_terms:
        start = 0
        while True:
            idx = lower.find(term, start)
            if idx < 0:
                break

            # Extract context around this mention (paragraph-level)
            # Go back to find paragraph start and forward to paragraph end
            para_start = max(0, full_content.rfind("\n\n", 0, idx))
            para_end = full_content.find("\n\n", idx + len(term))
            if para_end < 0:
                para_end = min(len(full_content), idx + 1000)

            # Avoid overlapping with already-captured ranges
            overlap = False
            for s, e in seen_ranges:
                if para_start < e and para_end > s:
                    overlap = True
                    break

            if not overlap:
                passage = full_content[para_start:para_end].strip()
                if len(passage) > 30:  # Skip very short fragments
                    supp_passages.append(passage)
                    seen_ranges.append((para_start, para_end))

            start = idx + len(term)

    if not supp_passages:
        return None

    # Part 3: Also grab the formal supplementary info section if present
    supp_section = ""
    for pattern in [
        "## supplementary information",
        "## supporting information",
        "supplementary information\n",
        "supporting information\n",
    ]:
        idx = lower.find(pattern)
        if idx >= 0:
            section = full_content[idx:idx + 3000]
            supp_section = f"\n\n---SUPPLEMENTARY INFORMATION SECTION---\n\n{section}"
            break

    # Combine all parts
    passages_text = "\n\n---\n\n".join(supp_passages)

    # Truncate to fit in LLM context
    combined = f"{article_header}\n\n---PASSAGES REFERENCING SUPPLEMENTARY DATA---\n\n{passages_text}{supp_section}"

    if len(combined) > 14000:
        combined = combined[:14000]

    return combined


async def fetch_paper_content(
    tavily_client: AsyncTavilyClient,
    url: str,
) -> tuple[str | None, str | None]:
    """
    Fetch content from a paper URL using Tavily extract.
    Returns (full_content, supplementary_context) tuple.
    The supplementary_context contains article header + all passages referencing
    supplementary materials + the supplementary info section.
    """
    try:
        response = await tavily_client.extract(urls=[url])
        results = response.get("results", [])
        if not results:
            return None, None
        raw_content = results[0].get("raw_content", "")
        if len(raw_content) < 500:
            return None, None

        # Extract supplementary-relevant context
        supp_context = extract_supplementary_context(raw_content)

        return raw_content, supp_context
    except Exception as e:
        print(f"  Fetch error for {url}: {e}")
        return None, None


GENERATION_PROMPT_TEMPLATE = """You are creating a scientific QA benchmark focused exclusively on SUPPLEMENTARY MATERIALS of published papers.

Given paper content that includes supplementary information, generate ONE high-quality question-answer pair where the answer comes EXCLUSIVELY from supplementary data (tables, methods, data sections).

CRITICAL REQUIREMENTS:
1. The answer MUST come from SUPPLEMENTARY material only (Supplementary Tables, Supplementary Methods, Supplementary Data, Supporting Information, Extended Data Tables, S1-S10 Tables, Additional Files)
2. DO NOT generate questions about Supplementary Figures — only use tables, methods, and data
3. DO NOT generate questions about the main text, abstract, or main results/discussion sections
4. The answer must be a SPECIFIC, VERIFIABLE FACT: a primer sequence, parameter value, statistical value, sample size, concentration, software version, reagent catalog number, cell line name, antibody dilution, etc.
5. The question should include enough context to identify the paper (study description, author name, year, method, or organism)
6. The answer should be SHORT and PRECISE (under 100 characters)
7. You must extract the EXACT passage from the supplementary section that contains the answer

QUESTION SPECIFICITY:
The question MUST contain enough distinctive detail to uniquely identify the paper. Include specific terms like organism names, gene names, method names, author names, or unique experimental setups.

BAD example: "What primer was used for PCR?"
- Too generic. Many papers use PCR primers. Nothing identifies this paper.

GOOD example: "In the 2021 study demonstrating that TET1-mediated DNA hydroxymethylation regulates oligodendrocyte myelination in mice, what was the forward primer sequence used for qRT-PCR validation of the calcium transporter gene Itpr2?"
- Specific study description, specific gene name, specific technique — uniquely identifies the paper.

GOOD example: "How many sgRNAs were present in the CD69 dataset after filtering with GuideScan in the RELICS analysis method for CRISPR screens in the paper led by Patrick C. Fiaux?"
- Specific dataset name, method name, and first author — uniquely identifies the paper.

Here are reference examples of the style and quality we need:

{reference_examples}

Now generate a QA pair from this paper content. Focus ONLY on supplementary material sections:

URL: {url}
Domain: {domain}

Paper content (look for Supplementary Tables, Supporting Information, Extended Data, S1-S10 Tables, Supplementary Methods):
{content}

Respond with a JSON object with these exact fields:
- "question": the question (string) — must ask about supplementary data specifically
- "answer": short precise answer (string, under 100 chars)
- "source_doi": the DOI if found in the content, otherwise construct from the URL (string starting with "https://doi.org/")
- "key_passage": the exact verbatim passage from the supplementary section that supports the answer (string)
- "supp_type": one of "table", "methods", "data", "text" — which type of supplementary material the answer comes from

IMPORTANT:
- The key_passage MUST be copied verbatim from the paper content above and must be from a supplementary section
- The answer must appear in or be directly derivable from the key_passage
- If the content does not contain identifiable supplementary material, respond with {{"error": "no supplementary content found"}}
- If you cannot find a suitable specific fact from supplementary data, respond with {{"error": "no suitable supplementary fact found"}}"""


async def generate_qa_from_content(
    oai_client: AsyncOpenAI,
    content: str,
    url: str,
    domain: str,
    reference_examples: str,
) -> QAPair | None:
    """Use LLM to generate a QA pair from supplementary content."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        reference_examples=reference_examples,
        url=url,
        domain=domain,
        content=content,
    )

    try:
        response = await oai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or ""
        result = json.loads(result_text)

        if "error" in result:
            return None

        # Validate fields exist
        required_fields = ["question", "answer", "source_doi", "key_passage", "supp_type"]
        if not all(k in result for k in required_fields):
            return None

        # Validate answer length
        if len(result["answer"]) > 100:
            return None

        # Validate question length (must be specific enough)
        if len(result["question"]) < 50:
            return None

        # Validate key_passage references supplementary material
        if not has_supplementary_content(result["key_passage"]):
            # Check if key passage at least comes from a supplementary context
            # Sometimes the passage itself doesn't say "supplementary" but is from that section
            if not has_supplementary_content(result.get("question", "")):
                return None

        # Validate supp_type
        valid_types = ["table", "methods", "data", "text"]
        supp_type = result["supp_type"]
        if supp_type not in valid_types:
            supp_type = "text"  # Default fallback

        # Validate DOI is from trusted source
        doi = result["source_doi"]
        if doi.startswith("https://doi.org/"):
            doi_id = doi[len("https://doi.org/"):]
        elif doi.startswith("http://doi.org/"):
            doi_id = doi[len("http://doi.org/"):]
        else:
            doi_id = doi

        if not is_trusted_doi(doi_id):
            return None

        return QAPair(
            question=result["question"],
            answer=result["answer"],
            source_doi=doi if doi.startswith("https://doi.org/") else f"https://doi.org/{doi_id}",
            key_passage=result["key_passage"],
            domain=domain,
            supp_type=supp_type,
        )
    except Exception as e:
        print(f"  QA generation error: {e}")
        return None


async def process_domain(
    oai_client: AsyncOpenAI,
    tavily_client: AsyncTavilyClient,
    domain: str,
    target_count: int,
    reference_examples: str,
    existing_questions: set[str],
) -> list[QAPair]:
    """Process a single domain: search, fetch, generate QA pairs from supplementary data."""
    print(f"\n{'='*60}")
    print(f"Processing domain: {domain}")
    print(f"Target: {target_count} QA pairs")
    print(f"{'='*60}")

    # Step 1: Search for papers
    print(f"  Searching for papers with supplementary materials...")
    paper_results = await search_papers_for_domain(tavily_client, domain, target_count)
    print(f"  Found {len(paper_results)} candidate URLs")

    qa_pairs = []
    processed = 0
    skipped_no_supp = 0

    for paper in paper_results:
        if len(qa_pairs) >= target_count:
            break

        processed += 1
        url = paper["url"]
        print(f"  [{processed}] Fetching: {url[:80]}...")

        # Step 2: Fetch content and extract supplementary section
        full_content, supp_section = await fetch_paper_content(tavily_client, url)
        if not full_content:
            print(f"    -> No content extracted, skipping")
            continue

        # Step 3: Check for supplementary content
        if not supp_section:
            skipped_no_supp += 1
            print(f"    -> No supplementary section found, skipping")
            continue

        content = supp_section
        print(f"    -> Got {len(content)} chars of supplementary content")

        # Step 4: Generate QA
        qa = await generate_qa_from_content(
            oai_client, content, url, domain, reference_examples
        )
        if not qa:
            print(f"    -> Failed to generate QA from supplementary data, skipping")
            continue

        # Step 5: Dedup check
        if qa.question in existing_questions:
            print(f"    -> Duplicate question, skipping")
            continue

        existing_questions.add(qa.question)
        qa_pairs.append(qa)
        print(f"    -> Generated QA #{len(qa_pairs)}: {qa.question[:60]}...")
        print(f"       Answer: {qa.answer}")
        print(f"       Type: {qa.supp_type}")

        await asyncio.sleep(1)  # Rate limiting

    print(f"  Domain complete: {len(qa_pairs)}/{target_count} QA pairs generated")
    print(f"  Skipped {skipped_no_supp} papers with no supplementary content")
    return qa_pairs


def save_progress(qa_pairs: list[QAPair], path: Path) -> None:
    """Save current progress to parquet."""
    records = [
        {
            "question": qa.question,
            "answer": qa.answer,
            "source_doi": qa.source_doi,
            "key_passage": qa.key_passage,
            "domain": qa.domain,
            "supp_type": qa.supp_type,
        }
        for qa in qa_pairs
    ]
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"  Progress saved: {len(df)} records -> {path}")


def load_progress(path: Path) -> list[QAPair]:
    """Load previous progress from parquet."""
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [
        QAPair(
            question=row["question"],
            answer=row["answer"],
            source_doi=row["source_doi"],
            key_passage=row["key_passage"],
            domain=row["domain"],
            supp_type=row.get("supp_type", "text"),
        )
        for _, row in df.iterrows()
    ]


async def main():
    # Validate API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not openai_api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    if not tavily_api_key:
        print("ERROR: Set TAVILY_API_KEY environment variable")
        sys.exit(1)

    oai_client = AsyncOpenAI(api_key=openai_api_key)
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    # Load reference examples
    reference_examples = load_reference_examples()

    # Load any previous progress
    all_qa_pairs = load_progress(PROGRESS_PARQUET)
    existing_questions = {qa.question for qa in all_qa_pairs}

    if all_qa_pairs:
        print(f"Loaded {len(all_qa_pairs)} existing QA pairs from progress file")
        domain_counts = {}
        for qa in all_qa_pairs:
            domain_counts[qa.domain] = domain_counts.get(qa.domain, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c}")

    # Process each domain
    for domain in DOMAINS:
        # Count existing for this domain
        existing_count = sum(1 for qa in all_qa_pairs if qa.domain == domain)
        remaining = TARGET_PER_DOMAIN - existing_count

        if remaining <= 0:
            print(f"\nSkipping {domain} (already have {existing_count}/{TARGET_PER_DOMAIN})")
            continue

        domain_pairs = await process_domain(
            oai_client, tavily_client, domain, remaining, reference_examples, existing_questions
        )
        all_qa_pairs.extend(domain_pairs)

        # Save progress after each domain
        save_progress(all_qa_pairs, PROGRESS_PARQUET)

    # Final output
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total QA pairs: {len(all_qa_pairs)}")

    # Add IDs and save final output
    records = []
    for idx, qa in enumerate(all_qa_pairs):
        records.append({
            "id": f"suppqatrain_train_{idx}",
            "question": qa.question,
            "answer": qa.answer,
            "source_doi": qa.source_doi,
            "key_passage": qa.key_passage,
            "domain": qa.domain,
            "supp_type": qa.supp_type,
        })

    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved to {OUTPUT_PARQUET}")

    # Print domain distribution
    print(f"\nDomain distribution:")
    for domain, count in df["domain"].value_counts().items():
        print(f"  {domain}: {count}")

    # Print supp_type distribution
    print(f"\nSupplementary type distribution:")
    for stype, count in df["supp_type"].value_counts().items():
        print(f"  {stype}: {count}")

    # Print answer length stats
    print(f"\nAnswer length stats:")
    print(df["answer"].str.len().describe())

    # Print sample
    print(f"\nSample QA pairs:")
    for _, row in df.head(3).iterrows():
        print(f"\n  Q: {row['question']}")
        print(f"  A: {row['answer']}")
        print(f"  DOI: {row['source_doi']}")
        print(f"  Type: {row['supp_type']}")
        print(f"  Domain: {row['domain']}")


if __name__ == "__main__":
    asyncio.run(main())
