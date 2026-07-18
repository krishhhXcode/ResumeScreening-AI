"""High-level screening pipeline for ResumeScreening-AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modules.embedding_model import generate_embeddings
from modules.llm import generate_hiring_report
from modules.pdf_parser import extract_text_from_pdf
from modules.preprocessing import clean_text
from modules.ranking import rank_candidates
from modules.similarity import calculate_similarity


@dataclass(slots=True)
class ScreeningResult:
    """Container for a completed resume-screening run."""

    ranking: pd.DataFrame
    hiring_report: str
    top_candidate: str
    average_score: float
    candidate_count: int
def _extract_candidate_name(filename: str) -> str:
    """Return a readable candidate name from a resume filename.

    Args:
        filename: Resume PDF filename.

    Returns:
        Candidate name without the file extension.
    """
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()


def _prepare_resume(uploaded_resume) -> tuple[str, str]:
    """Extract and clean a single resume.

    Args:
        uploaded_resume: Streamlit UploadedFile.

    Returns:
        Tuple containing candidate name and cleaned resume text.
    """
    candidate_name = _extract_candidate_name(uploaded_resume.name)

    raw_text = extract_text_from_pdf(uploaded_resume)
    cleaned_text = clean_text(raw_text)

    return candidate_name, cleaned_text


def _prepare_job_description(job_description: str) -> str:
    """Validate and clean a job description.

    Args:
        job_description: Raw job description text.

    Returns:
        Cleaned job description.
    """
    return clean_text(job_description)
def run_screening(
    job_description: str,
    uploaded_resumes: list,
) -> ScreeningResult:
    """Execute the complete resume-screening pipeline.

    Args:
        job_description: Job description supplied by the recruiter.
        uploaded_resumes: List of uploaded resume PDF files.

    Returns:
        ScreeningResult containing ranking, report and dashboard metrics.

    Raises:
        ValueError: If no resumes are supplied.
    """
    if not uploaded_resumes:
        raise ValueError("Please upload at least one resume.")

    cleaned_job = _prepare_job_description(job_description)

    candidate_names: list[str] = []
    resume_texts: list[str] = []

    for uploaded_resume in uploaded_resumes:
        candidate_name, cleaned_resume = _prepare_resume(uploaded_resume)
        candidate_names.append(candidate_name)
        resume_texts.append(cleaned_resume)

    job_embedding = generate_embeddings([cleaned_job])

    resume_embeddings = generate_embeddings(resume_texts)

    similarity_scores = calculate_similarity(
        job_embedding,
        resume_embeddings,
    )

    ranking = rank_candidates(
        candidate_names,
        similarity_scores,
    )

    hiring_report = generate_hiring_report(
        cleaned_job,
        ranking,
    )

    top_candidate = str(ranking.iloc[0]["Candidate"])

    average_score = float(
        ranking["Match Percentage"].mean()
    )

    return ScreeningResult(
        ranking=ranking,
        hiring_report=hiring_report,
        top_candidate=top_candidate,
        average_score=average_score,
        candidate_count=len(candidate_names),
    )
__all__ = [
    "ScreeningResult",
    "run_screening",
]