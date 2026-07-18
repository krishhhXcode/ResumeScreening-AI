"""Skill extraction and keyword matching utilities."""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "extract_skills",
    "missing_skills",
    "keyword_match_percentage",
]

# Expand this list whenever you want.
SKILL_DATABASE: Final[set[str]] = {
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node",
    "express",
    "django",
    "flask",
    "fastapi",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "sqlite",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "plotly",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "opencv",
    "langchain",
    "llm",
    "rag",
    "transformers",
    "huggingface",
    "gemini",
    "openai",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "git",
    "github",
    "linux",
    "streamlit",
    "power bi",
    "excel",
    "tableau",
}

_WORD_PATTERN = re.compile(r"\b[\w#+.-]+\b")


def extract_skills(text: str) -> set[str]:
    """Extract known skills from text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = text.lower()

    found: set[str] = set()

    for skill in SKILL_DATABASE:
        if " " in skill:
            if skill in normalized:
                found.add(skill)

    tokens = set(_WORD_PATTERN.findall(normalized))

    for skill in SKILL_DATABASE:
        if " " not in skill and skill in tokens:
            found.add(skill)

    return found


def missing_skills(
    job_skills: set[str],
    resume_skills: set[str],
) -> list[str]:
    """Return skills required by the job but missing from the resume."""
    return sorted(job_skills - resume_skills)


def keyword_match_percentage(
    job_skills: set[str],
    resume_skills: set[str],
) -> float:
    """Calculate keyword match percentage."""
    if not job_skills:
        return 100.0

    matched = len(job_skills & resume_skills)
    return round((matched / len(job_skills)) * 100, 2)