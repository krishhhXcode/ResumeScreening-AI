"""Candidate ranking utilities for precomputed similarity scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = ["rank_candidates"]


def rank_candidates(
    candidate_names: list[str],
    similarity_scores: np.ndarray,
) -> pd.DataFrame:
    """Create a descending candidate ranking from similarity scores.

    This function formats already-computed scores only. It does not calculate,
    normalize, or otherwise alter similarity values before sorting them.

    Args:
        candidate_names: Non-empty list of candidate names aligned with scores.
        similarity_scores: One-dimensional finite similarity score array.

    Returns:
        A DataFrame with rank, candidate, score, match percentage, and
        recommendation columns.

    Raises:
        TypeError: If candidate names are not a list of strings or scores are
            not a NumPy array.
        ValueError: If either input is empty, scores are not one-dimensional or
            finite real values, or input lengths do not match.
    """
    _validate_candidate_names(candidate_names)
    _validate_similarity_scores(similarity_scores)

    if len(candidate_names) != len(similarity_scores):
        raise ValueError(
            "candidate_names and similarity_scores must have the same length."
        )

    ranking = pd.DataFrame(
        {
            "Candidate": candidate_names,
            "Similarity Score": similarity_scores.astype(float, copy=False),
        }
    )
    ranking["Match Percentage"] = np.round(
        ranking["Similarity Score"] * 100,
        decimals=2,
    )
    ranking["Recommendation"] = ranking["Similarity Score"].map(
        _get_recommendation
    )
    ranking = ranking.sort_values(
        by="Similarity Score",
        ascending=False,
        kind="mergesort",
        ignore_index=True,
    )
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))

    return ranking[
        [
            "Rank",
            "Candidate",
            "Similarity Score",
            "Match Percentage",
            "Recommendation",
        ]
    ]


def _validate_candidate_names(candidate_names: list[str]) -> None:
    """Validate the candidate-name input.

    Args:
        candidate_names: Candidate list to validate.

    Raises:
        TypeError: If the input is not a list of strings.
        ValueError: If the candidate list is empty.
    """
    if not isinstance(candidate_names, list):
        raise TypeError("candidate_names must be a list of strings.")
    if not candidate_names:
        raise ValueError("candidate_names cannot be empty.")
    if any(not isinstance(name, str) for name in candidate_names):
        raise TypeError("Each candidate name must be a string.")


def _validate_similarity_scores(similarity_scores: np.ndarray) -> None:
    """Validate a similarity-score array.

    Args:
        similarity_scores: Candidate similarity scores to validate.

    Raises:
        TypeError: If scores are not provided as a NumPy array.
        ValueError: If scores are empty, non-real, non-finite, or not 1D.
    """
    if not isinstance(similarity_scores, np.ndarray):
        raise TypeError("similarity_scores must be a NumPy ndarray.")
    if similarity_scores.size == 0:
        raise ValueError("similarity_scores cannot be empty.")
    if similarity_scores.ndim != 1:
        raise ValueError("similarity_scores must be a one-dimensional array.")

    is_real_numeric = np.issubdtype(
        similarity_scores.dtype,
        np.number,
    ) and not np.issubdtype(similarity_scores.dtype, np.complexfloating)
    if not is_real_numeric:
        raise ValueError("similarity_scores must contain real numeric values.")
    if not np.isfinite(similarity_scores).all():
        raise ValueError("similarity_scores must contain only finite values.")


def _get_recommendation(similarity_score: float) -> str:
    """Map a similarity score to its candidate recommendation label.

    Args:
        similarity_score: Candidate similarity score.

    Returns:
        Recommendation label for the supplied score.
    """
    if similarity_score >= 0.85:
        return "Excellent Match"
    if similarity_score >= 0.70:
        return "Strong Match"
    if similarity_score >= 0.55:
        return "Good Match"
    if similarity_score >= 0.40:
        return "Moderate Match"
    return "Weak Match"
