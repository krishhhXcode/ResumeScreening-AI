"""Cosine-similarity utilities for job and resume embeddings."""

from __future__ import annotations

import numpy as np


__all__ = ["calculate_similarity"]


_EMBEDDING_DIMENSION = 384
_PRECISION_TOLERANCE = 1e-12


class _SimilarityComputationError(RuntimeError):
    """Raised when validated embeddings cannot be compared."""


def calculate_similarity(
    job_embedding: np.ndarray,
    resume_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between one job and multiple resume vectors.

    The job embedding must contain one all-MiniLM-L6-v2 vector, while resume
    embeddings must contain one vector per resume. The function does not sort
    or transform scores. It only corrects negligible floating-point drift just
    outside the expected 0-to-1 range.

    Args:
        job_embedding: One job-description embedding with shape ``(384,)`` or
            ``(1, 384)``.
        resume_embeddings: Resume embeddings with shape ``(n, 384)``.

    Returns:
        A one-dimensional NumPy array containing one score per resume.

    Raises:
        TypeError: If either embedding input is not a NumPy array.
        ValueError: If an array is empty, has an invalid shape, has mismatched
            dimensions, or contains non-finite/non-numeric values.
        _SimilarityComputationError: If scikit-learn is unavailable or cosine
            similarity cannot be computed for validated inputs.
    """
    _validate_array_type(job_embedding, "job_embedding")
    _validate_array_type(resume_embeddings, "resume_embeddings")

    job_vector = _validate_job_embedding(job_embedding)
    _validate_resume_embeddings(resume_embeddings, job_vector.shape[1])

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity(job_vector, resume_embeddings).ravel()
    except Exception as error:
        raise _SimilarityComputationError(
            "Unable to calculate cosine similarity for the supplied embeddings."
        ) from error

    return _correct_precision_drift(np.asarray(scores))


def _validate_array_type(embedding: object, name: str) -> None:
    """Ensure an embedding input is a NumPy array.

    Args:
        embedding: Candidate embedding input.
        name: Name used in validation messages.

    Raises:
        TypeError: If ``embedding`` is not a NumPy array.
    """
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray.")


def _validate_job_embedding(job_embedding: np.ndarray) -> np.ndarray:
    """Validate and reshape a single job-description embedding.

    Args:
        job_embedding: Candidate job embedding.

    Returns:
        A two-dimensional job embedding with shape ``(1, 384)``.

    Raises:
        ValueError: If the embedding is empty, malformed, or not numeric.
    """
    if job_embedding.size == 0:
        raise ValueError("job_embedding cannot be empty.")

    valid_shapes = {
        (_EMBEDDING_DIMENSION,),
        (1, _EMBEDDING_DIMENSION),
    }
    if job_embedding.shape not in valid_shapes:
        raise ValueError(
            "job_embedding must have shape (384,) or (1, 384) and contain "
            "exactly one embedding."
        )

    _validate_numeric_values(job_embedding, "job_embedding")
    return job_embedding.reshape(1, _EMBEDDING_DIMENSION)


def _validate_resume_embeddings(
    resume_embeddings: np.ndarray,
    expected_dimension: int,
) -> None:
    """Validate a batch of resume embeddings.

    Args:
        resume_embeddings: Candidate two-dimensional resume embedding array.
        expected_dimension: Number of dimensions in the job embedding.

    Raises:
        ValueError: If the array is empty, malformed, mismatched, or invalid.
    """
    if resume_embeddings.size == 0:
        raise ValueError("resume_embeddings cannot be empty.")
    if resume_embeddings.ndim != 2:
        raise ValueError("resume_embeddings must be a two-dimensional array.")
    if resume_embeddings.shape[1] != expected_dimension:
        raise ValueError(
            "resume_embeddings must have the same embedding dimension as "
            "job_embedding."
        )

    _validate_numeric_values(resume_embeddings, "resume_embeddings")


def _validate_numeric_values(embedding: np.ndarray, name: str) -> None:
    """Ensure an embedding contains finite, real numeric values.

    Args:
        embedding: Embedding array to inspect.
        name: Name used in validation messages.

    Raises:
        ValueError: If the array has non-numeric, complex, or non-finite data.
    """
    is_real_numeric = np.issubdtype(embedding.dtype, np.number) and not np.issubdtype(
        embedding.dtype,
        np.complexfloating,
    )
    if not is_real_numeric:
        raise ValueError(f"{name} must contain real numeric values.")
    if not np.isfinite(embedding).all():
        raise ValueError(f"{name} must contain only finite values.")


def _correct_precision_drift(scores: np.ndarray) -> np.ndarray:
    """Correct only insignificant floating-point drift outside [0, 1].

    Args:
        scores: Raw cosine-similarity scores.

    Returns:
        Scores with values within tolerance of 0 or 1 corrected to that bound.
    """
    corrected_scores = scores.copy()
    near_zero = (corrected_scores < 0) & (
        corrected_scores >= -_PRECISION_TOLERANCE
    )
    near_one = (corrected_scores > 1) & (
        corrected_scores <= 1 + _PRECISION_TOLERANCE
    )
    corrected_scores[near_zero] = 0.0
    corrected_scores[near_one] = 1.0
    return corrected_scores
