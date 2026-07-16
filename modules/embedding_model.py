"""Lazy loading and embedding generation for the resume screening pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


__all__ = ["get_embedding_model", "generate_embeddings"]


_MODEL_NAME = "all-MiniLM-L6-v2"


class _EmbeddingModelLoadError(RuntimeError):
    """Raised when the sentence-transformer model cannot be loaded."""


class _EmbeddingGenerationError(RuntimeError):
    """Raised when a validated text batch cannot be embedded."""


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Return the cached all-MiniLM-L6-v2 sentence-transformer model.

    The model and its dependencies are imported only on the first call. Later
    calls reuse the same instance for the lifetime of the Python process.

    Returns:
        A loaded ``SentenceTransformer`` model instance.

    Raises:
        _EmbeddingModelLoadError: If sentence-transformers is unavailable or
            the model cannot be initialized.
    """
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(_MODEL_NAME)
    except Exception as error:
        raise _EmbeddingModelLoadError(
            f"Unable to load the embedding model '{_MODEL_NAME}'."
        ) from error


def generate_embeddings(texts: list[str], normalize: bool = True) -> np.ndarray:
    """Generate semantic embeddings for a validated batch of resume text.

    Each text item is stripped before it is passed to the model. When
    ``normalize`` is ``True``, embeddings are L2-normalized by
    sentence-transformers, which makes them directly suitable for cosine
    similarity comparisons.

    Args:
        texts: Non-empty list of non-empty text strings to encode.
        normalize: Whether to request L2-normalized embeddings from the model.

    Returns:
        A two-dimensional NumPy array containing one embedding per input text.

    Raises:
        TypeError: If ``texts`` is not a list, an item is not a string, or
            ``normalize`` is not a boolean.
        ValueError: If ``texts`` is empty or an item is empty after stripping.
        _EmbeddingModelLoadError: If the model cannot be loaded.
        _EmbeddingGenerationError: If encoding fails or returns an invalid
            embedding shape.
    """
    cleaned_texts = _validate_and_strip_texts(texts)
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean.")

    try:
        embeddings = get_embedding_model().encode(
            cleaned_texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    except _EmbeddingModelLoadError:
        raise
    except Exception as error:
        raise _EmbeddingGenerationError(
            "Unable to generate embeddings for the provided text batch."
        ) from error

    embeddings_array = np.asarray(embeddings)
    if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(cleaned_texts):
        raise _EmbeddingGenerationError(
            "The embedding model returned an unexpected embedding shape."
        )

    return embeddings_array


def _validate_and_strip_texts(texts: list[str]) -> list[str]:
    """Validate an embedding batch and remove surrounding whitespace.

    Args:
        texts: Candidate batch of texts to encode.

    Returns:
        A validated list whose text values have been stripped.

    Raises:
        TypeError: If the batch is not a list or contains non-string values.
        ValueError: If the batch is empty or contains an empty text value.
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be provided as a list of strings.")
    if not texts:
        raise ValueError("texts must contain at least one string.")

    cleaned_texts: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be a string.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError(f"texts[{index}] cannot be empty or whitespace-only.")
        cleaned_texts.append(cleaned_text)

    return cleaned_texts
