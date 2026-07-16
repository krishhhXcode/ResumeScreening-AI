"""Text preprocessing utilities for extracted resume content."""

from __future__ import annotations

import re
import unicodedata


__all__ = ["clean_text"]


_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>()\[\]{}\"']+",
    flags=re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(
    r"(?<![\w.-])"
    r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}"
    r"(?![\w-])",
    flags=re.IGNORECASE,
)
_PROGRAMMING_LANGUAGE_PATTERN = re.compile(
    r"(?<!\w)c\+\+(?![\w+])|(?<!\w)c#(?![\w#])",
    flags=re.IGNORECASE,
)
_HASHED_TOKEN_PATTERN = re.compile(r"(?<!\w)#[\w-]+", flags=re.UNICODE)
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|\+)?"
    r"(?![\w.])"
)
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_PLACEHOLDER_PREFIX = "zzpreservedtoken"


def clean_text(text: str) -> str:
    """Normalize extracted resume text while retaining meaningful tokens.

    Unicode is normalized using NFKC, text is lowercased, and whitespace is
    standardized. Emails, URLs, C++, C#, hashtags, and numeric expressions
    are protected before unnecessary punctuation is removed. Stopword removal,
    stemming, and lemmatization are intentionally outside this module's scope.

    Args:
        text: Raw text extracted from a resume PDF.

    Returns:
        Cleaned, lowercase text ready for downstream embedding and comparison.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If ``text`` is empty, whitespace-only, or has no usable
            characters after cleaning.
    """
    if not isinstance(text, str):
        raise TypeError("Text to clean must be a string.")

    normalized_text = unicodedata.normalize("NFKC", text).lower()
    normalized_text = _normalize_whitespace(normalized_text)
    if not normalized_text:
        raise ValueError("Text to clean cannot be empty or whitespace-only.")

    protected_text, protected_tokens = _protect_meaningful_tokens(normalized_text)
    cleaned_text = _remove_unnecessary_punctuation(protected_text)
    cleaned_text = _restore_protected_tokens(cleaned_text, protected_tokens)
    cleaned_text = _normalize_whitespace(cleaned_text)

    if not cleaned_text:
        raise ValueError("Text does not contain any usable characters.")

    return cleaned_text


def _normalize_whitespace(text: str) -> str:
    """Standardize line endings and collapse excess horizontal whitespace.

    Args:
        text: Text to normalize.

    Returns:
        Text with trimmed lines and runs of blank lines reduced to one newline.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _protect_meaningful_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Replace punctuation-sensitive tokens with temporary placeholders.

    Args:
        text: Normalized lowercase text.

    Returns:
        The text containing placeholders and their original token mapping.
    """
    token_mapping: dict[str, str] = {}
    text = _replace_matches(text, _URL_PATTERN, token_mapping, trim_url=True)
    text = _replace_matches(text, _EMAIL_PATTERN, token_mapping)
    text = _replace_matches(text, _PROGRAMMING_LANGUAGE_PATTERN, token_mapping)
    text = _replace_matches(text, _HASHED_TOKEN_PATTERN, token_mapping)
    text = _replace_matches(text, _NUMBER_PATTERN, token_mapping)
    return text, token_mapping


def _replace_matches(
    text: str,
    pattern: re.Pattern[str],
    token_mapping: dict[str, str],
    *,
    trim_url: bool = False,
) -> str:
    """Replace pattern matches with collision-safe, alphanumeric placeholders.

    Args:
        text: Text containing tokens to protect.
        pattern: Regular expression matching tokens to preserve.
        token_mapping: Mutable placeholder-to-token mapping.
        trim_url: Whether to remove sentence-ending punctuation from a URL.

    Returns:
        Text with matching tokens replaced by placeholders.
    """

    def replace_match(match: re.Match[str]) -> str:
        token = match.group(0)
        trailing_punctuation = ""
        if trim_url:
            token, trailing_punctuation = _split_trailing_url_punctuation(token)

        placeholder = _create_placeholder(text, token_mapping)
        token_mapping[placeholder] = token
        return f"{placeholder}{trailing_punctuation}"

    return pattern.sub(replace_match, text)


def _split_trailing_url_punctuation(url: str) -> tuple[str, str]:
    """Separate sentence-ending punctuation from a matched URL.

    Args:
        url: URL matched within text.

    Returns:
        The URL without trailing punctuation and the removed suffix.
    """
    trimmed_url = url.rstrip(".,;:!?")
    return trimmed_url, url[len(trimmed_url) :]


def _create_placeholder(text: str, token_mapping: dict[str, str]) -> str:
    """Create a placeholder that cannot collide with existing text.

    Args:
        text: Current text being processed.
        token_mapping: Existing placeholder-to-token mapping.

    Returns:
        A unique alphanumeric placeholder.
    """
    index = len(token_mapping)
    placeholder = f"{_PLACEHOLDER_PREFIX}{index}zz"
    while placeholder in text or placeholder in token_mapping:
        index += 1
        placeholder = f"{_PLACEHOLDER_PREFIX}{index}zz"
    return placeholder


def _remove_unnecessary_punctuation(text: str) -> str:
    """Replace non-word punctuation with spaces without affecting placeholders.

    Args:
        text: Text containing protected-token placeholders.

    Returns:
        Text with unnecessary punctuation removed.
    """
    text = text.replace("_", " ")
    return _PUNCTUATION_PATTERN.sub(" ", text)


def _restore_protected_tokens(text: str, token_mapping: dict[str, str]) -> str:
    """Restore meaningful tokens after punctuation removal.

    Args:
        text: Text containing placeholders.
        token_mapping: Placeholder-to-original-token mapping.

    Returns:
        Text with all placeholders restored to their original tokens.
    """
    for placeholder, token in token_mapping.items():
        text = text.replace(placeholder, token)
    return text
