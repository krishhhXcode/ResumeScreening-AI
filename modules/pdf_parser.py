"""Utilities for extracting readable text from resume PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TypeAlias

import pdfplumber


PDFSource: TypeAlias = str | Path | BinaryIO


class PDFParserError(Exception):
    """Base exception for PDF parsing failures."""


class InvalidPDFSourceError(PDFParserError):
    """Raised when the supplied PDF source is invalid or unavailable."""


class PDFTextExtractionError(PDFParserError):
    """Raised when text cannot be extracted from a PDF document."""


def extract_text_from_pdf(pdf_source: PDFSource) -> str:
    """Extract text from every text-bearing page in a PDF document.

    The function accepts either a filesystem path or an already-open binary
    stream, such as Streamlit's ``UploadedFile``. It preserves page order and
    separates extracted pages with a blank line. Text cleaning belongs to the
    preprocessing module and is intentionally not performed here.

    Args:
        pdf_source: Path to a PDF file or an open binary PDF stream.

    Returns:
        The non-empty text extracted from the document.

    Raises:
        InvalidPDFSourceError: If the path is missing, is not a PDF, or the
            stream cannot be read.
        PDFTextExtractionError: If the PDF is malformed, password-protected,
            unreadable, or contains no extractable text.
    """
    source = _validate_pdf_source(pdf_source)
    original_position = _rewind_stream(source)

    try:
        with pdfplumber.open(source) as pdf:
            page_text = [
                text.strip()
                for page in pdf.pages
                if (text := page.extract_text()) and text.strip()
            ]
    except Exception as error:
        raise PDFTextExtractionError(
            "Unable to extract text from the supplied PDF. "
            "Ensure it is a readable, non-password-protected PDF."
        ) from error
    finally:
        _restore_stream_position(source, original_position)

    if not page_text:
        raise PDFTextExtractionError(
            "The PDF contains no extractable text. It may be image-only or empty."
        )

    return "\n\n".join(page_text)


def _validate_pdf_source(pdf_source: PDFSource) -> PDFSource:
    """Validate a PDF source before it is passed to pdfplumber.

    Args:
        pdf_source: Candidate file path or binary stream.

    Returns:
        A validated source usable by ``pdfplumber.open``.

    Raises:
        InvalidPDFSourceError: If the source is unsupported or unavailable.
    """
    if isinstance(pdf_source, (str, Path)):
        path = Path(pdf_source)
        if path.suffix.lower() != ".pdf":
            raise InvalidPDFSourceError("The selected file must have a .pdf extension.")
        if not path.is_file():
            raise InvalidPDFSourceError(f"PDF file not found: {path}")
        return path

    if not all(hasattr(pdf_source, attribute) for attribute in ("read", "seek")):
        raise InvalidPDFSourceError(
            "PDF source must be a file path or a seekable binary stream."
        )

    return pdf_source


def _rewind_stream(pdf_source: PDFSource) -> int | None:
    """Rewind a binary stream and return its original position when available."""
    if isinstance(pdf_source, (str, Path)):
        return None

    try:
        original_position = pdf_source.tell()
        pdf_source.seek(0)
    except (AttributeError, OSError) as error:
        raise InvalidPDFSourceError(
            "The PDF stream must support tell() and seek()."
        ) from error

    return original_position


def _restore_stream_position(pdf_source: PDFSource, position: int | None) -> None:
    """Restore a caller-owned stream to its original position when possible."""
    if position is None or isinstance(pdf_source, (str, Path)):
        return

    try:
        pdf_source.seek(position)
    except OSError:
        pass
