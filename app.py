"""Streamlit application for AI-assisted resume screening and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from modules.embedding_model import generate_embeddings, get_embedding_model
from modules.llm import generate_hiring_report
from modules.pdf_parser import extract_text_from_pdf
from modules.preprocessing import clean_text
from modules.ranking import rank_candidates
from modules.similarity import calculate_similarity
from modules.visualization import (
    create_match_bar_chart,
    create_recommendation_pie_chart,
    create_score_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIRECTORY = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIRECTORY / "logo.png"
STYLESHEET_PATH = ASSETS_DIRECTORY / "styles.css"
_SCREENING_RUN_KEY = "screening_run"


@dataclass(frozen=True)
class _ScreeningRun:
    """Data retained for rendering a completed screening run."""

    ranking_results: pd.DataFrame
    hiring_report: str | None
    hiring_report_error: str | None


def _is_available_file(path: Path) -> bool:
    """Return whether an asset path exists and contains data.

    Args:
        path: Asset path to check.

    Returns:
        True when the path is a non-empty file; otherwise, False.
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _load_stylesheet(path: Path) -> None:
    """Load a local stylesheet into the Streamlit page when available.

    Args:
        path: Path to the CSS stylesheet.
    """
    if not _is_available_file(path):
        return

    try:
        stylesheet = path.read_text(encoding="utf-8")
    except OSError:
        return

    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


PAGE_ICON = str(LOGO_PATH) if _is_available_file(LOGO_PATH) else "📄"
st.set_page_config(
    page_title="ResumeScreening-AI",
    page_icon=PAGE_ICON,
    layout="wide",
)
_load_stylesheet(STYLESHEET_PATH)


@st.cache_resource(show_spinner=False)
def _get_cached_embedding_model() -> object:
    """Load the shared sentence-transformer model once per Streamlit process.

    Returns:
        The cached sentence-transformer model instance.
    """
    return get_embedding_model()


def _render_hero_section() -> None:
    """Render the branded project hero section."""
    with st.container(border=True):
        logo_column, title_column = st.columns([1, 9], vertical_alignment="center")

        with logo_column:
            if _is_available_file(LOGO_PATH):
                st.image(str(LOGO_PATH), width=800)
            else:
                st.markdown("## AI")

        with title_column:
            st.title("ResumeScreening-AI")
            st.caption(
                "Semantic candidate screening and explainable hiring insights."
            )
            st.markdown(
                "`Sentence Transformers`  `Semantic Search`  `Gemini AI`"
            )


def _render_upload_workspace() -> tuple[str, str, Any | None, list[Any], bool]:
    """Render inputs for a job description and multiple resume PDFs.

    Returns:
        The job input mode, pasted job text, uploaded job PDF, uploaded resumes,
        and whether the screening action was requested.
    """
    st.header("Upload Workspace")
    job_column, resume_column = st.columns(2, gap="large")
    uploaded_job_pdf: Any | None = None

    with job_column:
        with st.container(border=True):
            st.subheader("Job Description")
            input_mode = st.radio(
                "Choose a job description source",
                options=("Paste Text", "Upload PDF"),
                horizontal=True,
            )
            pasted_job_description = st.text_area(
                "Job description",
                placeholder="Paste the role requirements and responsibilities.",
                height=220,
                key="job_description_text",
                disabled=input_mode != "Paste Text",
            )

            if input_mode == "Upload PDF":
                uploaded_job_pdf = st.file_uploader(
                    "Upload job description PDF",
                    type=["pdf"],
                    key="job_description_pdf",
                )
                if uploaded_job_pdf is not None:
                    st.caption(f"Selected file: {uploaded_job_pdf.name}")

    with resume_column:
        with st.container(border=True):
            st.subheader("Candidate Resumes")
            uploaded_resumes = st.file_uploader(
                "Upload one or more resume PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="resume_pdfs",
            )
            if uploaded_resumes:
                st.markdown("**Selected resumes**")
                for uploaded_resume in uploaded_resumes:
                    st.markdown(f"- `{uploaded_resume.name}`")
            else:
                st.caption("Uploaded filenames will appear here.")

    button_column = st.columns([1, 2, 1])[1]
    run_screening = button_column.button(
        "Run AI Screening",
        width="stretch",
        help="Extract text, compare semantic embeddings, and rank candidates.",
        type="primary",
    )

    return (
        input_mode,
        pasted_job_description,
        uploaded_job_pdf,
        list(uploaded_resumes or []),
        run_screening,
    )


def _render_processing_timeline() -> None:
    """Render the processing timeline for the screening workflow."""
    st.header("Processing Timeline")
    steps = (
        ("1", "Parse PDFs", "Extract job and resume text"),
        ("2", "Prepare Text", "Clean content for semantic analysis"),
        ("3", "Compare Matches", "Generate similarity scores"),
        ("4", "Review Insights", "Present ranking and hiring report"),
    )

    timeline_columns = st.columns(len(steps), gap="small")
    for column, (step_number, title, description) in zip(
        timeline_columns,
        steps,
    ):
        with column:
            with st.container(border=True):
                st.caption(f"STEP {step_number}")
                st.markdown(f"**{title}**")
                st.caption(description)


def _resolve_job_description(
    input_mode: str,
    pasted_job_description: str,
    uploaded_job_pdf: Any | None,
) -> str:
    """Read the job description from the active input source.

    Args:
        input_mode: Selected job-description input mode.
        pasted_job_description: Value from the job text area.
        uploaded_job_pdf: Optional uploaded job-description PDF.

    Returns:
        Raw job-description text extracted from the selected source.

    Raises:
        ValueError: If the selected source has no usable job description.
        RuntimeError: If the uploaded job-description PDF cannot be read.
    """
    if input_mode == "Paste Text":
        if not pasted_job_description.strip():
            raise ValueError("Paste a job description before running screening.")
        return pasted_job_description

    if input_mode == "Upload PDF":
        if uploaded_job_pdf is None:
            raise ValueError("Upload a job description PDF before running screening.")
        try:
            return extract_text_from_pdf(uploaded_job_pdf)
        except Exception as error:
            raise RuntimeError(
                "The uploaded job description PDF could not be processed."
            ) from error

    raise ValueError("Select a valid job-description input mode.")


def _extract_resume_texts(
    uploaded_resumes: list[Any],
) -> tuple[list[str], list[str]]:
    """Extract text and candidate names from uploaded resume PDFs.

    Args:
        uploaded_resumes: Uploaded resume files supplied by Streamlit.

    Returns:
        Candidate names and their raw extracted resume text, in matching order.

    Raises:
        ValueError: If no resume files were supplied.
        RuntimeError: If a resume cannot be extracted.
    """
    if not uploaded_resumes:
        raise ValueError("Upload at least one resume PDF before running screening.")

    candidate_names: list[str] = []
    resume_texts: list[str] = []
    for index, uploaded_resume in enumerate(uploaded_resumes, start=1):
        file_name = getattr(uploaded_resume, "name", f"candidate_{index}.pdf")
        candidate_name = Path(file_name).stem.strip() or f"Candidate {index}"
        try:
            resume_text = extract_text_from_pdf(uploaded_resume)
        except Exception as error:
            raise RuntimeError(
                f"Resume '{file_name}' could not be processed."
            ) from error

        candidate_names.append(candidate_name)
        resume_texts.append(resume_text)

    return candidate_names, resume_texts


def _run_screening_pipeline(
    input_mode: str,
    pasted_job_description: str,
    uploaded_job_pdf: Any | None,
    uploaded_resumes: list[Any],
) -> tuple[str, pd.DataFrame]:
    """Execute extraction, preprocessing, embedding, similarity, and ranking.

    Args:
        input_mode: Selected job-description input mode.
        pasted_job_description: Text entered in the job description text area.
        uploaded_job_pdf: Optional job-description PDF upload.
        uploaded_resumes: Resume PDF uploads to screen.

    Returns:
        Cleaned job description and the ranked candidate DataFrame.

    Raises:
        ValueError: If a required user input is missing or unusable.
        RuntimeError: If a PDF cannot be extracted or pipeline processing fails.
    """
    raw_job_description = _resolve_job_description(
        input_mode,
        pasted_job_description,
        uploaded_job_pdf,
    )
    candidate_names, raw_resume_texts = _extract_resume_texts(uploaded_resumes)

    try:
        cleaned_job_description = clean_text(raw_job_description)
        cleaned_resume_texts = [
            clean_text(resume_text) for resume_text in raw_resume_texts
        ]
    except Exception as error:
        raise RuntimeError("Text preprocessing failed for the uploaded documents.") from error

    try:
        _get_cached_embedding_model()
        job_embedding = generate_embeddings([cleaned_job_description])
        resume_embeddings = generate_embeddings(cleaned_resume_texts)
        similarity_scores = calculate_similarity(job_embedding, resume_embeddings)
        ranking_results = rank_candidates(candidate_names, similarity_scores)
    except Exception as error:
        raise RuntimeError(
            "Semantic screening could not be completed. Check model dependencies "
            "and try again."
        ) from error

    return cleaned_job_description, ranking_results


def _generate_optional_hiring_report(
    job_description: str,
    ranking_results: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """Generate Gemini insights without discarding completed screening results.

    Args:
        job_description: Cleaned job description used in the screening run.
        ranking_results: Completed candidate ranking table.

    Returns:
        The generated report and an optional user-facing generation error.
    """
    try:
        return generate_hiring_report(job_description, ranking_results), None
    except Exception as error:
        return None, _error_message(error)


def _error_message(error: Exception) -> str:
    """Return a concise, user-facing error message.

    Args:
        error: Exception raised during a pipeline step.

    Returns:
        Exception text or its class name when no text is available.
    """
    return str(error).strip() or error.__class__.__name__


def _current_screening_run() -> _ScreeningRun | None:
    """Return the current completed screening run from session state.

    Returns:
        The active screening result, or None when no run has completed.
    """
    screening_run = st.session_state.get(_SCREENING_RUN_KEY)
    if isinstance(screening_run, _ScreeningRun):
        return screening_run
    return None


def _render_metrics(ranking_results: pd.DataFrame) -> None:
    """Render summary metrics for a completed ranking.

    Args:
        ranking_results: Completed candidate ranking table.
    """
    top_match = float(ranking_results["Match Percentage"].iloc[0])
    average_match = float(ranking_results["Match Percentage"].mean())
    strong_matches = int((ranking_results["Similarity Score"] >= 0.70).sum())
    metrics = (
        ("Candidates Screened", str(len(ranking_results))),
        ("Average Match", f"{average_match:.2f}%"),
        ("Top Match", f"{top_match:.2f}%"),
        ("Strong Matches", str(strong_matches)),
    )

    metric_columns = st.columns(4, gap="medium")
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            with st.container(border=True):
                st.metric(label, value)


def _render_analytics(ranking_results: pd.DataFrame) -> None:
    """Render the ranking charts and show any visualization failure in the UI.

    Args:
        ranking_results: Completed candidate ranking table.
    """
    try:
        bar_chart = create_match_bar_chart(ranking_results)
        pie_chart = create_recommendation_pie_chart(ranking_results)
        distribution_chart = create_score_distribution(ranking_results)
    except Exception as error:
        st.error(f"Analytics could not be displayed: {_error_message(error)}")
        return

    first_chart_column, second_chart_column = st.columns(2, gap="large")
    with first_chart_column:
        st.pyplot(bar_chart, width="stretch")
    with second_chart_column:
        st.pyplot(pie_chart, width="stretch")
    st.pyplot(distribution_chart, width="stretch")


def _render_dashboard(screening_run: _ScreeningRun | None) -> None:
    """Render dashboard placeholders or completed screening results.

    Args:
        screening_run: Completed screening data, if available.
    """
    st.header("Screening Dashboard")
    if screening_run is None:
        _render_dashboard_placeholder()
        return

    ranking_results = screening_run.ranking_results
    _render_metrics(ranking_results)
    st.divider()
    candidate_column, ranking_column = st.columns([1, 2], gap="large")
    top_candidate = ranking_results.iloc[0]

    with candidate_column:
        with st.container(border=True):
            st.subheader("Top Candidate")
            st.markdown(f"### {top_candidate['Candidate']}")
            st.metric("Match Percentage", f"{top_candidate['Match Percentage']:.2f}%")
            st.caption(top_candidate["Recommendation"])

    with ranking_column:
        with st.container(border=True):
            st.subheader("Candidate Ranking")
            st.dataframe(
                ranking_results,
                hide_index=True,
                width="stretch",
            )

    analytics_column, report_column = st.columns(2, gap="large")
    with analytics_column:
        with st.container(border=True):
            st.subheader("Analytics")
            _render_analytics(ranking_results)

    with report_column:
        with st.container(border=True):
            st.subheader("AI Hiring Report")
            if screening_run.hiring_report is not None:
                st.markdown(screening_run.hiring_report)
            else:
                st.error(
                    "Gemini hiring report unavailable: "
                    f"{screening_run.hiring_report_error}"
                )

    try:
        csv_data = ranking_results.to_csv(index=False).encode("utf-8")
    except Exception as error:
        st.error(f"CSV export could not be prepared: {_error_message(error)}")
        return

    download_column = st.columns([1, 2, 1])[1]
    download_column.download_button(
        "Download CSV",
        data=csv_data,
        file_name="candidate_ranking.csv",
        mime="text/csv",
        width="stretch",
    )


def _render_dashboard_placeholder() -> None:
    """Render dashboard placeholders before a screening run completes."""
    metric_labels = (
        "Candidates Screened",
        "Average Match",
        "Top Match",
        "Strong Matches",
    )
    metric_columns = st.columns(4, gap="medium")
    for column, label in zip(metric_columns, metric_labels):
        with column:
            with st.container(border=True):
                st.metric(label, "—")

    st.divider()
    candidate_column, ranking_column = st.columns([1, 2], gap="large")
    with candidate_column:
        with st.container(border=True):
            st.subheader("Top Candidate")
            st.markdown("**No screening results yet**")
            st.caption(
                "The strongest candidate and recommendation will appear here."
            )

    with ranking_column:
        with st.container(border=True):
            st.subheader("Candidate Ranking")
            st.dataframe(
                {
                    "Rank": [],
                    "Candidate": [],
                    "Match Percentage": [],
                    "Recommendation": [],
                },
                hide_index=True,
                width="stretch",
            )
            st.caption("The ranked candidate table will appear after screening.")

    analytics_column, report_column = st.columns(2, gap="large")
    with analytics_column:
        with st.container(border=True):
            st.subheader("Analytics")
            st.info("Match and recommendation charts will appear here.")

    with report_column:
        with st.container(border=True):
            st.subheader("AI Hiring Report")
            st.info("Gemini-powered hiring insights will appear here.")


def _render_footer() -> None:
    """Render the application footer."""
    st.divider()
    st.caption(
        "Built with Python, Streamlit, Sentence Transformers, and Gemini AI."
    )


def main() -> None:
    """Render the application and execute screening when requested."""
    _render_hero_section()
    st.divider()
    (
        input_mode,
        pasted_job_description,
        uploaded_job_pdf,
        uploaded_resumes,
        run_screening,
    ) = _render_upload_workspace()

    if run_screening:
        st.session_state.pop(_SCREENING_RUN_KEY, None)
        try:
            with st.spinner("Parsing documents and computing semantic matches..."):
                cleaned_job_description, ranking_results = _run_screening_pipeline(
                    input_mode,
                    pasted_job_description,
                    uploaded_job_pdf,
                    uploaded_resumes,
                )
        except Exception as error:
            st.error(f"Screening could not be completed: {_error_message(error)}")
        else:
            with st.spinner("Generating Gemini hiring insights..."):
                hiring_report, hiring_report_error = _generate_optional_hiring_report(
                    cleaned_job_description,
                    ranking_results,
                )

            st.session_state[_SCREENING_RUN_KEY] = _ScreeningRun(
                ranking_results=ranking_results,
                hiring_report=hiring_report,
                hiring_report_error=hiring_report_error,
            )
            st.success("Screening completed successfully.")

    st.divider()
    _render_processing_timeline()
    st.divider()
    _render_dashboard(_current_screening_run())
    _render_footer()


if __name__ == "__main__":
    main()
