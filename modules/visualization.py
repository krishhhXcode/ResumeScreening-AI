"""Matplotlib visualizations for ranked candidate screening results."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


__all__ = [
    "create_match_bar_chart",
    "create_recommendation_pie_chart",
    "create_score_distribution",
]


_REQUIRED_COLUMNS = {
    "Candidate",
    "Similarity Score",
    "Match Percentage",
    "Recommendation",
}
_RECOMMENDATION_COLORS = {
    "Excellent Match": "#2e7d32",
    "Strong Match": "#1565c0",
    "Good Match": "#00838f",
    "Moderate Match": "#f9a825",
    "Weak Match": "#c62828",
}
_FALLBACK_COLORS = ("#5e81ac", "#8fbcbb", "#d08770", "#b48ead", "#a3be8c")


def create_match_bar_chart(results: pd.DataFrame) -> Figure:
    """Create a labeled bar chart of candidate match percentages.

    Args:
        results: Ranked candidate results containing the required columns.

    Returns:
        A Matplotlib figure with candidate names on the x-axis and match
        percentages on the y-axis.

    Raises:
        TypeError: If ``results`` is not a pandas DataFrame.
        ValueError: If required columns or usable chart data are missing.
    """
    _validate_results(results)

    candidate_names = results["Candidate"].tolist()
    match_percentages = results["Match Percentage"].to_numpy(dtype=float)
    figure = Figure(figsize=(max(8.0, len(candidate_names) * 1.1), 5.8))
    axis = figure.add_subplot(111)

    bars = axis.bar(
        candidate_names,
        match_percentages,
        color="#2563eb",
        edgecolor="#1d4ed8",
        linewidth=0.7,
    )
    axis.bar_label(
        bars,
        labels=[f"{score:.2f}%" for score in match_percentages],
        padding=3,
        fontsize=9,
    )
    axis.set_title("Candidate Match Percentages", pad=14, weight="bold")
    axis.set_xlabel("Candidate")
    axis.set_ylabel("Match Percentage (%)")
    axis.set_ylim(*_get_percentage_limits(match_percentages))
    _rotate_candidate_labels(axis, candidate_names)
    _apply_axis_style(axis)
    figure.tight_layout()

    return figure


def create_recommendation_pie_chart(results: pd.DataFrame) -> Figure:
    """Create a pie chart showing the distribution of recommendations.

    Args:
        results: Ranked candidate results containing the required columns.

    Returns:
        A Matplotlib figure showing recommendation-label counts and percentages.

    Raises:
        TypeError: If ``results`` is not a pandas DataFrame.
        ValueError: If required columns or usable chart data are missing.
    """
    _validate_results(results)

    recommendation_counts = results["Recommendation"].value_counts()
    labels = recommendation_counts.index.tolist()
    colors = [
        _RECOMMENDATION_COLORS.get(
            label,
            _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)],
        )
        for index, label in enumerate(labels)
    ]

    figure = Figure(figsize=(7.2, 6.0))
    axis = figure.add_subplot(111)
    axis.pie(
        recommendation_counts.to_numpy(),
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
    )
    axis.set_title("Candidate Recommendation Distribution", pad=14, weight="bold")
    axis.axis("equal")
    figure.tight_layout()

    return figure


def create_score_distribution(results: pd.DataFrame) -> Figure:
    """Create a histogram of candidate match percentages.

    Args:
        results: Ranked candidate results containing the required columns.

    Returns:
        A Matplotlib figure showing the distribution of match percentages.

    Raises:
        TypeError: If ``results`` is not a pandas DataFrame.
        ValueError: If required columns or usable chart data are missing.
    """
    _validate_results(results)

    match_percentages = results["Match Percentage"].to_numpy(dtype=float)
    bin_count = _get_histogram_bin_count(len(match_percentages))
    figure = Figure(figsize=(8.0, 5.5))
    axis = figure.add_subplot(111)
    axis.hist(
        match_percentages,
        bins=bin_count,
        color="#2563eb",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.9,
    )
    axis.set_title("Match Percentage Distribution", pad=14, weight="bold")
    axis.set_xlabel("Match Percentage (%)")
    axis.set_ylabel("Candidate Count")
    axis.set_xlim(*_get_percentage_limits(match_percentages))
    _apply_axis_style(axis)
    figure.tight_layout()

    return figure


def _validate_results(results: pd.DataFrame) -> None:
    """Validate that ranking results contain usable visualization data.

    Args:
        results: Candidate ranking results to validate.

    Raises:
        TypeError: If ``results`` is not a pandas DataFrame.
        ValueError: If required columns, rows, or valid plotting values are
            missing.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")

    missing_columns = sorted(_REQUIRED_COLUMNS.difference(results.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"results is missing required columns: {missing}.")
    if results.empty:
        raise ValueError("results must contain at least one candidate.")

    _validate_text_column(results["Candidate"], "Candidate")
    _validate_text_column(results["Recommendation"], "Recommendation")
    _validate_numeric_column(results["Similarity Score"], "Similarity Score")
    _validate_numeric_column(results["Match Percentage"], "Match Percentage")


def _validate_text_column(column: pd.Series, name: str) -> None:
    """Validate a required non-empty text column.

    Args:
        column: Series containing text values.
        name: Name used in validation messages.

    Raises:
        ValueError: If the series has missing, blank, or non-string values.
    """
    if column.isna().any() or not column.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(f"{name} must contain non-empty string values.")


def _validate_numeric_column(column: pd.Series, name: str) -> None:
    """Validate a required finite, real numeric column.

    Args:
        column: Series containing numeric values.
        name: Name used in validation messages.

    Raises:
        ValueError: If the series is non-numeric or has non-finite values.
    """
    if not pd.api.types.is_numeric_dtype(column):
        raise ValueError(f"{name} must contain numeric values.")

    if column.isna().any():
        raise ValueError(f"{name} must contain only finite real values.")

    values = column.to_numpy()
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain only finite real values.")
    if not np.isfinite(values.astype(float)).all():
        raise ValueError(f"{name} must contain only finite real values.")


def _get_percentage_limits(values: np.ndarray) -> tuple[float, float]:
    """Calculate a readable axis range for percentage chart values.

    Args:
        values: Finite match-percentage values.

    Returns:
        Lower and upper chart limits with modest padding around the values.
    """
    lower_limit = min(0.0, float(values.min()) - 5.0)
    upper_limit = max(100.0, float(values.max()) + 5.0)
    return lower_limit, upper_limit


def _rotate_candidate_labels(axis: Axes, candidate_names: list[str]) -> None:
    """Rotate x-axis labels when names or result count require more space.

    Args:
        axis: Matplotlib axis containing candidate labels.
        candidate_names: Candidate names displayed on the x-axis.
    """
    should_rotate = len(candidate_names) > 5 or max(
        len(name) for name in candidate_names
    ) > 12
    if should_rotate:
        axis.tick_params(axis="x", labelrotation=35)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")


def _apply_axis_style(axis: Axes) -> None:
    """Apply a consistent, restrained style to a Matplotlib axis.

    Args:
        axis: Matplotlib axis to style.
    """
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _get_histogram_bin_count(candidate_count: int) -> int:
    """Select a readable histogram bin count for a candidate population.

    Args:
        candidate_count: Number of candidates represented in the histogram.

    Returns:
        A bin count between five and ten.
    """
    return min(10, max(5, math.ceil(math.sqrt(candidate_count))))
