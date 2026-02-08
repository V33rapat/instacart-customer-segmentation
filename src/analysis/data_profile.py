from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    output_dir: Path
    merged_filename: str = "merged_data.csv"

    @property
    def merged_path(self) -> Path:
        return self.data_dir / self.merged_filename


@dataclass(frozen=True)
class ProfileConfig:
    categorical_threshold: int = 30
    numeric_quantiles: tuple[float, float, float] = (0.25, 0.5, 0.75)
    outlier_iqr_multiplier: float = 1.5


def load_dataset(paths: DatasetPaths) -> pd.DataFrame:
    if not paths.merged_path.exists():
        raise FileNotFoundError(
            f"Merged dataset not found at {paths.merged_path}. "
            "Run datapipline.py or place merged_data.csv in data/."
        )
    return pd.read_csv(paths.merged_path)


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df)).mul(100).round(2)
    summary = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_pct": missing_pct,
            "dtype": df.dtypes.astype(str),
        }
    )
    summary = summary.sort_values(by=["missing_count", "missing_pct"], ascending=False)
    return summary


def detect_numeric_outliers(df: pd.DataFrame, config: ProfileConfig) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame(columns=["column", "outlier_count", "outlier_pct"])

    outlier_rows: list[dict[str, Any]] = []
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            outlier_rows.append(
                {"column": column, "outlier_count": 0, "outlier_pct": 0.0}
            )
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_rows.append(
                {"column": column, "outlier_count": 0, "outlier_pct": 0.0}
            )
            continue
        lower = q1 - config.outlier_iqr_multiplier * iqr
        upper = q3 + config.outlier_iqr_multiplier * iqr
        outliers = series[(series < lower) | (series > upper)]
        outlier_pct = (len(outliers) / len(series) * 100) if len(series) else 0.0
        outlier_rows.append(
            {
                "column": column,
                "outlier_count": len(outliers),
                "outlier_pct": round(outlier_pct, 2),
            }
        )

    return pd.DataFrame(outlier_rows).sort_values(
        by=["outlier_count", "outlier_pct"], ascending=False
    )


def feature_distributions(
    df: pd.DataFrame, config: ProfileConfig
) -> dict[str, pd.DataFrame]:
    distributions: dict[str, pd.DataFrame] = {}

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        quantiles = list(config.numeric_quantiles)
        numeric_summary = numeric_df.describe(percentiles=quantiles).T
        distributions["numeric"] = numeric_summary

    categorical_candidates = df.select_dtypes(exclude=[np.number])
    if not categorical_candidates.empty:
        categorical_frames = []
        for column in categorical_candidates.columns:
            value_counts = df[column].value_counts(dropna=False)
            if len(value_counts) <= config.categorical_threshold:
                categorical_frames.append(value_counts.rename(column).to_frame("count"))
        if categorical_frames:
            distributions["categorical"] = pd.concat(
                categorical_frames, axis=0
            ).reset_index(names=["value"])

    return distributions


def suggest_feature_improvements(df: pd.DataFrame) -> list[str]:
    suggestions = [
        "Aggregate orders to user-level features (order frequency, basket size mean/median).",
        "Create recency features (days since last order, order number percentile).",
        "Derive temporal patterns (order hour/day-of-week entropy, preferred shopping window).",
        "Compute product diversity metrics (unique aisles/departments per user).",
        "Capture reorder behavior (reorder ratio, repeat purchase rate by department).",
    ]

    if "days_since_prior_order" in df.columns:
        suggestions.append(
            "Use days_since_prior_order to compute user-level interpurchase variability."
        )
    if "order_hour_of_day" in df.columns:
        suggestions.append(
            "Bucket order_hour_of_day into time-of-day segments to reduce sparsity."
        )
    if "order_dow" in df.columns:
        suggestions.append("Encode order_dow as cyclical features (sin/cos).")

    return suggestions


def write_outputs(
    paths: DatasetPaths,
    missing_summary: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    outliers: pd.DataFrame,
    suggestions: list[str],
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    missing_summary.to_csv(paths.output_dir / "missing_values.csv")
    outliers.to_csv(paths.output_dir / "outlier_summary.csv", index=False)

    for name, table in distributions.items():
        table.to_csv(paths.output_dir / f"feature_distribution_{name}.csv")

    suggestions_path = paths.output_dir / "feature_improvements.md"
    suggestions_path.write_text(
        "# Feature Improvement Ideas\n\n"
        + "\n".join(f"- {item}" for item in suggestions)
        + "\n",
        encoding="utf-8",
    )


def profile_dataset(paths: DatasetPaths, config: ProfileConfig) -> None:
    df = load_dataset(paths)
    missing_summary = summarize_missing_values(df)
    distributions = feature_distributions(df, config)
    outliers = detect_numeric_outliers(df, config)
    suggestions = suggest_feature_improvements(df)
    write_outputs(paths, missing_summary, distributions, outliers, suggestions)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = DatasetPaths(
        data_dir=project_root / "data",
        output_dir=project_root / "reports" / "data_profile",
    )
    config = ProfileConfig()
    profile_dataset(paths, config)


if __name__ == "__main__":
    main()
