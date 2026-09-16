from typing import Any

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate structured metadata about a Pandas DataFrame.

    The profiler only inspects the dataset.
    It does not modify the original DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot profile an empty DataFrame.")

    profile: dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_details": {},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for column in df.columns:
        series = df[column]

        details: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100),
                2,
            ),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": series.dropna().head(5).tolist(),
        }

        if pd.api.types.is_numeric_dtype(series):
            details["statistics"] = {
                "mean": _safe_float(series.mean()),
                "median": _safe_float(series.median()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "std": _safe_float(series.std()),
            }

        elif pd.api.types.is_object_dtype(series):
            details["top_values"] = (
                series.value_counts(dropna=True)
                .head(5)
                .to_dict()
            )

        profile["column_details"][str(column)] = details

    return profile


def _safe_float(value: Any) -> float | None:
    """
    Convert numerical values into JSON-friendly Python floats.
    """

    if pd.isna(value):
        return None

    return float(value)