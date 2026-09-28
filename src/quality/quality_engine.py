from typing import Any

import pandas as pd

from src.quality.missing import MissingValueDetector


class DataQualityEngine:
    """
    Deterministic data-quality analysis engine.

    The engine detects quality issues but never modifies
    the input dataframe.
    """

    def __init__(self):
        self.missing_detector = MissingValueDetector()

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        if df.empty:
            raise ValueError("Cannot analyze an empty dataset.")

        issues = []

        # Missing values
        issues.extend(self.missing_detector.analyze(df))

        # Duplicate rows
        issues.extend(self._duplicates(df))

        # Datatype inconsistencies
        issues.extend(self._datatype_issues(df))

        # Categorical inconsistencies
        issues.extend(self._categorical_inconsistencies(df))

        # Invalid dates
        issues.extend(self._date_issues(df))

        # Uniqueness violations
        issues.extend(self._uniqueness_issues(df))

        # Statistical outliers
        issues.extend(self._outliers(df))

        summary = self._build_summary(issues)

        return {
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "issues": issues,
            "summary": summary,
        }

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    @staticmethod
    def _duplicates(df: pd.DataFrame) -> list[dict[str, Any]]:
        duplicate_count = int(df.duplicated().sum())

        if duplicate_count == 0:
            return []

        percentage = (duplicate_count / len(df)) * 100

        return [
            {
                "type": "duplicate_rows",
                "column": None,
                "severity": "warning",
                "message": (
                    f"Dataset contains {duplicate_count} duplicate rows "
                    f"({percentage:.2f}%)."
                ),
                "evidence": {
                    "duplicate_count": duplicate_count,
                    "duplicate_percentage": round(percentage, 2),
                    "total_rows": len(df),
                },
            }
        ]

    # ---------------------------------------------------------
    # Datatype issues
    # ---------------------------------------------------------

    @staticmethod
    def _datatype_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
        issues = []

        for column in df.columns:
            series = df[column].dropna()

            if series.empty:
                continue

            if series.dtype == "object":
                numeric_values = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                numeric_ratio = numeric_values.notna().mean()

                if numeric_ratio >= 0.9:
                    issues.append(
                        {
                            "type": "datatype_inconsistency",
                            "column": column,
                            "severity": "warning",
                            "message": (
                                f"Column '{column}' is stored as text "
                                "but contains mostly numeric values."
                            ),
                            "evidence": {
                                "current_dtype": str(series.dtype),
                                "numeric_ratio": round(
                                    float(numeric_ratio),
                                    4,
                                ),
                            },
                        }
                    )

        return issues

    # ---------------------------------------------------------
    # Categorical consistency
    # ---------------------------------------------------------

    @staticmethod
    def _categorical_inconsistencies(
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        issues = []

        for column in df.columns:
            series = df[column].dropna()

            if series.dtype != "object":
                continue

            values = series.astype(str)
            normalized = values.str.strip().str.lower()

            grouped = pd.DataFrame(
                {
                    "original": values,
                    "normalized": normalized,
                }
            )

            for normalized_value, group in grouped.groupby(
                "normalized"
            ):
                variants = group["original"].unique()

                if len(variants) > 1:
                    issues.append(
                        {
                            "type": "categorical_inconsistency",
                            "column": column,
                            "severity": "warning",
                            "message": (
                                f"Column '{column}' contains multiple "
                                f"representations of "
                                f"'{normalized_value}'."
                            ),
                            "evidence": {
                                "normalized_value": normalized_value,
                                "variants": variants.tolist(),
                            },
                        }
                    )

        return issues

    # ---------------------------------------------------------
    # Date validation
    # ---------------------------------------------------------

    @staticmethod
    def _date_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
        issues = []

        for column in df.columns:
            series = df[column].dropna()

            if series.empty:
                continue

            column_name = str(column).lower()

            if not any(
                keyword in column_name
                for keyword in (
                    "date",
                    "time",
                    "timestamp",
                    "created",
                    "updated",
                )
            ):
                continue

            if pd.api.types.is_datetime64_any_dtype(series):
                continue

            converted = pd.to_datetime(
                series,
                errors="coerce",
            )

            invalid_count = int(converted.isna().sum())

            if invalid_count == 0:
                continue

            percentage = (invalid_count / len(series)) * 100

            issues.append(
                {
                    "type": "invalid_dates",
                    "column": column,
                    "severity": "warning",
                    "message": (
                        f"Column '{column}' contains "
                        f"{invalid_count} values that cannot be "
                        "parsed as dates."
                    ),
                    "evidence": {
                        "invalid_count": invalid_count,
                        "invalid_percentage": round(
                            percentage,
                            2,
                        ),
                    },
                }
            )

        return issues

    # ---------------------------------------------------------
    # Uniqueness / candidate key detection
    # ---------------------------------------------------------

    @staticmethod
    def _uniqueness_issues(
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        issues = []

        for column in df.columns:
            series = df[column]

            if series.isna().all():
                continue

            unique_count = int(
                series.nunique(dropna=True)
            )

            non_null_count = int(series.notna().sum())

            if non_null_count == 0:
                continue

            unique_ratio = unique_count / non_null_count

            column_name = str(column).lower()

            looks_like_identifier = any(
                keyword in column_name
                for keyword in (
                    "id",
                    "identifier",
                    "code",
                    "key",
                )
            )

            if looks_like_identifier and unique_ratio < 1:
                duplicate_count = (
                    non_null_count - unique_count
                )

                issues.append(
                    {
                        "type": "uniqueness_violation",
                        "column": column,
                        "severity": "warning",
                        "message": (
                            f"Identifier-like column '{column}' "
                            f"contains {duplicate_count} duplicate "
                            "values."
                        ),
                        "evidence": {
                            "unique_count": unique_count,
                            "non_null_count": non_null_count,
                            "unique_ratio": round(
                                unique_ratio,
                                4,
                            ),
                            "duplicate_values": duplicate_count,
                        },
                    }
                )

        return issues

    # ---------------------------------------------------------
    # Statistical outliers
    # ---------------------------------------------------------

    @staticmethod
    def _outliers(df: pd.DataFrame) -> list[dict[str, Any]]:
        issues = []

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:
            series = df[column].dropna()

            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outliers = series[
                (series < lower_bound)
                | (series > upper_bound)
            ]

            count = int(len(outliers))

            if count == 0:
                continue

            issues.append(
                {
                    "type": "outliers",
                    "column": column,
                    "severity": "info",
                    "message": (
                        f"Column '{column}' contains "
                        f"{count} statistical outliers."
                    ),
                    "evidence": {
                        "outlier_count": count,
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "q1": float(q1),
                        "q3": float(q3),
                    },
                }
            )

        return issues

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    @staticmethod
    def _build_summary(
        issues: list[dict[str, Any]],
    ) -> dict[str, int]:
        critical = sum(
            issue["severity"] == "critical"
            for issue in issues
        )

        warning = sum(
            issue["severity"] == "warning"
            for issue in issues
        )

        info = sum(
            issue["severity"] == "info"
            for issue in issues
        )

        return {
            "total_issues": len(issues),
            "critical": critical,
            "warning": warning,
            "info": info,
        }