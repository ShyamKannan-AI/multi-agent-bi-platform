from typing import Any

import pandas as pd


class MissingValueDetector:
    """
    Detects missing values in dataframe columns.

    This detector only reports missing-value evidence.
    It does not modify the dataframe.
    """

    def analyze(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Detect missing values in the dataframe.

        Returns
        -------
        list[dict]
            One issue record for each column containing missing values.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        if df.empty:
            raise ValueError("Cannot analyze an empty dataset.")

        issues = []

        for column in df.columns:
            missing_count = int(df[column].isna().sum())

            if missing_count == 0:
                continue

            missing_percentage = (missing_count / len(df)) * 100

            issues.append(
                {
                    "type": "missing_values",
                    "column": column,
                    "severity": "warning",
                    "message": (
                        f"Column '{column}' contains "
                        f"{missing_count} missing values "
                        f"({missing_percentage:.2f}%)."
                    ),
                    "evidence": {
                        "missing_count": missing_count,
                        "missing_percentage": round(
                            missing_percentage, 2
                        ),
                        "total_rows": len(df),
                    },
                }
            )

        return issues