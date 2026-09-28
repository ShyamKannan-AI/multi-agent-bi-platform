from typing import Any

import pandas as pd


class PreparationAuditor:
    """
    Generates deterministic audit information describing
    how a dataset changed during preparation.
    """

    def audit(
        self,
        original_df: pd.DataFrame,
        prepared_df: pd.DataFrame,
        preparation_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compare the original and prepared datasets.
        """

        if not isinstance(original_df, pd.DataFrame):
            raise TypeError(
                "original_df must be a Pandas DataFrame."
            )

        if not isinstance(prepared_df, pd.DataFrame):
            raise TypeError(
                "prepared_df must be a Pandas DataFrame."
            )

        if not preparation_plan:
            raise ValueError(
                "Preparation plan cannot be empty."
            )

        original_missing = int(
            original_df.isna().sum().sum()
        )

        prepared_missing = int(
            prepared_df.isna().sum().sum()
        )

        original_duplicates = int(
            original_df.duplicated().sum()
        )

        prepared_duplicates = int(
            prepared_df.duplicated().sum()
        )

        operations = preparation_plan.get(
            "operations",
            [],
        )

        operation_summary = self._summarize_operations(
            operations
        )

        return {
            "original": {
                "rows": len(original_df),
                "columns": len(original_df.columns),
                "missing_values": original_missing,
                "duplicate_rows": original_duplicates,
            },
            "prepared": {
                "rows": len(prepared_df),
                "columns": len(prepared_df.columns),
                "missing_values": prepared_missing,
                "duplicate_rows": prepared_duplicates,
            },
            "changes": {
                "rows_removed": (
                    len(original_df)
                    - len(prepared_df)
                ),
                "columns_added": len(
                    set(prepared_df.columns)
                    - set(original_df.columns)
                ),
                "columns_removed": len(
                    set(original_df.columns)
                    - set(prepared_df.columns)
                ),
                "missing_values_resolved": (
                    original_missing
                    - prepared_missing
                ),
                "duplicates_resolved": (
                    original_duplicates
                    - prepared_duplicates
                ),
            },
            "operations": operation_summary,
        }

    @staticmethod
    def _summarize_operations(
        operations: list[dict[str, Any]],
    ) -> dict[str, int]:
        """
        Count how many times each preparation operation
        was requested.
        """

        summary: dict[str, int] = {}

        for operation in operations:

            operation_type = operation.get("type")

            if not operation_type:
                continue

            summary[operation_type] = (
                summary.get(operation_type, 0) + 1
            )

        return summary