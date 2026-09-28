from typing import Any

import pandas as pd


class DataCleaner:
    """
    Deterministic engine for applying approved data-preparation operations.

    The cleaner does not make decisions about what should be cleaned.
    It only executes explicit preparation operations.
    """

    def clean(
        self,
        df: pd.DataFrame,
        preparation_plan: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply a preparation plan to a copy of the dataset.

        Parameters
        ----------
        df:
            Original dataset.

        preparation_plan:
            Structured plan describing approved transformations.

        Returns
        -------
        pd.DataFrame
            Prepared dataset.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        if not preparation_plan:
            raise ValueError(
                "Preparation plan cannot be empty."
            )

        cleaned_df = df.copy()

        operations = preparation_plan.get(
            "operations",
            [],
        )

        for operation in operations:

            operation_type = operation.get("type")

            if operation_type == "rename_columns":
                cleaned_df = self._rename_columns(
                    cleaned_df,
                    operation,
                )

            elif operation_type == "drop_duplicates":
                cleaned_df = self._drop_duplicates(
                    cleaned_df,
                )

            elif operation_type == "fill_missing":
                cleaned_df = self._fill_missing(
                    cleaned_df,
                    operation,
                )

            elif operation_type == "convert_datetime":
                cleaned_df = self._convert_datetime(
                    cleaned_df,
                    operation,
                )

            elif operation_type == "strip_strings":
                cleaned_df = self._strip_strings(
                    cleaned_df,
                    operation,
                )

            else:
                raise ValueError(
                    f"Unsupported cleaning operation: "
                    f"{operation_type}"
                )

        return cleaned_df

    @staticmethod
    def _rename_columns(
        df: pd.DataFrame,
        operation: dict[str, Any],
    ) -> pd.DataFrame:

        mapping = operation.get("mapping", {})

        if not isinstance(mapping, dict):
            raise ValueError(
                "Column rename mapping must be a dictionary."
            )

        return df.rename(columns=mapping)

    @staticmethod
    def _drop_duplicates(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        return df.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _fill_missing(
        df: pd.DataFrame,
        operation: dict[str, Any],
    ) -> pd.DataFrame:

        column = operation.get("column")
        strategy = operation.get("strategy")

        if column not in df.columns:
            raise ValueError(
                f"Column not found: {column}"
            )

        if strategy == "mean":
            df[column] = df[column].fillna(
                df[column].mean()
            )

        elif strategy == "median":
            df[column] = df[column].fillna(
                df[column].median()
            )

        elif strategy == "mode":
            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(
                    mode.iloc[0]
                )

        elif strategy == "value":
            value = operation.get("value")

            df[column] = df[column].fillna(value)

        elif strategy == "ffill":
            df[column] = df[column].ffill()

        elif strategy == "bfill":
            df[column] = df[column].bfill()

        else:
            raise ValueError(
                f"Unsupported missing-value strategy: {strategy}"
            )

        return df

    @staticmethod
    def _convert_datetime(
        df: pd.DataFrame,
        operation: dict[str, Any],
    ) -> pd.DataFrame:

        column = operation.get("column")

        if column not in df.columns:
            raise ValueError(
                f"Column not found: {column}"
            )

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

        return df

    @staticmethod
    def _strip_strings(
        df: pd.DataFrame,
        operation: dict[str, Any],
    ) -> pd.DataFrame:

        columns = operation.get("columns")

        if not columns:
            columns = df.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()

        for column in columns:

            if column not in df.columns:
                continue

            df[column] = df[column].apply(
                lambda value:
                value.strip()
                if isinstance(value, str)
                else value
            )

        return df