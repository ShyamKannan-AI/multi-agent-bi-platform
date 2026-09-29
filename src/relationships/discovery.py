from __future__ import annotations

from itertools import combinations

import pandas as pd


class RelationshipDiscovery:
    """
    Discovers potential relationships between multiple datasets.

    The engine is deterministic and never mutates the input DataFrames.
    """

    def discover(self, datasets: dict[str, pd.DataFrame]) -> dict:
        if not isinstance(datasets, dict):
            raise TypeError("datasets must be a dictionary.")

        if len(datasets) < 2:
            raise ValueError("At least two datasets are required.")

        for dataset_name, dataframe in datasets.items():
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(
                    f"Dataset '{dataset_name}' must be a pandas DataFrame."
                )

        relationships = []

        for (left_name, left_df), (right_name, right_df) in combinations(
            datasets.items(), 2
        ):
            relationships.extend(
                self._compare_datasets(
                    left_name,
                    left_df,
                    right_name,
                    right_df,
                )
            )

        return {
            "status": "success",
            "dataset_count": len(datasets),
            "relationship_count": len(relationships),
            "relationships": relationships,
            "summary": {
                "relationship_count": len(relationships),
                "message": (
                    f"Discovered {len(relationships)} potential "
                    f"relationship(s) across {len(datasets)} datasets."
                ),
            },
        }

    def _compare_datasets(
        self,
        left_name: str,
        left_df: pd.DataFrame,
        right_name: str,
        right_df: pd.DataFrame,
    ) -> list[dict]:
        relationships = []

        for left_column in left_df.columns:
            for right_column in right_df.columns:
                if not self._columns_are_compatible(
                    left_df[left_column],
                    right_df[right_column],
                ):
                    continue

                name_similarity = self._name_similarity(
                    left_column,
                    right_column,
                )

                left_unique_ratio = self._unique_ratio(
                    left_df[left_column]
                )

                right_unique_ratio = self._unique_ratio(
                    right_df[right_column]
                )

                containment = self._containment(
                    left_df[left_column],
                    right_df[right_column],
                )

                reverse_containment = self._containment(
                    right_df[right_column],
                    left_df[left_column],
                )

                confidence = self._confidence(
                    name_similarity=name_similarity,
                    containment=containment,
                    reverse_containment=reverse_containment,
                    left_unique_ratio=left_unique_ratio,
                    right_unique_ratio=right_unique_ratio,
                )

                if confidence < 0.60:
                    continue

                relationship_type = self._relationship_type(
                    left_unique_ratio=left_unique_ratio,
                    right_unique_ratio=right_unique_ratio,
                    containment=containment,
                    reverse_containment=reverse_containment,
                )

                relationships.append(
                    {
                        "left_dataset": left_name,
                        "right_dataset": right_name,
                        "left_column": left_column,
                        "right_column": right_column,
                        "relationship": relationship_type,
                        "relationship_type": relationship_type,
                        "confidence": round(confidence, 4),
                        "containment": round(containment, 4),
                        "reverse_containment": round(reverse_containment, 4),
                        "left_unique_ratio": round(left_unique_ratio, 4),
                        "right_unique_ratio": round(right_unique_ratio, 4),
                        "name_similarity": round(name_similarity, 4),
                    }
                )

        return relationships

    @staticmethod
    def _columns_are_compatible(
        left: pd.Series,
        right: pd.Series,
    ) -> bool:
        """
        Check whether two columns have compatible basic data types.
        """

        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
            right
        ):
            return True

        if pd.api.types.is_datetime64_any_dtype(
            left
        ) and pd.api.types.is_datetime64_any_dtype(right):
            return True

        if pd.api.types.is_object_dtype(left) and pd.api.types.is_object_dtype(
            right
        ):
            return True

        if pd.api.types.is_string_dtype(left) and pd.api.types.is_string_dtype(
            right
        ):
            return True

        return False

    @staticmethod
    def _unique_ratio(series: pd.Series) -> float:
        if len(series) == 0:
            return 0.0

        return float(series.nunique(dropna=True) / len(series))

    @staticmethod
    def _containment(
        source: pd.Series,
        target: pd.Series,
    ) -> float:
        """
        Percentage of non-null source values that occur in target.
        """

        source_values = set(source.dropna().tolist())
        target_values = set(target.dropna().tolist())

        if not source_values:
            return 0.0

        return len(source_values.intersection(target_values)) / len(source_values)

    @staticmethod
    def _name_similarity(left: str, right: str) -> float:
        """
        Lightweight deterministic column-name similarity.
        """

        left_normalized = (
            str(left)
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        right_normalized = (
            str(right)
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if left_normalized == right_normalized:
            return 1.0

        if (
            left_normalized in right_normalized
            or right_normalized in left_normalized
        ):
            return 0.75

        return 0.0

    @staticmethod
    def _confidence(
        *,
        name_similarity: float,
        containment: float,
        reverse_containment: float,
        left_unique_ratio: float,
        right_unique_ratio: float,
    ) -> float:
        """
        Calculate relationship confidence from deterministic evidence.
        """

        overlap_score = max(containment, reverse_containment)

        confidence = (
            0.35 * name_similarity
            + 0.40 * overlap_score
            + 0.15 * min(left_unique_ratio, right_unique_ratio)
            + 0.10 * max(left_unique_ratio, right_unique_ratio)
        )

        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _relationship_type(
        *,
        left_unique_ratio: float,
        right_unique_ratio: float,
        containment: float,
        reverse_containment: float,
    ) -> str:
        """
        Infer the cardinality/reference relationship.
        """

        left_is_unique = left_unique_ratio >= 0.95
        right_is_unique = right_unique_ratio >= 0.95

        if left_is_unique and right_is_unique:
            return "one_to_one"

        if (
            left_is_unique
            and reverse_containment >= 0.95
        ):
            return "one_to_many"

        if (
            right_is_unique
            and containment >= 0.95
        ):
            return "many_to_one"

        if containment >= 0.95 or reverse_containment >= 0.95:
            return "reference"

        return "possible_relationship"


class ReferentialIntegrityChecker:
    """
    Checks whether values in a child/reference column exist
    in the corresponding parent column.
    """

    def check(
        self,
        child_df: pd.DataFrame,
        child_column: str,
        parent_df: pd.DataFrame,
        parent_column: str,
    ) -> dict:
        if not isinstance(child_df, pd.DataFrame):
            raise TypeError("child_df must be a pandas DataFrame.")

        if not isinstance(parent_df, pd.DataFrame):
            raise TypeError("parent_df must be a pandas DataFrame.")

        if child_column not in child_df.columns:
            raise ValueError(
                f"Column '{child_column}' not found in child dataset."
            )

        if parent_column not in parent_df.columns:
            raise ValueError(
                f"Column '{parent_column}' not found in parent dataset."
            )

        child_values = set(child_df[child_column].dropna().tolist())
        parent_values = set(parent_df[parent_column].dropna().tolist())

        orphan_values = child_values - parent_values

        total_values = len(child_values)

        if total_values == 0:
            integrity_percentage = 100.0
        else:
            valid_values = total_values - len(orphan_values)
            integrity_percentage = (
                valid_values / total_values
            ) * 100

        return {
            "status": "success",
            "child_column": child_column,
            "parent_column": parent_column,
            "orphan_values": sorted(
                orphan_values,
                key=lambda value: str(value),
            ),
            "orphan_count": len(orphan_values),
            "integrity_percentage": round(
                integrity_percentage,
                2,
            ),
            "is_valid": len(orphan_values) == 0,
        }