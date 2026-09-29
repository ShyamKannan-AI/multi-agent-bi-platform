from typing import Any, Callable

import pandas as pd

from src.processing.pipeline import DataPreparationPipeline
from src.relationships.discovery import RelationshipDiscovery


class UnifiedReadinessWorkflow:
    """
    Orchestrates the complete multi-dataset data-readiness workflow.

    Responsibilities:
    1. Process each dataset independently.
    2. Preserve the existing per-dataset preparation pipeline.
    3. Discover relationships across prepared datasets.
    4. Produce one unified machine-readable report.

    The workflow does not directly modify datasets.
    All transformations are delegated to DataPreparationPipeline.
    """

    def __init__(
        self,
        pipeline_factory: Callable[[], DataPreparationPipeline] | None = None,
        relationship_discovery: RelationshipDiscovery | None = None,
    ):
        self.pipeline_factory = (
            pipeline_factory
            or DataPreparationPipeline
        )

        self.relationship_discovery = (
            relationship_discovery
            or RelationshipDiscovery()
        )

    def run(
        self,
        dataset_paths: dict[str, str],
    ) -> dict[str, Any]:
        """
        Process multiple datasets and generate a unified readiness report.

        Parameters
        ----------
        dataset_paths:
            Mapping of dataset name to file path.

            Example:
                {
                    "customers": "data/customers.csv",
                    "orders": "data/orders.csv",
                }

        Returns
        -------
        dict
            Unified workflow result.
        """

        if not isinstance(dataset_paths, dict):
            raise TypeError(
                "dataset_paths must be a dictionary."
            )

        if not dataset_paths:
            raise ValueError(
                "dataset_paths cannot be empty."
            )

        dataset_reports = {}
        prepared_datasets = {}

        for dataset_name, file_path in dataset_paths.items():

            if not isinstance(dataset_name, str):
                raise TypeError(
                    "Dataset names must be strings."
                )

            if not dataset_name.strip():
                raise ValueError(
                    "Dataset names cannot be empty."
                )

            if not isinstance(file_path, str):
                raise TypeError(
                    f"File path for '{dataset_name}' "
                    "must be a string."
                )

            pipeline = self.pipeline_factory()

            result = pipeline.run(file_path)

            dataset_reports[dataset_name] = result

            prepared_df = result.get(
                "prepared_dataframe"
            )

            if not isinstance(
                prepared_df,
                pd.DataFrame,
            ):
                raise ValueError(
                    f"Pipeline for '{dataset_name}' "
                    "did not return a prepared dataframe."
                )

            prepared_datasets[dataset_name] = prepared_df

        relationship_report = (
            self.relationship_discovery.discover(
                prepared_datasets
            )
        )

        return {
            "status": "success",
            "dataset_count": len(dataset_reports),
            "datasets": dataset_reports,
            "prepared_datasets": prepared_datasets,
            "relationships": relationship_report,
            "summary": self._build_summary(
                dataset_reports,
                relationship_report,
            ),
        }

    @staticmethod
    def _build_summary(
        dataset_reports: dict[str, Any],
        relationship_report: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a compact summary of the unified workflow.
        """

        readiness_statuses = {}

        for dataset_name, report in dataset_reports.items():

            readiness = report.get(
                "readiness_report",
                {},
            )

            readiness_statuses[dataset_name] = (
                readiness.get(
                    "readiness_status",
                    "unknown",
                )
            )

        relationship_count = int(
            relationship_report.get(
                "relationship_count",
                0,
            )
        )

        return {
            "datasets_processed": len(
                dataset_reports
            ),
            "readiness_statuses": readiness_statuses,
            "relationships_discovered": relationship_count,
        }