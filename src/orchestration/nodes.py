from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from langgraph.types import interrupt

from src.orchestration.routing import (
    determine_governance_route,
)
from src.processing.pipeline import (
    DataPreparationPipeline,
)
from src.relationships.discovery import (
    RelationshipDiscovery,
)


def _serialize_dataframe(
    df: pd.DataFrame,
) -> str:
    return df.to_json(
        orient="split",
        date_format="iso",
    )


def _deserialize_dataframe(
    serialized: str,
) -> pd.DataFrame:
    return pd.read_json(
        StringIO(serialized),
        orient="split",
    )


def _make_checkpoint_safe(
    value: Any,
) -> Any:

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return _serialize_dataframe(value)

    if isinstance(
        value,
        pd.Series,
    ):
        return value.to_list()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _make_checkpoint_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _make_checkpoint_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        set,
    ):
        return [
            _make_checkpoint_safe(item)
            for item in value
        ]

    return value


def prepare_datasets_node(
    state: dict[str, Any],
    pipeline_factory=None,
) -> dict[str, Any]:

    dataset_paths = state.get(
        "dataset_paths",
        {},
    )

    if not isinstance(
        dataset_paths,
        dict,
    ):
        raise ValueError(
            "dataset_paths must be a dictionary."
        )

    if not dataset_paths:
        raise ValueError(
            "Dataset paths cannot be empty."
        )

    # IMPORTANT:
    # If no factory is explicitly supplied, use the
    # module-level DataPreparationPipeline.
    #
    # This also preserves compatibility with tests that
    # monkeypatch nodes.DataPreparationPipeline.
    if pipeline_factory is None:
        pipeline_factory = DataPreparationPipeline

    dataset_reports = {}
    prepared_datasets = {}

    for (
        dataset_name,
        dataset_path,
    ) in dataset_paths.items():

        pipeline = pipeline_factory()

        result = pipeline.run(
            dataset_path
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                f"Pipeline result for '{dataset_name}' "
                "must be a dictionary."
            )

        prepared_df = result.get(
            "prepared_dataframe"
        )

        if not isinstance(
            prepared_df,
            pd.DataFrame,
        ):
            raise ValueError(
                f"Pipeline result for '{dataset_name}' "
                "does not contain a valid prepared_dataframe."
            )

        prepared_datasets[
            dataset_name
        ] = _serialize_dataframe(
            prepared_df
        )

        report = dict(result)

        report.pop(
            "prepared_dataframe",
            None,
        )

        report.pop(
            "original_dataframe",
            None,
        )

        dataset_reports[
            dataset_name
        ] = _make_checkpoint_safe(
            report
        )

    return {
        "dataset_reports": dataset_reports,
        "prepared_datasets": prepared_datasets,
    }


def governance_decision_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    dataset_reports = state.get(
        "dataset_reports",
        {},
    )

    route = determine_governance_route(
        dataset_reports
    )

    quality_decisions = {}

    for (
        dataset_name,
        report,
    ) in dataset_reports.items():

        preparation_plan = report.get(
            "preparation_plan",
            {},
        )

        if not isinstance(
            preparation_plan,
            dict,
        ):
            preparation_plan = {}

        operations = preparation_plan.get(
            "operations",
            [],
        )

        review_required = preparation_plan.get(
            "review_required",
            [],
        )

        ignored_operations = preparation_plan.get(
            "ignored_operations",
            [],
        )

        quality_decisions[
            dataset_name
        ] = {
            "dataset": dataset_name,
            "action": (
                "review"
                if review_required
                else "auto_action"
                if operations
                else "ignore"
            ),
            "operations": operations,
            "review_required": review_required,
            "ignored_operations": ignored_operations,
        }

    return {
        "governance_route": route,
        "quality_decisions": _make_checkpoint_safe(
            quality_decisions
        ),
    }


def human_review_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    dataset_reports = state.get(
        "dataset_reports",
        {},
    )

    affected_datasets = []
    review_items = []

    for (
        dataset_name,
        report,
    ) in dataset_reports.items():

        preparation_plan = report.get(
            "preparation_plan",
            {},
        )

        if not isinstance(
            preparation_plan,
            dict,
        ):
            continue

        review_required = preparation_plan.get(
            "review_required",
            [],
        )

        if review_required:

            affected_datasets.append(
                dataset_name
            )

            review_items.append(
                {
                    "dataset": dataset_name,
                    "operations": _make_checkpoint_safe(
                        review_required
                    ),
                }
            )

    review_request = {
        "type": "human_review",
        "message": (
            "Human approval is required before "
            "the proposed data-preparation operations "
            "can continue."
        ),
        "datasets": affected_datasets,
        "review_items": review_items,
        "options": [
            "approve",
            "reject",
        ],
    }

    decision = interrupt(
        review_request
    )

    if isinstance(
        decision,
        dict,
    ):
        decision = decision.get(
            "decision"
        )

    if decision not in {
        "approve",
        "reject",
    }:
        raise ValueError(
            "Human decision must be "
            "'approve' or 'reject'."
        )

    return {
        "human_review": {
            "status": "completed",
            "datasets": affected_datasets,
            "review_items": review_items,
            "decision": decision,
        },
        "human_approval": decision,
    }


def rejected_review_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    human_review = {
        **state.get(
            "human_review",
            {},
        ),
        "status": "human_rejected",
        "decision": "reject",
    }

    return {
        "human_review": human_review,

        "final_report": {
            "status": "human_rejected",
            "governance_route": state.get(
                "governance_route",
                "review",
            ),
            "datasets": state.get(
                "dataset_reports",
                {},
            ),
            "human_review": human_review,
            "summary": {
                "human_decision": "reject",
            },
        },

        "post_processing_complete": False,
    }


def publish_stage_results_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    dataset_reports = state.get(
        "dataset_reports",
        {},
    )

    stage_results = {}

    for (
        dataset_name,
        report,
    ) in dataset_reports.items():

        stage_results[
            dataset_name
        ] = {
            "ingestion": {
                "status": "completed",
                "result": report.get(
                    "profile",
                    {},
                ),
            },
            "profiling": {
                "status": "completed",
                "result": report.get(
                    "profile",
                    {},
                ),
            },
            "semantic_understanding": {
                "status": "completed",
                "result": report.get(
                    "semantic_context",
                    {},
                ),
            },
            "quality_analysis": {
                "status": "completed",
                "result": report.get(
                    "quality_report",
                    {},
                ),
            },
            "semantic_quality": {
                "status": "completed",
                "result": report.get(
                    "semantic_quality",
                    {},
                ),
            },
            "preparation_plan": {
                "status": "completed",
                "result": report.get(
                    "preparation_plan",
                    {},
                ),
            },
            "validation": {
                "status": "completed",
                "result": report.get(
                    "validation",
                    {},
                ),
            },
            "audit": {
                "status": "completed",
                "result": report.get(
                    "audit",
                    {},
                ),
            },
            "readiness": {
                "status": "completed",
                "result": report.get(
                    "readiness_report",
                    {},
                ),
            },
        }

    return {
        "stage_results": _make_checkpoint_safe(
            stage_results
        )
    }


def relationship_discovery_node(
    state: dict[str, Any],
    relationship_discovery=None,
) -> dict[str, Any]:

    if relationship_discovery is None:
        relationship_discovery = (
            RelationshipDiscovery()
        )

    serialized_datasets = state.get(
        "prepared_datasets",
        {},
    )

    datasets = {
        dataset_name: _deserialize_dataframe(
            serialized_df
        )
        for (
            dataset_name,
            serialized_df,
        ) in serialized_datasets.items()
    }

    # HITL and single-dataset workflows are valid.
    # The RelationshipDiscovery engine itself remains strict
    # and requires two datasets.
    if len(datasets) < 2:

        return {
            "relationship_report": {
                "status": "success",
                "dataset_count": len(
                    datasets
                ),
                "relationship_count": 0,
                "relationships": [],
                "summary": {
                    "relationship_count": 0,
                    "message": (
                        "Relationship discovery skipped "
                        "because fewer than two datasets "
                        "were provided."
                    ),
                },
            }
        }

    report = relationship_discovery.discover(
        datasets
    )

    return {
        "relationship_report": _make_checkpoint_safe(
            report
        )
    }


def join_post_processing_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    return {
        "post_processing_complete": True
    }


def final_report_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    dataset_reports = state.get(
        "dataset_reports",
        {},
    )

    readiness_summaries = {}

    for (
        dataset_name,
        report,
    ) in dataset_reports.items():

        readiness_summaries[
            dataset_name
        ] = report.get(
            "readiness_report",
            {},
        )

    relationship_report = state.get(
        "relationship_report",
        {},
    )

    final_report = {
        "status": "completed",
        "governance_route": state.get(
            "governance_route",
            "ignore",
        ),
        "datasets": dataset_reports,
        "stage_results": state.get(
            "stage_results",
            {},
        ),
        "relationship_report": relationship_report,
        "human_review": state.get(
            "human_review",
            {},
        ),
        "readiness_summary": readiness_summaries,
        "summary": {
            "dataset_count": len(
                dataset_reports
            ),
            "relationship_count": relationship_report.get(
                "relationship_count",
                len(
                    relationship_report.get(
                        "relationships",
                        [],
                    )
                ),
            ),
        },
    }

    return {
        "final_report": _make_checkpoint_safe(
            final_report
        )
    }