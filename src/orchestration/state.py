from typing import Any, TypedDict


class ReadinessGraphState(TypedDict, total=False):
    dataset_paths: dict[str, str]

    # Checkpoint-safe dataset reports.
    # Raw pandas DataFrames must never be stored here.
    dataset_reports: dict[str, Any]

    # Prepared DataFrames are stored as serialized JSON strings.
    prepared_datasets: dict[str, str]

    quality_decisions: dict[str, Any]

    human_review: dict[str, Any]
    human_approval: str

    stage_results: dict[str, Any]

    relationship_report: dict[str, Any]

    final_report: dict[str, Any]

    governance_route: str

    post_processing_complete: bool

    error: str