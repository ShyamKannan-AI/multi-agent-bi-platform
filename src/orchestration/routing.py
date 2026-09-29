from typing import Any


VALID_ROUTES = {
    "auto_action",
    "review",
    "ignore",
}


def route_quality_decision(
    decision: dict[str, Any],
) -> str:
    """
    Validate and return a supported governance route.

    Supported routes:
    - auto_action
    - review
    - ignore
    """

    if not isinstance(decision, dict):
        raise ValueError(
            "Quality decision must be a dictionary."
        )

    action = decision.get("action")

    if action not in VALID_ROUTES:
        raise ValueError(
            f"Invalid quality action: {action}"
        )

    return action


def determine_governance_route(
    dataset_reports: dict[str, Any],
) -> str:
    """
    Determine the overall governance route from
    preparation results across all datasets.

    Priority:
        review > auto_action > ignore
    """

    if not isinstance(dataset_reports, dict):
        raise ValueError(
            "dataset_reports must be a dictionary."
        )

    has_review = False
    has_auto_action = False

    for report in dataset_reports.values():

        if not isinstance(report, dict):
            continue

        preparation_plan = report.get(
            "preparation_plan",
            {},
        )

        if not isinstance(preparation_plan, dict):
            continue

        review_required = preparation_plan.get(
            "review_required",
            [],
        )

        operations = preparation_plan.get(
            "operations",
            [],
        )

        if review_required:
            has_review = True

        if operations:
            has_auto_action = True

    if has_review:
        return "review"

    if has_auto_action:
        return "auto_action"

    return "ignore"