import pytest

from src.orchestration.routing import (
    determine_governance_route,
    route_quality_decision,
)


def test_auto_action_route():
    decision = {
        "action": "auto_action",
        "confidence": 0.95,
    }

    assert route_quality_decision(
        decision
    ) == "auto_action"


def test_review_route():
    decision = {
        "action": "review",
        "confidence": 0.65,
    }

    assert route_quality_decision(
        decision
    ) == "review"


def test_ignore_route():
    decision = {
        "action": "ignore",
        "confidence": 0.90,
    }

    assert route_quality_decision(
        decision
    ) == "ignore"


def test_invalid_route():
    decision = {
        "action": "delete_everything",
    }

    with pytest.raises(ValueError):
        route_quality_decision(
            decision
        )


def test_invalid_decision_type():
    with pytest.raises(ValueError):
        route_quality_decision(
            "auto_action"
        )


def test_governance_route_review_has_priority():
    reports = {
        "customers": {
            "preparation_plan": {
                "operations": [
                    {
                        "operation": "fill_missing"
                    }
                ],
                "review_required": [],
            }
        },
        "transactions": {
            "preparation_plan": {
                "operations": [],
                "review_required": [
                    {
                        "operation": "convert_datetime"
                    }
                ],
            }
        },
    }

    assert determine_governance_route(
        reports
    ) == "review"


def test_governance_route_auto_action():
    reports = {
        "customers": {
            "preparation_plan": {
                "operations": [
                    {
                        "operation": "fill_missing"
                    }
                ],
                "review_required": [],
            }
        }
    }

    assert determine_governance_route(
        reports
    ) == "auto_action"


def test_governance_route_ignore():
    reports = {
        "customers": {
            "preparation_plan": {
                "operations": [],
                "review_required": [],
            }
        }
    }

    assert determine_governance_route(
        reports
    ) == "ignore"