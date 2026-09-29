import json

import pandas as pd
import pytest

from src.agents.data_preparation_agent import (
    DataPreparationAgent,
)


def test_build_prompt():

    agent = object.__new__(
        DataPreparationAgent
    )

    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse"],
            "sales": [1000, 500],
        }
    )

    profile = {
        "rows": 2,
        "columns": 2,
        "column_details": {},
        "duplicate_rows": 0,
    }

    semantic_context = {
        "dataset_description": "Sales dataset",
        "columns": {},
        "relationships": [],
        "potential_derived_metrics": [],
        "data_quality_concerns": [],
    }

    prompt = agent._build_prompt(
        df,
        profile,
        semantic_context,
    )

    assert "data preparation agent" in prompt.lower()
    assert "product" in prompt
    assert "sales" in prompt
    assert "operations" in prompt


def test_parse_valid_plan():

    agent = object.__new__(
        DataPreparationAgent
    )

    response = json.dumps(
        {
            "reasoning_summary": (
                "Dataset requires no changes."
            ),
            "operations": [],
            "warnings": [],
        }
    )

    result = agent._parse_response(
        response
    )

    assert isinstance(result, dict)
    assert "operations" in result
    assert result["operations"] == []


def test_parse_plan_with_operation():

    agent = object.__new__(
        DataPreparationAgent
    )

    response = json.dumps(
        {
            "reasoning_summary": (
                "Convert order date."
            ),
            "operations": [
                {
                    "type": "convert_datetime",
                    "column": "order_date",
                }
            ],
            "warnings": [],
        }
    )

    result = agent._parse_response(
        response
    )

    assert len(result["operations"]) == 1

    assert (
        result["operations"][0]["type"]
        == "convert_datetime"
    )


def test_parse_invalid_json():

    agent = object.__new__(
        DataPreparationAgent
    )

    with pytest.raises(ValueError):

        agent._parse_response(
            "This is not JSON"
        )


def test_missing_operations():

    agent = object.__new__(
        DataPreparationAgent
    )

    response = json.dumps(
        {
            "reasoning_summary": "Something",
            "warnings": [],
        }
    )

    with pytest.raises(ValueError):

        agent._parse_response(
            response
        )


def test_invalid_dataframe():

    agent = object.__new__(
        DataPreparationAgent
    )

    with pytest.raises(TypeError):

        agent.create_plan(
            "not dataframe",
            {},
            {},
        )


def test_empty_profile():

    agent = object.__new__(
        DataPreparationAgent
    )

    df = pd.DataFrame(
        {
            "sales": [100, 200]
        }
    )

    with pytest.raises(ValueError):

        agent.create_plan(
            df,
            {},
            {
                "dataset_description": "Sales"
            },
        )


def test_empty_semantic_context():

    agent = object.__new__(
        DataPreparationAgent
    )

    df = pd.DataFrame(
        {
            "sales": [100, 200]
        }
    )

    profile = {
        "rows": 2,
        "columns": 1,
    }

    with pytest.raises(ValueError):

        agent.create_plan(
            df,
            profile,
            {},
        )

def test_auto_action_high_confidence_is_executable():
    agent = DataPreparationAgent.__new__(
        DataPreparationAgent
    )

    proposed_plan = {
        "status": "success",
        "operations": [
            {
                "operation": "drop_duplicates",
                "column": None,
                "strategy": None,
                "value": None,
                "reason": "Duplicate rows detected.",
                "confidence": 0.95,
                "classification": "auto_action",
                "source_issue_type": "duplicate_rows",
                "source_column": None,
            }
        ],
    }

    semantic_quality = {
        "interpretations": [
            {
                "issue_type": "duplicate_rows",
                "column": None,
                "classification": "auto_action",
                "reason": "Duplicate records.",
                "recommended_action": "Remove duplicates.",
                "confidence": 0.95,
            }
        ]
    }

    result = agent._apply_safety_filter(
        proposed_plan=proposed_plan,
        semantic_quality=semantic_quality,
        dataframe_columns=["customer_id", "name"],
    )

    assert len(result["operations"]) == 1
    assert result["safety_summary"]["executable"] == 1
    assert result["safety_summary"]["review_required"] == 0

def test_review_classification_is_not_executable():
    agent = DataPreparationAgent.__new__(
        DataPreparationAgent
    )

    proposed_plan = {
        "status": "success",
        "operations": [
            {
                "operation": "fill_missing",
                "column": "age",
                "strategy": "median",
                "value": None,
                "reason": "Age contains missing values.",
                "confidence": 0.95,
                "classification": "review",
                "source_issue_type": "missing_values",
                "source_column": "age",
            }
        ],
    }

    semantic_quality = {
        "interpretations": [
            {
                "issue_type": "missing_values",
                "column": "age",
                "classification": "review",
                "reason": "Review required.",
                "recommended_action": "Review.",
                "confidence": 0.95,
            }
        ]
    }

    result = agent._apply_safety_filter(
        proposed_plan=proposed_plan,
        semantic_quality=semantic_quality,
        dataframe_columns=["age"],
    )

    assert len(result["operations"]) == 0
    assert len(result["review_required"]) == 1
    assert result["safety_summary"]["review_required"] == 1

def test_low_confidence_auto_action_requires_review():
    agent = DataPreparationAgent.__new__(
        DataPreparationAgent
    )

    proposed_plan = {
        "status": "success",
        "operations": [
            {
                "operation": "drop_duplicates",
                "column": None,
                "strategy": None,
                "value": None,
                "reason": "Possible duplicates.",
                "confidence": 0.65,
                "classification": "auto_action",
                "source_issue_type": "duplicate_rows",
                "source_column": None,
            }
        ],
    }

    semantic_quality = {
        "interpretations": [
            {
                "issue_type": "duplicate_rows",
                "column": None,
                "classification": "auto_action",
                "reason": "Possible duplicate records.",
                "recommended_action": "Review duplicates.",
                "confidence": 0.65,
            }
        ]
    }

    result = agent._apply_safety_filter(
        proposed_plan=proposed_plan,
        semantic_quality=semantic_quality,
        dataframe_columns=["customer_id"],
    )

    assert len(result["operations"]) == 0
    assert len(result["review_required"]) == 1
    assert result["safety_summary"]["executable"] == 0

def test_ignore_classification_is_skipped():
    agent = DataPreparationAgent.__new__(
        DataPreparationAgent
    )

    proposed_plan = {
        "status": "success",
        "operations": [
            {
                "operation": "fill_missing",
                "column": "age",
                "strategy": "median",
                "value": None,
                "reason": "Missing value detected.",
                "confidence": 0.99,
                "classification": "ignore",
                "source_issue_type": "missing_values",
                "source_column": "age",
            }
        ],
    }

    semantic_quality = {
        "interpretations": [
            {
                "issue_type": "missing_values",
                "column": "age",
                "classification": "ignore",
                "reason": "Missing value is legitimate.",
                "recommended_action": "Do nothing.",
                "confidence": 0.99,
            }
        ]
    }

    result = agent._apply_safety_filter(
        proposed_plan=proposed_plan,
        semantic_quality=semantic_quality,
        dataframe_columns=["age"],
    )

    assert len(result["operations"]) == 0
    assert len(result["ignored_operations"]) == 1
    assert result["safety_summary"]["ignored"] == 1

def test_nonexistent_column_requires_review():
    agent = DataPreparationAgent.__new__(
        DataPreparationAgent
    )

    proposed_plan = {
        "status": "success",
        "operations": [
            {
                "operation": "fill_missing",
                "column": "salary",
                "strategy": "median",
                "value": None,
                "reason": "Missing salary.",
                "confidence": 0.99,
                "classification": "auto_action",
                "source_issue_type": "missing_values",
                "source_column": "salary",
            }
        ],
    }

    semantic_quality = {
        "interpretations": []
    }

    result = agent._apply_safety_filter(
        proposed_plan=proposed_plan,
        semantic_quality=semantic_quality,
        dataframe_columns=["age", "name"],
    )

    assert len(result["operations"]) == 0
    assert len(result["review_required"]) == 1
