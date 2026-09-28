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