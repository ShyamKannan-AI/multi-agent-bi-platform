import json

import pandas as pd
import pytest

from src.semantics.semantic_analyzer import SemanticAnalyzer


def test_build_context():

    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse"],
            "sales": [1000, 500],
        }
    )

    analyzer = object.__new__(SemanticAnalyzer)

    profile = {
        "rows": 2,
        "columns": 2,
        "column_details": {},
        "duplicate_rows": 0,
    }

    context = analyzer._build_context(
        df,
        profile,
    )

    assert context["dataset_shape"]["rows"] == 2
    assert context["dataset_shape"]["columns"] == 2

    assert len(context["sample_rows"]) == 2


def test_build_prompt():

    analyzer = object.__new__(SemanticAnalyzer)

    context = {
        "dataset_shape": {
            "rows": 2,
            "columns": 2,
        },
        "profile": {},
        "sample_rows": [
            {
                "product": "Laptop",
                "sales": 1000,
            }
        ],
    }

    prompt = analyzer._build_prompt(context)

    assert "semantic" in prompt.lower()
    assert "product" in prompt
    assert "sales" in prompt


def test_parse_valid_response():

    analyzer = object.__new__(SemanticAnalyzer)

    response = json.dumps(
        {
            "dataset_description": "Sales dataset",
            "columns": {},
            "relationships": [],
            "potential_derived_metrics": [],
            "data_quality_concerns": [],
        }
    )

    result = analyzer._parse_response(response)

    assert isinstance(result, dict)
    assert result["dataset_description"] == "Sales dataset"


def test_parse_invalid_response():

    analyzer = object.__new__(SemanticAnalyzer)

    with pytest.raises(ValueError):

        analyzer._parse_response(
            "This is not JSON"
        )