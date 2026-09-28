import pandas as pd

from src.processing import PreparationAuditor


def test_audit_detects_changes():

    original_df = pd.DataFrame(
        {
            "product": [
                " Laptop ",
                " Laptop ",
                " Mouse ",
            ],
            "sales": [
                1000,
                1000,
                None,
            ],
        }
    )

    prepared_df = pd.DataFrame(
        {
            "product": [
                "Laptop",
                "Mouse",
            ],
            "sales": [
                1000,
                500,
            ],
        }
    )

    plan = {
        "operations": [
            {
                "type": "strip_strings",
                "columns": ["product"],
            },
            {
                "type": "drop_duplicates",
            },
            {
                "type": "fill_missing",
                "column": "sales",
                "strategy": "value",
                "value": 500,
            },
        ]
    }

    auditor = PreparationAuditor()

    result = auditor.audit(
        original_df,
        prepared_df,
        plan,
    )

    assert result["original"]["rows"] == 3
    assert result["prepared"]["rows"] == 2

    assert result["original"]["missing_values"] == 1
    assert result["prepared"]["missing_values"] == 0

    assert result["changes"]["rows_removed"] == 1
    assert result["changes"]["missing_values_resolved"] == 1

    assert (
        result["changes"]["duplicates_resolved"]
        == 1
    )


def test_operation_summary():

    auditor = PreparationAuditor()

    operations = [
        {
            "type": "strip_strings",
        },
        {
            "type": "strip_strings",
        },
        {
            "type": "drop_duplicates",
        },
        {
            "type": "fill_missing",
        },
    ]

    result = auditor._summarize_operations(
        operations
    )

    assert result["strip_strings"] == 2
    assert result["drop_duplicates"] == 1
    assert result["fill_missing"] == 1