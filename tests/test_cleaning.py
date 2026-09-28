import pandas as pd

from src.processing import DataCleaner


def test_rename_columns():

    df = pd.DataFrame(
        {
            "Product Name": ["Laptop", "Mouse"],
            "Sales": [1000, 500],
        }
    )

    plan = {
        "operations": [
            {
                "type": "rename_columns",
                "mapping": {
                    "Product Name": "product_name",
                    "Sales": "sales",
                },
            }
        ]
    }

    cleaner = DataCleaner()

    result = cleaner.clean(df, plan)

    assert "product_name" in result.columns
    assert "sales" in result.columns


def test_drop_duplicates():

    df = pd.DataFrame(
        {
            "product": ["Laptop", "Laptop", "Mouse"],
            "sales": [1000, 1000, 500],
        }
    )

    plan = {
        "operations": [
            {
                "type": "drop_duplicates",
            }
        ]
    }

    cleaner = DataCleaner()

    result = cleaner.clean(df, plan)

    assert len(result) == 2


def test_fill_missing_with_median():

    df = pd.DataFrame(
        {
            "sales": [100, 200, None, 300],
        }
    )

    plan = {
        "operations": [
            {
                "type": "fill_missing",
                "column": "sales",
                "strategy": "median",
            }
        ]
    }

    cleaner = DataCleaner()

    result = cleaner.clean(df, plan)

    assert result["sales"].isna().sum() == 0


def test_convert_datetime():

    df = pd.DataFrame(
        {
            "order_date": [
                "2026-01-01",
                "2026-02-01",
            ]
        }
    )

    plan = {
        "operations": [
            {
                "type": "convert_datetime",
                "column": "order_date",
            }
        ]
    }

    cleaner = DataCleaner()

    result = cleaner.clean(df, plan)

    assert pd.api.types.is_datetime64_any_dtype(
        result["order_date"]
    )


def test_strip_strings():

    df = pd.DataFrame(
        {
            "product": [
                " Laptop ",
                " Mouse ",
            ]
        }
    )

    plan = {
        "operations": [
            {
                "type": "strip_strings",
                "columns": ["product"],
            }
        ]
    }

    cleaner = DataCleaner()

    result = cleaner.clean(df, plan)

    assert result["product"].tolist() == [
        "Laptop",
        "Mouse",
    ]