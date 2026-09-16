import pandas as pd
import pytest

from src.profiling import profile_dataset


def test_profile_dataset():

    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse", "Keyboard", None],
            "sales": [1000, 500, 750, 900],
        }
    )

    profile = profile_dataset(df)

    assert profile["rows"] == 4
    assert profile["columns"] == 2

    assert profile["column_details"]["product"]["missing_count"] == 1
    assert (
        profile["column_details"]["product"]["missing_percentage"]
        == 25.0
    )

    assert profile["column_details"]["sales"]["missing_count"] == 0

    assert "statistics" in profile["column_details"]["sales"]


def test_duplicate_detection():

    df = pd.DataFrame(
        {
            "name": ["A", "B", "A"],
            "value": [10, 20, 10],
        }
    )

    profile = profile_dataset(df)

    assert profile["duplicate_rows"] == 1


def test_empty_dataframe():

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        profile_dataset(df)


def test_invalid_input():

    with pytest.raises(TypeError):
        profile_dataset("not a dataframe")