from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.loaders import load_dataset
from src.ingestion.validator import validate_file


def test_csv_loading(tmp_path: Path):
    file_path = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "product": ["A", "B", "C"],
            "sales": [100, 200, 300],
        }
    ).to_csv(file_path, index=False)

    validate_file(file_path)

    df = load_dataset(file_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["product", "sales"]


def test_unsupported_file():
    with pytest.raises(ValueError):
        validate_file("sample.txt")


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_file("does_not_exist.csv")