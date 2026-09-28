import pandas as pd
import pytest

from src.quality import DataQualityEngine


def test_quality_engine_returns_success_report():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "revenue": [100, 200, 300],
        }
    )

    engine = DataQualityEngine()

    result = engine.analyze(df)

    assert result["status"] == "success"
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["issues"] == []

    assert result["summary"]["total_issues"] == 0
    assert result["summary"]["critical"] == 0
    assert result["summary"]["warning"] == 0
    assert result["summary"]["info"] == 0


def test_quality_engine_rejects_non_dataframe():
    engine = DataQualityEngine()

    with pytest.raises(TypeError):
        engine.analyze(
            {
                "customer_id": [1, 2, 3],
                "revenue": [100, 200, 300],
            }
        )


def test_quality_engine_rejects_empty_dataframe():
    engine = DataQualityEngine()

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        engine.analyze(df)


def test_quality_engine_does_not_modify_input():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "revenue": [100, 200, 300],
        }
    )

    original_df = df.copy(deep=True)

    engine = DataQualityEngine()
    engine.analyze(df)

    pd.testing.assert_frame_equal(df, original_df)