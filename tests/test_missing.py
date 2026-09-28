import pandas as pd
import pytest

from src.quality import MissingValueDetector


def test_detects_missing_values():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "age": [25, None, 31, None],
            "revenue": [100, 200, 300, 400],
        }
    )

    detector = MissingValueDetector()

    issues = detector.analyze(df)

    assert len(issues) == 1

    issue = issues[0]

    assert issue["type"] == "missing_values"
    assert issue["column"] == "age"
    assert issue["severity"] == "warning"

    assert issue["evidence"]["missing_count"] == 2
    assert issue["evidence"]["missing_percentage"] == 50.0
    assert issue["evidence"]["total_rows"] == 4


def test_ignores_columns_without_missing_values():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "revenue": [100, 200, 300],
        }
    )

    detector = MissingValueDetector()

    issues = detector.analyze(df)

    assert issues == []


def test_detects_multiple_columns_with_missing_values():
    df = pd.DataFrame(
        {
            "customer_id": [1, None, 3, None],
            "age": [25, None, 31, 40],
            "revenue": [100, 200, None, None],
        }
    )

    detector = MissingValueDetector()

    issues = detector.analyze(df)

    assert len(issues) == 3

    columns = {issue["column"] for issue in issues}

    assert columns == {
        "customer_id",
        "age",
        "revenue",
    }


def test_missing_detector_does_not_modify_dataframe():
    df = pd.DataFrame(
        {
            "customer_id": [1, None, 3],
            "revenue": [100, 200, 300],
        }
    )

    original_df = df.copy(deep=True)

    detector = MissingValueDetector()
    detector.analyze(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_missing_detector_rejects_non_dataframe():
    detector = MissingValueDetector()

    with pytest.raises(TypeError):
        detector.analyze(
            {
                "age": [20, None, 30],
            }
        )


def test_missing_detector_rejects_empty_dataframe():
    detector = MissingValueDetector()

    with pytest.raises(ValueError):
        detector.analyze(pd.DataFrame())