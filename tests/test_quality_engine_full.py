import pandas as pd

from src.quality import DataQualityEngine


def test_quality_engine_detects_multiple_quality_issues():
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 3, 4, 5],
            "age": [25, None, 30, 31, 200],
            "status": [
                "Active",
                "active ",
                "Inactive",
                "ACTIVE",
                "Inactive",
            ],
            "transaction_date": [
                "2025-01-01",
                "2025-01-02",
                "invalid-date",
                "2025-01-04",
                "2025-01-05",
            ],
            "revenue": [
                100,
                200,
                300,
                400,
                100000,
            ],
        }
    )

    engine = DataQualityEngine()

    report = engine.analyze(df)

    issue_types = {
        issue["type"]
        for issue in report["issues"]
    }

    assert "missing_values" in issue_types
    assert "uniqueness_violation" in issue_types
    assert "categorical_inconsistency" in issue_types
    assert "invalid_dates" in issue_types
    assert "outliers" in issue_types

    assert report["status"] == "success"
    assert report["summary"]["total_issues"] > 0


def test_quality_engine_does_not_modify_data():
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "revenue": [100, 200, 300],
        }
    )

    original = df.copy(deep=True)

    engine = DataQualityEngine()
    engine.analyze(df)

    pd.testing.assert_frame_equal(df, original)