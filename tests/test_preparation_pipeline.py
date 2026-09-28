import pandas as pd
import pytest

from src.processing.audit import PreparationAuditor
from src.processing.cleaning import DataCleaner
from src.processing.pipeline import DataPreparationPipeline


# ============================================================
# Mock semantic analyzer
# ============================================================

class MockSemanticAnalyzer:
    def analyze(self, df, profile):
        return {
            "status": "success",
            "dataset_description": "Test dataset",
            "columns": {
                column: {
                    "semantic_role": "unknown"
                }
                for column in df.columns
            },
        }


# ============================================================
# Mock preparation agent
# ============================================================

class MockPreparationAgent:
    def create_plan(
        self,
        df,
        profile,
        semantic_context,
    ):
        return {
            "status": "success",
            "operations": [],
        }


# ============================================================
# Mock quality engine
# ============================================================

class MockQualityEngine:
    def analyze(self, df):
        return {
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "issues": [
                {
                    "type": "missing_values",
                    "column": "age",
                    "severity": "warning",
                    "message": (
                        "Column 'age' contains "
                        "1 missing value."
                    ),
                    "evidence": {
                        "missing_count": 1,
                        "missing_percentage": 33.33,
                        "total_rows": 3,
                    },
                }
            ],
            "summary": {
                "total_issues": 1,
                "critical": 0,
                "warning": 1,
                "info": 0,
            },
        }


# ============================================================
# Mock quality agent
# ============================================================

class MockQualityAgent:
    def analyze(
        self,
        quality_report,
        semantic_context=None,
    ):
        return {
            "status": "success",
            "interpretations": [
                {
                    "issue_type": "missing_values",
                    "column": "age",
                    "classification": "review",
                    "reason": (
                        "Missing age value requires "
                        "semantic review."
                    ),
                    "recommended_action": (
                        "Review before filling."
                    ),
                    "confidence": 0.90,
                }
            ],
            "summary": {
                "total": 1,
                "auto_action": 0,
                "review": 1,
                "ignore": 0,
            },
        }


# ============================================================
# Basic pipeline test
# ============================================================

def test_pipeline_runs_successfully(tmp_path):
    file_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }
    ).to_csv(file_path, index=False)

    pipeline = DataPreparationPipeline(
        semantic_analyzer=MockSemanticAnalyzer(),
        preparation_agent=MockPreparationAgent(),
        cleaner=DataCleaner(),
        auditor=PreparationAuditor(),
        quality_engine=MockQualityEngine(),
        quality_agent=MockQualityAgent(),
    )

    result = pipeline.run(str(file_path))

    assert result["validation"]["status"] == "success"

    assert "original_dataframe" in result
    assert "prepared_dataframe" in result
    assert "profile" in result
    assert "semantic_context" in result
    assert "preparation_plan" in result
    assert "validation" in result
    assert "audit" in result


# ============================================================
# Original dataframe preserved
# ============================================================

def test_pipeline_preserves_original_dataframe(tmp_path):
    file_path = tmp_path / "customers.csv"

    original = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }
    )

    original.to_csv(file_path, index=False)

    pipeline = DataPreparationPipeline(
        semantic_analyzer=MockSemanticAnalyzer(),
        preparation_agent=MockPreparationAgent(),
        cleaner=DataCleaner(),
        auditor=PreparationAuditor(),
        quality_engine=MockQualityEngine(),
        quality_agent=MockQualityAgent(),
    )

    result = pipeline.run(str(file_path))

    pd.testing.assert_frame_equal(
        result["original_dataframe"],
        original,
    )


# ============================================================
# Prepared dataframe exists
# ============================================================

def test_pipeline_returns_prepared_dataframe(tmp_path):
    file_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }
    ).to_csv(file_path, index=False)

    pipeline = DataPreparationPipeline(
        semantic_analyzer=MockSemanticAnalyzer(),
        preparation_agent=MockPreparationAgent(),
        cleaner=DataCleaner(),
        auditor=PreparationAuditor(),
        quality_engine=MockQualityEngine(),
        quality_agent=MockQualityAgent(),
    )

    result = pipeline.run(str(file_path))

    assert isinstance(
        result["prepared_dataframe"],
        pd.DataFrame,
    )

    assert len(
        result["prepared_dataframe"]
    ) == 3


# ============================================================
# Validation result
# ============================================================

def test_pipeline_validation_contains_expected_fields(
    tmp_path,
):
    file_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }
    ).to_csv(file_path, index=False)

    pipeline = DataPreparationPipeline(
        semantic_analyzer=MockSemanticAnalyzer(),
        preparation_agent=MockPreparationAgent(),
        cleaner=DataCleaner(),
        auditor=PreparationAuditor(),
        quality_engine=MockQualityEngine(),
        quality_agent=MockQualityAgent(),
    )

    result = pipeline.run(str(file_path))

    validation = result["validation"]

    assert validation["status"] == "success"
    assert validation["original_rows"] == 3
    assert validation["prepared_rows"] == 3
    assert validation["original_columns"] == 2
    assert validation["prepared_columns"] == 2
    assert validation["operations_applied"] == 0
    assert validation["rows_removed"] == 0


# ============================================================
# Quality analysis integration
# ============================================================

def test_pipeline_includes_quality_analysis(tmp_path):
    file_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": [20, None, 30],
        }
    ).to_csv(file_path, index=False)

    pipeline = DataPreparationPipeline(
        semantic_analyzer=MockSemanticAnalyzer(),
        preparation_agent=MockPreparationAgent(),
        cleaner=DataCleaner(),
        auditor=PreparationAuditor(),
        quality_engine=MockQualityEngine(),
        quality_agent=MockQualityAgent(),
    )

    result = pipeline.run(str(file_path))

    # Quality report should be returned
    assert "quality_report" in result

    # LLM-based semantic quality interpretation
    # should also be returned
    assert "semantic_quality" in result

    # Deterministic quality engine detected one issue
    assert (
        result["quality_report"]["summary"]["total_issues"]
        == 1
    )

    # Semantic quality agent classified it for review
    assert (
        result["semantic_quality"]["summary"]["review"]
        == 1
    )


# ============================================================
# Quality analysis uses original dataframe
# ============================================================

def test_quality_analysis_runs_before_preparation(
    tmp_path,
):
    file_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": [20, None, 30],
        }
    ).to_csv(file_path, index=False)

    class TrackingQualityEngine:
        def __init__(self):
            self.received_dataframe = None

        def analyze(self, df):
            self.received_dataframe = df.copy()

            return {
                "status": "success",
                "rows": len(df),
                "columns": len(df.columns),
                "issues": [],
                "summary": {
                    "total_issues": 0,
                    "critical": 0,
                    "warning": 0,
                    "info": 0,
                },
            }

    quality_engine = TrackingQualityEngine()

    pipeline = DataPreparationPipeline(
        semantic_analyzer=MockSemanticAnalyzer(),
        preparation_agent=MockPreparationAgent(),
        cleaner=DataCleaner(),
        auditor=PreparationAuditor(),
        quality_engine=quality_engine,
        quality_agent=MockQualityAgent(),
    )

    pipeline.run(str(file_path))

    assert quality_engine.received_dataframe is not None

    assert (
        quality_engine.received_dataframe["age"]
        .isna()
        .sum()
        == 1
    )


# ============================================================
# Pipeline rejects invalid input path
# ============================================================

def test_pipeline_rejects_invalid_file():
    pipeline = DataPreparationPipeline(
        semantic_analyzer=MockSemanticAnalyzer(),
        preparation_agent=MockPreparationAgent(),
        cleaner=DataCleaner(),
        auditor=PreparationAuditor(),
        quality_engine=MockQualityEngine(),
        quality_agent=MockQualityAgent(),
    )

    with pytest.raises(Exception):
        pipeline.run("does_not_exist.csv")