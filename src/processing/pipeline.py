from typing import Any

import pandas as pd

from src.agents import DataPreparationAgent, DataQualityAgent
from src.ingestion import load_dataset, validate_file
from src.processing.audit import PreparationAuditor
from src.processing.cleaning import DataCleaner
from src.profiling import profile_dataset
from src.quality import DataQualityEngine
from src.semantics import SemanticAnalyzer


class DataPreparationPipeline:
    """
    End-to-end dataset understanding and preparation pipeline.

    Pipeline stages:
        1. Validate input file
        2. Load dataset
        3. Profile dataset
        4. Understand dataset semantics
        5. Detect deterministic data-quality issues
        6. Interpret quality issues using LLM
        7. Generate preparation plan
        8. Execute deterministic preparation
        9. Validate preparation result
        10. Audit transformations

    The pipeline never allows the LLM to directly modify the dataframe.
    """

    def __init__(
        self,
        semantic_analyzer=None,
        preparation_agent=None,
        cleaner=None,
        auditor=None,
        quality_engine=None,
        quality_agent=None,
    ):
        self.semantic_analyzer = (
            semantic_analyzer or SemanticAnalyzer()
        )

        self.preparation_agent = (
            preparation_agent or DataPreparationAgent()
        )

        self.cleaner = cleaner or DataCleaner()

        self.auditor = auditor or PreparationAuditor()

        self.quality_engine = (
            quality_engine or DataQualityEngine()
        )

        self.quality_agent = (
            quality_agent or DataQualityAgent()
        )

    def run(self, file_path: str) -> dict[str, Any]:
        """
        Run the complete data understanding and preparation pipeline.

        Parameters
        ----------
        file_path:
            Path to CSV, Excel, or JSON dataset.

        Returns
        -------
        dict
            Complete pipeline result containing profiling,
            semantic understanding, quality analysis,
            preparation plan, prepared dataframe,
            validation, and audit information.
        """

        # ---------------------------------------------------------
        # 1. Validate input file
        # ---------------------------------------------------------
        validate_file(file_path)

        # ---------------------------------------------------------
        # 2. Load dataset
        # ---------------------------------------------------------
        original_df = load_dataset(file_path)

        # ---------------------------------------------------------
        # 3. Deterministic profiling
        # ---------------------------------------------------------
        profile = profile_dataset(original_df)

        # ---------------------------------------------------------
        # 4. Semantic understanding
        # ---------------------------------------------------------
        semantic_context = self.semantic_analyzer.analyze(
            original_df,
            profile,
        )

        # ---------------------------------------------------------
        # 5. Deterministic data-quality analysis
        # ---------------------------------------------------------
        quality_report = self.quality_engine.analyze(
            original_df
        )

        # ---------------------------------------------------------
        # 6. Semantic interpretation of quality findings
        # ---------------------------------------------------------
        semantic_quality = self.quality_agent.analyze(
            quality_report=quality_report,
            semantic_context=semantic_context,
        )

        # ---------------------------------------------------------
        # 7. Generate preparation plan
        # ---------------------------------------------------------
        preparation_plan = self.preparation_agent.create_plan(
           df=original_df,
           profile=profile,
           semantic_context=semantic_context,
           quality_report=quality_report,
           semantic_quality=semantic_quality,
        )

        # ---------------------------------------------------------
        # 8. Execute preparation deterministically
        # ---------------------------------------------------------
        prepared_df = self.cleaner.clean(
            original_df,
            preparation_plan,
        )

        # ---------------------------------------------------------
        # 9. Validate preparation result
        # ---------------------------------------------------------
        validation = self._validate_preparation(
            original_df,
            prepared_df,
            preparation_plan,
        )

        # ---------------------------------------------------------
        # 10. Audit preparation
        # ---------------------------------------------------------
        audit = self.auditor.audit(
            original_df=original_df,
            prepared_df=prepared_df,
            preparation_plan=preparation_plan,
        )

        # ---------------------------------------------------------
        # Final result
        # ---------------------------------------------------------
        return {
            "original_dataframe": original_df,
            "prepared_dataframe": prepared_df,
            "profile": profile,
            "semantic_context": semantic_context,
            "quality_report": quality_report,
            "semantic_quality": semantic_quality,
            "preparation_plan": preparation_plan,
            "validation": validation,
            "audit": audit,
        }

    @staticmethod
    def _validate_preparation(
        original_df: pd.DataFrame,
        prepared_df: pd.DataFrame,
        preparation_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate that the preparation stage produced
        a structurally valid dataframe.
        """

        if not isinstance(prepared_df, pd.DataFrame):
            raise TypeError(
                "Prepared dataset must be a Pandas DataFrame."
            )

        if prepared_df.empty:
            raise ValueError(
                "Preparation produced an empty dataset."
            )

        operations = preparation_plan.get(
            "operations",
            [],
        )

        return {
            "status": "success",
            "original_rows": len(original_df),
            "prepared_rows": len(prepared_df),
            "original_columns": len(original_df.columns),
            "prepared_columns": len(prepared_df.columns),
            "operations_applied": len(operations),
            "rows_removed": (
                len(original_df) - len(prepared_df)
            ),
        }