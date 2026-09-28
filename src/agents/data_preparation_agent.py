import json
from typing import Any

import pandas as pd

from src.llm import get_llm


class DataPreparationAgent:
    """
    Agent responsible for deciding how a dataset should be prepared.

    The agent analyzes deterministic profiling information and
    semantic dataset understanding, then produces a structured
    preparation plan.

    It does not directly modify the DataFrame.
    """

    def __init__(self):
        self.llm = get_llm()

    def create_plan(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a data-preparation plan using the LLM.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Expected a Pandas DataFrame."
            )

        if not profile:
            raise ValueError(
                "Dataset profile cannot be empty."
            )

        if not semantic_context:
            raise ValueError(
                "Semantic context cannot be empty."
            )

        prompt = self._build_prompt(
            df=df,
            profile=profile,
            semantic_context=semantic_context,
        )

        response = self.llm.invoke(prompt)

        return self._parse_response(response.content)

    def _build_prompt(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
    ) -> str:
        """
        Build the preparation-planning prompt.
        """

        context = {
            "dataset_shape": {
                "rows": len(df),
                "columns": len(df.columns),
            },
            "columns": df.columns.tolist(),
            "profile": profile,
            "semantic_context": semantic_context,
        }

        context_json = json.dumps(
            context,
            indent=2,
            default=str,
        )

        return f"""
You are a data preparation agent inside an autonomous
business intelligence system.

Your task is to determine how a structured dataset should
be prepared before business analysis.

You are given:

1. Deterministic dataset profiling information.
2. Semantic understanding of the dataset.

Your job is to create a preparation plan.

IMPORTANT RULES:

- Do not calculate business insights.
- Do not invent dataset information.
- Do not directly modify the dataset.
- Only recommend transformations that are justified by
  the available evidence.
- Avoid unnecessary transformations.
- Do not automatically fill missing identifier values.
- Do not remove rows unless there is a clear reason.
- Preserve useful information whenever possible.
- Do not treat outliers as errors automatically.
- Date columns should be converted to datetime when
  they are clearly identified as date/time fields.
- String values may be stripped when appropriate.
- Column names should be normalized when useful.
- Duplicate rows may be removed when they appear to be
  true duplicate records.

Available operations are:

1. rename_columns
2. drop_duplicates
3. fill_missing
4. convert_datetime
5. strip_strings

For fill_missing, allowed strategies are:

- mean
- median
- mode
- value
- ffill
- bfill

Return ONLY valid JSON.

Use this exact structure:

{{
    "reasoning_summary": "Short explanation of the preparation decisions",

    "operations": [
        {{
            "type": "operation_name",
            "column": "column_name if applicable",
            "strategy": "strategy if applicable",
            "value": "value if applicable",
            "mapping": {{
                "old_name": "new_name"
            }},
            "columns": ["column1", "column2"]
        }}
    ],

    "warnings": [
        "Potential issue that should not automatically be modified"
    ]
}}

If no preparation is necessary, return:

{{
    "reasoning_summary": "Dataset is already suitable for analysis.",
    "operations": [],
    "warnings": []
}}

Dataset context:

{context_json}
"""

    def _parse_response(
        self,
        response: str,
    ) -> dict[str, Any]:
        """
        Parse and validate the LLM response.
        """

        response = response.strip()

        if response.startswith("```"):
            response = response.replace(
                "```json",
                "",
            )
            response = response.replace(
                "```",
                "",
            )
            response = response.strip()

        try:
            result = json.loads(response)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Preparation plan must be a JSON object."
            )

        if "operations" not in result:
            raise ValueError(
                "Preparation plan must contain 'operations'."
            )

        if not isinstance(
            result["operations"],
            list,
        ):
            raise ValueError(
                "'operations' must be a list."
            )

        return result