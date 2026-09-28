import json
from typing import Any

from src.llm.factory import get_llm


class DataQualityAgent:
    """
    Uses an LLM to interpret deterministic data-quality findings.

    The agent does not modify the dataframe.
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def analyze(
        self,
        quality_report: dict[str, Any],
        semantic_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Interpret deterministic quality findings using the LLM.
        """

        if not isinstance(quality_report, dict):
            raise TypeError(
                "quality_report must be a dictionary."
            )

        issues = quality_report.get("issues", [])

        if not issues:
            return {
                "status": "success",
                "interpretations": [],
                "summary": {
                    "total": 0,
                    "auto_action": 0,
                    "review": 0,
                    "ignore": 0,
                },
            }

        prompt = self._build_prompt(
            quality_report,
            semantic_context,
        )

        response = self.llm.invoke(prompt)

        content = response.content

        result = self._parse_response(content)

        return result

    @staticmethod
    def _build_prompt(
        quality_report: dict[str, Any],
        semantic_context: dict[str, Any] | None,
    ) -> str:
        return f"""
You are a data quality reasoning agent.

Your job is to interpret deterministic data-quality
findings using the available dataset semantic context.

IMPORTANT RULES:

1. Do not invent evidence.
2. Do not modify the dataset.
3. Do not automatically assume an anomaly is an error.
4. Distinguish genuine data-quality problems from
   potentially legitimate business values.
5. Recommend an action only when the evidence supports it.
6. If uncertain, recommend human review.
7. Return ONLY valid JSON.

For each issue classify it as one of:

- "auto_action"
  High-confidence issue where a deterministic correction
  may be appropriate.

- "review"
  Evidence suggests a problem but human confirmation
  is appropriate.

- "ignore"
  Evidence may represent legitimate business data.

Return this structure:

{{
    "status": "success",
    "interpretations": [
        {{
            "issue_type": "...",
            "column": "...",
            "classification": "auto_action | review | ignore",
            "reason": "...",
            "recommended_action": "...",
            "confidence": 0.0
        }}
    ],
    "summary": {{
        "total": 0,
        "auto_action": 0,
        "review": 0,
        "ignore": 0
    }}
}}

DATA QUALITY REPORT:

{json.dumps(quality_report, indent=2, default=str)}

SEMANTIC CONTEXT:

{json.dumps(semantic_context or {}, indent=2, default=str)}
""".strip()

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        """
        Parse and validate the LLM JSON response.
        """

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "LLM response must be a JSON object."
            )

        interpretations = result.get(
            "interpretations",
            [],
        )

        if not isinstance(interpretations, list):
            raise ValueError(
                "'interpretations' must be a list."
            )

        for interpretation in interpretations:
            if not isinstance(interpretation, dict):
                raise ValueError(
                    "Each interpretation must be an object."
                )

            classification = interpretation.get(
                "classification"
            )

            if classification not in {
                "auto_action",
                "review",
                "ignore",
            }:
                raise ValueError(
                    "Invalid quality classification."
                )

            confidence = interpretation.get(
                "confidence"
            )

            if not isinstance(
                confidence,
                (int, float),
            ):
                raise ValueError(
                    "Confidence must be numeric."
                )

            if not 0 <= confidence <= 1:
                raise ValueError(
                    "Confidence must be between 0 and 1."
                )

        return result