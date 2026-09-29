import json
from typing import Any

import pandas as pd

from src.llm.factory import get_llm


class DataPreparationAgent:
    """
    Creates an evidence-aware data preparation plan.

    The LLM proposes operations.
    The deterministic safety layer decides which operations
    are eligible for automatic execution.

    This agent never directly modifies the dataframe.
    """

    ALLOWED_OPERATIONS = {
        "rename_columns",
        "drop_duplicates",
        "fill_missing",
        "convert_datetime",
        "strip_strings",
    }

    ALLOWED_CLASSIFICATIONS = {
        "auto_action",
        "review",
        "ignore",
    }

    ALLOWED_FILL_STRATEGIES = {
        "mean",
        "median",
        "mode",
        "value",
        "ffill",
        "bfill",
    }

    AUTO_ACTION_THRESHOLD = 0.80

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    # ============================================================
    # CREATE PLAN
    # ============================================================

    def create_plan(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
        quality_report: dict[str, Any] | None = None,
        semantic_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create an evidence-aware preparation plan.
        """

        # --------------------------------------------------------
        # Validate dataframe FIRST
        # --------------------------------------------------------

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a Pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Cannot create a preparation plan "
                "for an empty dataframe."
            )

        # --------------------------------------------------------
        # Validate profile
        # --------------------------------------------------------

        if not isinstance(profile, dict):
            raise TypeError(
                "profile must be a dictionary."
            )

        if not profile:
            raise ValueError(
                "profile cannot be empty."
            )

        # --------------------------------------------------------
        # Validate semantic context
        # --------------------------------------------------------

        if not isinstance(semantic_context, dict):
            raise TypeError(
                "semantic_context must be a dictionary."
            )

        if not semantic_context:
            raise ValueError(
                "semantic_context cannot be empty."
            )

        # --------------------------------------------------------
        # Default quality context
        # --------------------------------------------------------

        if quality_report is None:
            quality_report = {
                "status": "success",
                "issues": [],
                "summary": {
                    "total_issues": 0,
                    "critical": 0,
                    "warning": 0,
                    "info": 0,
                },
            }

        if semantic_quality is None:
            semantic_quality = {
                "status": "success",
                "interpretations": [],
                "summary": {
                    "total": 0,
                    "auto_action": 0,
                    "review": 0,
                    "ignore": 0,
                },
            }

        # --------------------------------------------------------
        # Build prompt
        # --------------------------------------------------------

        prompt = self._build_prompt(
            df=df,
            profile=profile,
            semantic_context=semantic_context,
            quality_report=quality_report,
            semantic_quality=semantic_quality,
        )

        # --------------------------------------------------------
        # Ask LLM
        # --------------------------------------------------------

        response = self.llm.invoke(prompt)

        if not hasattr(response, "content"):
            raise ValueError(
                "LLM response does not contain content."
            )

        proposed_plan = self._parse_response(
            response.content
        )

        # --------------------------------------------------------
        # Deterministic safety filtering
        # --------------------------------------------------------

        return self._apply_safety_filter(
            proposed_plan=proposed_plan,
            semantic_quality=semantic_quality,
            dataframe_columns=[
                str(column)
                for column in df.columns
            ],
        )

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
        quality_report: dict[str, Any] | None = None,
        semantic_quality: dict[str, Any] | None = None,
    ) -> str:
        """
        Build the LLM prompt.

        quality_report and semantic_quality remain optional
        for compatibility with existing callers/tests.
        """

        if quality_report is None:
            quality_report = {
                "status": "success",
                "issues": [],
                "summary": {
                    "total_issues": 0,
                    "critical": 0,
                    "warning": 0,
                    "info": 0,
                },
            }

        if semantic_quality is None:
            semantic_quality = {
                "status": "success",
                "interpretations": [],
                "summary": {
                    "total": 0,
                    "auto_action": 0,
                    "review": 0,
                    "ignore": 0,
                },
            }

        dataframe_columns = [
            str(column)
            for column in df.columns
        ]

        return f"""
You are a conservative data preparation agent.

Your job is to propose SAFE and EVIDENCE-BASED preparation
operations for the provided dataframe.

IMPORTANT RULES:

1. Do NOT modify the dataframe.
2. Do NOT invent columns.
3. Do NOT invent data-quality problems.
4. Only reference columns that exist in the dataframe.
5. Only use the allowed operations listed below.
6. Every operation must contain an explicit "operation" field.
7. The "operation" field MUST NEVER be null.
8. The "operation" field MUST exactly match one of the allowed operations.
9. If there is no safe preparation action, return an empty operations list.
10. Do not return explanatory text outside the JSON object.
11. Return valid JSON only.
12. Do not perform joins.
13. Do not perform statistical modelling.
14. Do not perform prediction.
15. Do not perform forecasting.
16. Do not invent replacement values without evidence.
17. Operations classified as "review" or "ignore" must not be
    automatically executed.
18. "auto_action" requires confidence >= 0.80.

ALLOWED OPERATIONS:

- rename_columns
- drop_duplicates
- fill_missing
- convert_datetime
- strip_strings

ALLOWED CLASSIFICATIONS:

- auto_action
- review
- ignore

ALLOWED FILL STRATEGIES:

- mean
- median
- mode
- value
- ffill
- bfill

DATAFRAME COLUMNS:

{json.dumps(dataframe_columns, indent=2, default=str)}

RETURN EXACTLY THIS JSON STRUCTURE:

{{
    "reasoning_summary": "Brief explanation.",
    "operations": [
        {{
            "operation": "fill_missing",
            "column": "column_name",
            "strategy": "median",
            "value": null,
            "reason": "Reason supported by quality evidence.",
            "confidence": 0.95,
            "classification": "auto_action",
            "source_issue_type": "missing_values",
            "source_column": "column_name"
        }}
    ],
    "warnings": []
}}

If no safe operation exists:

{{
    "reasoning_summary": "No safe preparation operation is required.",
    "operations": [],
    "warnings": []
}}

DATASET PROFILE:

{json.dumps(profile, indent=2, default=str)}

SEMANTIC CONTEXT:

{json.dumps(semantic_context, indent=2, default=str)}

DETERMINISTIC QUALITY REPORT:

{json.dumps(quality_report, indent=2, default=str)}

SEMANTIC QUALITY INTERPRETATION:

{json.dumps(semantic_quality, indent=2, default=str)}

Create the safest preparation plan supported by the
available evidence.
""".strip()

    # ============================================================
    # PARSE RESPONSE
    # ============================================================

    @classmethod
    def _parse_response(
        cls,
        content: str,
    ) -> dict[str, Any]:
        """
        Parse and validate the LLM response.
        """

        if not isinstance(content, str):
            raise TypeError(
                "LLM response content must be a string."
            )

        try:
            result = json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Preparation plan must be a JSON object."
            )

        # Existing contract:
        # missing operations must raise an error.
        if "operations" not in result:
            raise ValueError(
                "Preparation plan must contain 'operations'."
            )

        operations = result["operations"]

        if not isinstance(operations, list):
            raise ValueError(
                "'operations' must be a list."
            )

        for operation in operations:

            if not isinstance(operation, dict):
                raise ValueError(
                    "Each operation must be an object."
                )

            # Support both:
            # "operation": "fill_missing"
            #
            # and the older:
            # "type": "fill_missing"
            operation_type = operation.get(
                "operation"
            )

            if operation_type is None:
                operation_type = operation.get(
                    "type"
                )

                if operation_type is not None:
                    operation["operation"] = operation_type

            if operation_type not in cls.ALLOWED_OPERATIONS:
                raise ValueError(
                    f"Unsupported operation: {operation_type}"
                )

            classification = operation.get(
                "classification"
            )

            if classification is not None:
                if classification not in cls.ALLOWED_CLASSIFICATIONS:
                    raise ValueError(
                        "Invalid operation classification."
                    )

            confidence = operation.get(
                "confidence"
            )

            if confidence is not None:

                if not isinstance(
                    confidence,
                    (int, float),
                ):
                    raise ValueError(
                        "Operation confidence must be numeric."
                    )

                if not 0 <= confidence <= 1:
                    raise ValueError(
                        "Operation confidence must be between 0 and 1."
                    )

            if operation_type == "fill_missing":

                strategy = operation.get(
                    "strategy"
                )

                if strategy is not None:

                    if strategy not in cls.ALLOWED_FILL_STRATEGIES:
                        raise ValueError(
                            f"Unsupported fill strategy: {strategy}"
                        )

        return result

    # ============================================================
    # SAFETY FILTER
    # ============================================================

    @classmethod
    def _apply_safety_filter(
        cls,
        proposed_plan: dict[str, Any],
        semantic_quality: dict[str, Any],
        dataframe_columns: list[str],
    ) -> dict[str, Any]:
        """
        Deterministically decide which proposed operations
        are safe to execute automatically.

        This is the final safety boundary before DataCleaner.
        """

        operations = proposed_plan.get(
            "operations",
            [],
        )

        executable = []
        review_required = []
        ignored = []

        semantic_interpretations = semantic_quality.get(
            "interpretations",
            [],
        )

        interpretation_map = {}

        for interpretation in semantic_interpretations:

            if not isinstance(
                interpretation,
                dict,
            ):
                continue

            issue_type = interpretation.get(
                "issue_type"
            )

            column = interpretation.get(
                "column"
            )

            interpretation_map[
                (issue_type, column)
            ] = interpretation

        for operation in operations:

            if not isinstance(
                operation,
                dict,
            ):
                ignored.append({
                    "operation": None,
                    "classification": "ignore",
                    "safety_reason": (
                        "Operation was not a valid object."
                    ),
                })

                continue

            # ----------------------------------------------------
            # Final operation-type validation
            # ----------------------------------------------------

            operation_type = operation.get(
                "operation"
            )

            if operation_type is None:
                operation_type = operation.get(
                    "type"
                )

                if operation_type is not None:
                    operation["operation"] = operation_type

            if operation_type not in cls.ALLOWED_OPERATIONS:

                ignored.append({
                    **operation,
                    "operation": operation_type,
                    "classification": "ignore",
                    "safety_reason": (
                        "Unsupported or missing cleaning "
                        f"operation: {operation_type}"
                    ),
                })

                continue

            column = operation.get(
                "column"
            )

            classification = operation.get(
                "classification"
            )

            if classification is None:
                classification = "auto_action"

            confidence = operation.get(
                "confidence"
            )

            if confidence is None:
                confidence = 1.0

            try:
                confidence = float(confidence)

            except (
                TypeError,
                ValueError,
            ):

                review_required.append({
                    **operation,
                    "classification": "review",
                    "safety_reason": (
                        "Operation confidence is not numeric."
                    ),
                })

                continue

            source_issue_type = operation.get(
                "source_issue_type"
            )

            source_column = operation.get(
                "source_column"
            )

            # ----------------------------------------------------
            # Column safety
            # ----------------------------------------------------

            if (
                column is not None
                and str(column) not in dataframe_columns
            ):

                review_required.append({
                    **operation,
                    "classification": "review",
                    "confidence": confidence,
                    "safety_reason": (
                        "Column does not exist in "
                        "the dataframe."
                    ),
                })

                continue

            # ----------------------------------------------------
            # Explicit ignore
            # ----------------------------------------------------

            if classification == "ignore":

                ignored.append({
                    **operation,
                    "classification": "ignore",
                    "confidence": confidence,
                    "safety_reason": (
                        "Quality analysis classified "
                        "this issue as ignorable."
                    ),
                })

                continue

            # ----------------------------------------------------
            # Explicit review
            # ----------------------------------------------------

            if classification == "review":

                review_required.append({
                    **operation,
                    "classification": "review",
                    "confidence": confidence,
                    "safety_reason": (
                        "Quality analysis requires "
                        "human review."
                    ),
                })

                continue

            # ----------------------------------------------------
            # Automatic action
            # ----------------------------------------------------

            if classification == "auto_action":

                if confidence < cls.AUTO_ACTION_THRESHOLD:

                    review_required.append({
                        **operation,
                        "classification": "review",
                        "confidence": confidence,
                        "safety_reason": (
                            "Confidence is below the "
                            "automatic execution threshold "
                            f"of {cls.AUTO_ACTION_THRESHOLD:.2f}."
                        ),
                    })

                    continue

                # ------------------------------------------------
                # Cross-check semantic quality interpretation
                # ------------------------------------------------

                if (
                    source_issue_type is not None
                    and source_column is not None
                ):

                    key = (
                        source_issue_type,
                        source_column,
                    )

                    interpretation = interpretation_map.get(
                        key
                    )

                    if interpretation is not None:

                        semantic_classification = (
                            interpretation.get(
                                "classification"
                            )
                        )

                        semantic_confidence = (
                            interpretation.get(
                                "confidence"
                            )
                        )

                        if semantic_classification == "ignore":

                            ignored.append({
                                **operation,
                                "classification": "ignore",
                                "confidence": confidence,
                                "safety_reason": (
                                    "Semantic quality analysis "
                                    "classified the source issue "
                                    "as ignorable."
                                ),
                            })

                            continue

                        if semantic_classification == "review":

                            review_required.append({
                                **operation,
                                "classification": "review",
                                "confidence": confidence,
                                "safety_reason": (
                                    "Semantic quality analysis "
                                    "requires human review."
                                ),
                            })

                            continue

                        if (
                            isinstance(
                                semantic_confidence,
                                (int, float),
                            )
                            and semantic_confidence
                            < cls.AUTO_ACTION_THRESHOLD
                        ):

                            review_required.append({
                                **operation,
                                "classification": "review",
                                "confidence": semantic_confidence,
                                "safety_reason": (
                                    "Semantic quality confidence "
                                    "is below the automatic "
                                    "execution threshold."
                                ),
                            })

                            continue

                executable.append({
                    **operation,
                    "operation": operation_type,
                    "classification": "auto_action",
                    "confidence": confidence,
                })

                continue

            # ----------------------------------------------------
            # Unknown classification
            # ----------------------------------------------------

            review_required.append({
                **operation,
                "classification": "review",
                "confidence": confidence,
                "safety_reason": (
                    "Unknown operation classification."
                ),
            })

        return {
            "status": "success",
            "operations": executable,
            "review_required": review_required,
            "ignored_operations": ignored,
            "safety_summary": {
                "proposed": len(operations),
                "executable": len(executable),
                "review_required": len(review_required),
                "ignored": len(ignored),
            },
        }