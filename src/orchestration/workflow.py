from __future__ import annotations

from typing import Any

from langgraph.types import Command

from src.orchestration.graph import build_readiness_graph


class AgenticReadinessWorkflow:
    """
    Public interface for the LangGraph-based autonomous
    data-readiness workflow.

    Handles:
    - workflow execution
    - thread-based checkpointing
    - human-in-the-loop interruption
    - human approval/rejection
    - stable public response formatting
    """

    def __init__(
        self,
        pipeline_factory=None,
        relationship_discovery=None,
        checkpointer=None,
    ):
        self.graph = build_readiness_graph(
            pipeline_factory=pipeline_factory,
            relationship_discovery=relationship_discovery,
            checkpointer=checkpointer,
        )

    def run(
        self,
        dataset_paths: dict[str, str],
        thread_id: str = "default",
    ) -> dict[str, Any]:
        if not isinstance(dataset_paths, dict):
            raise ValueError(
                "dataset_paths must be a dictionary."
            )

        if not dataset_paths:
         raise ValueError(
        "Dataset paths cannot be empty."
    )

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = self.graph.invoke(
            {
                "dataset_paths": dataset_paths,
            },
            config=config,
        )

        interrupts = result.get(
            "__interrupt__",
            [],
        )

        if interrupts:
            review_payload = self._extract_interrupt_value(
                interrupts[0]
            )

            return {
                "status": "pending_review",
                "thread_id": thread_id,
                "review": review_payload,
            }

        return self._normalize_completed_result(
            result,
            thread_id,
        )

    def resume(
        self,
        thread_id: str,
        decision: str,
    ) -> dict[str, Any]:
        if decision not in {
            "approve",
            "reject",
        }:
            raise ValueError(
                "Decision must be 'approve' or 'reject'."
            )

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = self.graph.invoke(
            Command(resume=decision),
            config=config,
        )

        interrupts = result.get(
            "__interrupt__",
            [],
        )

        if interrupts:
            review_payload = self._extract_interrupt_value(
                interrupts[0]
            )

            return {
                "status": "pending_review",
                "thread_id": thread_id,
                "review": review_payload,
            }

        return self._normalize_resume_result(
            result,
            thread_id,
            decision,
        )

    @staticmethod
    def _extract_interrupt_value(
        interrupt_value: Any,
    ) -> dict[str, Any]:
        value = getattr(
            interrupt_value,
            "value",
            interrupt_value,
        )

        if isinstance(value, dict):
            review = dict(value)
        else:
            review = {
                "message": str(value),
            }

        raw_items = review.get(
            "review_items",
            [],
        )

        if isinstance(raw_items, list):
            normalized_items = {}

            for item in raw_items:
                if not isinstance(item, dict):
                    continue

                dataset = item.get(
                    "dataset"
                )

                if dataset is None:
                    continue

                normalized_items[dataset] = item.get(
                    "operations",
                    [],
                )

            review["review_items"] = normalized_items

        elif not isinstance(raw_items, dict):
            review["review_items"] = {}

        return review

    @staticmethod
    def _normalize_completed_result(
        result: dict[str, Any],
        thread_id: str,
    ) -> dict[str, Any]:
        final_report = result.get(
            "final_report",
            {},
        )

        if not isinstance(final_report, dict):
            final_report = {}

        status = final_report.get(
            "status",
            "success",
        )

        if status == "completed":
            status = "success"

        return {
            "status": status,
            "thread_id": thread_id,
            "final_report": final_report,
            "stage_results": result.get(
                "stage_results",
                {},
            ),
            "relationship_report": result.get(
                "relationship_report",
                {},
            ),
            "summary": final_report.get(
                "summary",
                {},
            ),
        }

    @staticmethod
    def _normalize_resume_result(
        result: dict[str, Any],
        thread_id: str,
        decision: str,
    ) -> dict[str, Any]:
        final_report = result.get(
            "final_report",
            {},
        )

        if not isinstance(final_report, dict):
            final_report = {}

        # --------------------------------------------------
        # HUMAN REJECTION
        # --------------------------------------------------
        if decision == "reject":
            existing_review = result.get(
                "human_review",
                {},
            )

            if not isinstance(existing_review, dict):
                existing_review = {}

            human_review = {
                **existing_review,
                "decision": "reject",
                "status": "human_rejected",
            }

            summary = final_report.get(
                "summary",
                {},
            )

            if not isinstance(summary, dict):
                summary = {}

            summary = {
                **summary,
                "human_decision": "reject",
            }

            return {
                "status": "human_rejected",
                "thread_id": thread_id,
                "summary": summary,
                "final_report": final_report,
                "human_review": human_review,
            }

        # --------------------------------------------------
        # HUMAN APPROVAL
        # --------------------------------------------------
        existing_review = result.get(
            "human_review",
            {},
        )

        if not isinstance(existing_review, dict):
            existing_review = {}

        human_review = {
            **existing_review,
            "decision": "approve",
            "status": "approved",
        }

        summary = final_report.get(
            "summary",
            {},
        )

        if not isinstance(summary, dict):
            summary = {}

        summary = {
            **summary,
            "human_decision": "approve",
        }

        return {
            "status": "success",
            "thread_id": thread_id,
            "final_report": final_report,
            "stage_results": result.get(
                "stage_results",
                {},
            ),
            "relationship_report": result.get(
                "relationship_report",
                {},
            ),
            "summary": summary,
            "human_review": human_review,
        }