import pandas as pd
import pytest

from src.orchestration.workflow import (
    AgenticReadinessWorkflow,
)


class MockPipeline:
    def run(self, file_path):
        dataframe = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "name": [
                    "Alice",
                    "Bob",
                    "Charlie",
                ],
            }
        )

        return {
            "prepared_dataframe": dataframe,
            "profile": {
                "rows": 3,
                "columns": 2,
            },
            "semantic_context": {
                "dataset_type": "customer_data",
            },
            "quality_report": {
                "summary": {
                    "critical": 0,
                    "warnings": 1,
                }
            },
            "semantic_quality": {
                "issues": []
            },
            "preparation_plan": {
                "operations": [],
                "review_required": [
                    {
                        "operation": "convert_datetime",
                        "column": "signup_date",
                    }
                ],
            },
            "validation": {
                "status": "passed",
            },
            "audit": {
                "status": "completed",
            },
            "readiness_report": {
                "readiness_status": "needs_review",
            },
        }


def test_human_review_interrupts_workflow():

    workflow = AgenticReadinessWorkflow(
        pipeline_factory=MockPipeline
    )

    result = workflow.run(
        {
            "customers": "customers.csv"
        },
        thread_id="hitl-test-1",
    )

    assert result["status"] == "pending_review"

    assert (
        result["thread_id"]
        == "hitl-test-1"
    )

    assert (
        result["review"]["type"]
        == "human_review"
    )

    assert (
        "customers"
        in result["review"]["review_items"]
    )


def test_human_approval_resumes_workflow():

    workflow = AgenticReadinessWorkflow(
        pipeline_factory=MockPipeline
    )

    workflow.run(
        {
            "customers": "customers.csv"
        },
        thread_id="hitl-test-2",
    )

    result = workflow.resume(
        thread_id="hitl-test-2",
        decision="approve",
    )

    assert result["status"] == "success"

    assert (
        result["human_review"]["decision"]
        == "approve"
    )

    assert (
        result["summary"]["human_decision"]
        == "approve"
    )


def test_human_rejection_stops_workflow():

    workflow = AgenticReadinessWorkflow(
        pipeline_factory=MockPipeline
    )

    workflow.run(
        {
            "customers": "customers.csv"
        },
        thread_id="hitl-test-3",
    )

    result = workflow.resume(
        thread_id="hitl-test-3",
        decision="reject",
    )

    assert (
        result["status"]
        == "human_rejected"
    )

    assert (
        result["summary"]["human_decision"]
        == "reject"
    )


def test_invalid_human_decision():

    workflow = AgenticReadinessWorkflow(
        pipeline_factory=MockPipeline
    )

    with pytest.raises(ValueError):

        workflow.resume(
            thread_id="invalid-thread",
            decision="maybe",
        )


def test_stage_results_are_published():

    workflow = AgenticReadinessWorkflow(
        pipeline_factory=MockPipeline
    )

    result = workflow.run(
        {
            "customers": "customers.csv"
        },
        thread_id="stage-test-1",
    )

    # This workflow requires review, so stage results
    # are exposed after approval.

    assert result["status"] == "pending_review"

    final_result = workflow.resume(
        thread_id="stage-test-1",
        decision="approve",
    )

    assert "stage_results" in final_result

    assert (
        "customers"
        in final_result["stage_results"]
    )

    customer_stages = (
        final_result["stage_results"]["customers"]
    )

    assert (
        "profiling"
        in customer_stages
    )

    assert (
        "quality_analysis"
        in customer_stages
    )

    assert (
        "validation"
        in customer_stages
    )


def test_thread_state_is_preserved():

    workflow = AgenticReadinessWorkflow(
        pipeline_factory=MockPipeline
    )

    first_result = workflow.run(
        {
            "customers": "customers.csv"
        },
        thread_id="memory-test-1",
    )

    assert (
        first_result["status"]
        == "pending_review"
    )

    second_result = workflow.resume(
        thread_id="memory-test-1",
        decision="approve",
    )

    assert second_result["status"] == "success"

    assert (
        second_result["human_review"]["decision"]
        == "approve"
    )
    