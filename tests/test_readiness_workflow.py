import pandas as pd

from src.orchestration import UnifiedReadinessWorkflow


class MockPipeline:
    def run(self, file_path):
        if "customers" in file_path:
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

        else:
            dataframe = pd.DataFrame(
                {
                    "order_id": [101, 102, 103],
                    "customer_id": [1, 2, 3],
                    "amount": [100, 200, 300],
                }
            )

        return {
            "prepared_dataframe": dataframe,
            "readiness_report": {
                "readiness_score": 100,
                "readiness_status": "ready",
            },
        }


def mock_pipeline_factory():
    return MockPipeline()


def test_workflow_processes_multiple_datasets():

    workflow = UnifiedReadinessWorkflow(
        pipeline_factory=mock_pipeline_factory,
    )

    result = workflow.run(
        {
            "customers": "customers.csv",
            "orders": "orders.csv",
        }
    )

    assert result["status"] == "success"

    assert result["dataset_count"] == 2

    assert "customers" in result["datasets"]
    assert "orders" in result["datasets"]

    assert (
        "customers"
        in result["prepared_datasets"]
    )

    assert (
        "orders"
        in result["prepared_datasets"]
    )


def test_workflow_discovers_relationships():

    workflow = UnifiedReadinessWorkflow(
        pipeline_factory=mock_pipeline_factory,
    )

    result = workflow.run(
        {
            "customers": "customers.csv",
            "orders": "orders.csv",
        }
    )

    relationships = result["relationships"]

    assert relationships["status"] == "success"

    assert relationships["relationship_count"] >= 1


def test_workflow_builds_summary():

    workflow = UnifiedReadinessWorkflow(
        pipeline_factory=mock_pipeline_factory,
    )

    result = workflow.run(
        {
            "customers": "customers.csv",
            "orders": "orders.csv",
        }
    )

    summary = result["summary"]

    assert summary["datasets_processed"] == 2

    assert (
        summary["relationships_discovered"]
        >= 1
    )

    assert (
        summary["readiness_statuses"]["customers"]
        == "ready"
    )


def test_workflow_rejects_empty_input():

    workflow = UnifiedReadinessWorkflow(
        pipeline_factory=mock_pipeline_factory,
    )

    try:
        workflow.run({})
    except ValueError as exc:
        assert "cannot be empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_workflow_rejects_non_dictionary_input():

    workflow = UnifiedReadinessWorkflow(
        pipeline_factory=mock_pipeline_factory,
    )

    try:
        workflow.run([])
    except TypeError as exc:
        assert "dictionary" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected TypeError."
        )