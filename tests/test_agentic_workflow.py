import pandas as pd

from src.orchestration.nodes import (
    prepare_datasets_node,
)
from src.orchestration.workflow import (
    AgenticReadinessWorkflow,
)


class MockPipeline:

    def run(self, file_path):

        if "customers" in file_path:

            df = pd.DataFrame(
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

            df = pd.DataFrame(
                {
                    "order_id": [101, 102, 103],
                    "customer_id": [1, 2, 3],
                    "amount": [100, 200, 300],
                }
            )

        return {
            "prepared_dataframe": df,
            "readiness_report": {
                "readiness_score": 100,
                "readiness_status": "ready",
            },
        }


def test_langgraph_builds():

    from src.orchestration.graph import (
        build_readiness_graph,
    )

    graph = build_readiness_graph()

    assert graph is not None


def test_prepare_node():

    import src.orchestration.nodes as nodes

    original_pipeline = (
        nodes.DataPreparationPipeline
    )

    nodes.DataPreparationPipeline = MockPipeline

    try:

        result = prepare_datasets_node(
            {
                "dataset_paths": {
                    "customers": "customers.csv",
                    "orders": "orders.csv",
                }
            }
        )

        assert len(
            result["dataset_reports"]
        ) == 2

        assert len(
            result["prepared_datasets"]
        ) == 2

    finally:

        nodes.DataPreparationPipeline = (
            original_pipeline
        )


def test_agentic_workflow_rejects_empty_input():

    workflow = AgenticReadinessWorkflow()

    try:
        workflow.run({})

    except ValueError as exc:

        assert "empty" in str(exc).lower()

    else:

        raise AssertionError(
            "Expected ValueError."
        )