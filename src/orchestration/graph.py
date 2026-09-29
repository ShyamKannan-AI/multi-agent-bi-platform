from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.orchestration.nodes import (
    final_report_node,
    governance_decision_node,
    human_review_node,
    join_post_processing_node,
    prepare_datasets_node,
    publish_stage_results_node,
    rejected_review_node,
    relationship_discovery_node,
)
from src.orchestration.state import ReadinessGraphState


def build_readiness_graph(
    pipeline_factory=None,
    relationship_discovery=None,
    checkpointer=None,
):
    """
    Build the LangGraph readiness workflow.

    Features:
    - conditional governance routing
    - human-in-the-loop interrupt/resume
    - checkpoint memory
    - parallel post-processing
    - fan-in synchronization
    - final unified report
    """

    graph = StateGraph(ReadinessGraphState)

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    graph.add_node(
        "prepare_datasets",
        lambda state: prepare_datasets_node(
            state,
            pipeline_factory=pipeline_factory,
        ),
    )

    graph.add_node(
        "governance_decision",
        governance_decision_node,
    )

    graph.add_node(
        "human_review",
        human_review_node,
    )

    graph.add_node(
        "post_processing",
        lambda state: {},
    )

    graph.add_node(
        "publish_stage_results",
        publish_stage_results_node,
    )

    graph.add_node(
        "discover_relationships",
        lambda state: relationship_discovery_node(
            state,
            relationship_discovery=relationship_discovery,
        ),
    )

    graph.add_node(
        "rejected_review",
        rejected_review_node,
    )

    graph.add_node(
        "join_post_processing",
        join_post_processing_node,
    )

    graph.add_node(
        "build_final_report",
        final_report_node,
    )

    # ---------------------------------------------------------
    # Main flow
    # ---------------------------------------------------------

    graph.add_edge(
        START,
        "prepare_datasets",
    )

    graph.add_edge(
        "prepare_datasets",
        "governance_decision",
    )

    # ---------------------------------------------------------
    # Governance routing
    # ---------------------------------------------------------

    def route_governance(
        state: dict[str, Any],
    ) -> str:
        """
        Read the governance route produced by
        governance_decision_node().

        IMPORTANT:
        The route is stored directly in:

            state["governance_route"]

        It is NOT stored inside:

            state["quality_decisions"]["route"]
        """

        route = state.get(
            "governance_route",
            "ignore",
        )

        if route == "review":
            return "human_review"

        if route in {
            "auto_action",
            "ignore",
        }:
            return "post_processing"

        # Defensive fallback.
        raise ValueError(
            f"Invalid governance route: {route}"
        )

    graph.add_conditional_edges(
        "governance_decision",
        route_governance,
        {
            "human_review": "human_review",
            "post_processing": "post_processing",
        },
    )

    # ---------------------------------------------------------
    # Human review routing
    # ---------------------------------------------------------

    def route_human_review(
        state: dict[str, Any],
    ) -> str:
        """
        Route the workflow after human approval/rejection.
        """

        decision = state.get(
            "human_approval",
            "reject",
        )

        if decision == "approve":
            return "post_processing"

        if decision == "reject":
            return "rejected_review"

        raise ValueError(
            f"Invalid human approval state: {decision}"
        )

    graph.add_conditional_edges(
        "human_review",
        route_human_review,
        {
            "post_processing": "post_processing",
            "rejected_review": "rejected_review",
        },
    )

    # ---------------------------------------------------------
    # Parallel post-processing
    # ---------------------------------------------------------

    # Both branches start after governance/HITL clearance.

    graph.add_edge(
        "post_processing",
        "publish_stage_results",
    )

    graph.add_edge(
        "post_processing",
        "discover_relationships",
    )

    # ---------------------------------------------------------
    # Fan-in
    # ---------------------------------------------------------

    # The join node waits for both post-processing branches.

    graph.add_edge(
        "publish_stage_results",
        "join_post_processing",
    )

    graph.add_edge(
        "discover_relationships",
        "join_post_processing",
    )

    graph.add_edge(
        "join_post_processing",
        "build_final_report",
    )

    # ---------------------------------------------------------
    # Terminal nodes
    # ---------------------------------------------------------

    graph.add_edge(
        "rejected_review",
        END,
    )

    graph.add_edge(
        "build_final_report",
        END,
    )

    # ---------------------------------------------------------
    # LangGraph checkpoint memory
    # ---------------------------------------------------------

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(
        checkpointer=checkpointer,
    )
