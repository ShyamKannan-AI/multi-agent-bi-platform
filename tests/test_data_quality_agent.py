import json

import pytest

from src.agents import DataQualityAgent


class MockResponse:
    def __init__(self, content):
        self.content = content


class MockLLM:
    def __init__(self, response):
        self.response = response
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return MockResponse(self.response)


def test_quality_agent_interprets_quality_findings():
    response = json.dumps(
        {
            "status": "success",
            "interpretations": [
                {
                    "issue_type": "missing_values",
                    "column": "age",
                    "classification": "review",
                    "reason": (
                        "Missing age values may require "
                        "business-specific handling."
                    ),
                    "recommended_action": (
                        "Review missing age records."
                    ),
                    "confidence": 0.91,
                }
            ],
            "summary": {
                "total": 1,
                "auto_action": 0,
                "review": 1,
                "ignore": 0,
            },
        }
    )

    llm = MockLLM(response)

    agent = DataQualityAgent(llm=llm)

    quality_report = {
        "status": "success",
        "issues": [
            {
                "type": "missing_values",
                "column": "age",
                "severity": "warning",
                "evidence": {
                    "missing_count": 10,
                    "missing_percentage": 5.0,
                },
            }
        ],
    }

    result = agent.analyze(
        quality_report=quality_report,
        semantic_context={
            "columns": {
                "age": {
                    "semantic_role": "customer attribute"
                }
            }
        },
    )

    assert result["status"] == "success"
    assert len(result["interpretations"]) == 1

    interpretation = result["interpretations"][0]

    assert interpretation["column"] == "age"
    assert interpretation["classification"] == "review"
    assert interpretation["confidence"] == 0.91


def test_quality_agent_handles_no_quality_issues():
    llm = MockLLM("{}")

    agent = DataQualityAgent(llm=llm)

    result = agent.analyze(
        quality_report={
            "status": "success",
            "issues": [],
        }
    )

    assert result["status"] == "success"
    assert result["interpretations"] == []
    assert result["summary"]["total"] == 0

    # LLM should not be called when there are no issues.
    assert llm.last_prompt is None


def test_quality_agent_rejects_invalid_classification():
    response = json.dumps(
        {
            "status": "success",
            "interpretations": [
                {
                    "issue_type": "outliers",
                    "column": "revenue",
                    "classification": "delete",
                    "reason": "Bad value",
                    "recommended_action": "Delete",
                    "confidence": 0.9,
                }
            ],
            "summary": {
                "total": 1,
                "auto_action": 0,
                "review": 0,
                "ignore": 0,
            },
        }
    )

    llm = MockLLM(response)

    agent = DataQualityAgent(llm=llm)

    with pytest.raises(ValueError):
        agent.analyze(
            quality_report={
                "issues": [
                    {
                        "type": "outliers",
                        "column": "revenue",
                    }
                ]
            }
        )


def test_quality_agent_rejects_invalid_json():
    llm = MockLLM("this is not json")

    agent = DataQualityAgent(llm=llm)

    with pytest.raises(ValueError):
        agent.analyze(
            quality_report={
                "issues": [
                    {
                        "type": "outliers",
                        "column": "revenue",
                    }
                ]
            }
        )


def test_quality_agent_rejects_invalid_confidence():
    response = json.dumps(
        {
            "status": "success",
            "interpretations": [
                {
                    "issue_type": "outliers",
                    "column": "revenue",
                    "classification": "review",
                    "reason": "Needs investigation",
                    "recommended_action": "Review",
                    "confidence": 2.0,
                }
            ],
            "summary": {
                "total": 1,
                "auto_action": 0,
                "review": 1,
                "ignore": 0,
            },
        }
    )

    llm = MockLLM(response)

    agent = DataQualityAgent(llm=llm)

    with pytest.raises(ValueError):
        agent.analyze(
            quality_report={
                "issues": [
                    {
                        "type": "outliers",
                        "column": "revenue",
                    }
                ]
            }
        )