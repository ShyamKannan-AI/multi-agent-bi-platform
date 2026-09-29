from src.readiness import DataReadinessAssessor


def test_ready_dataset():

    assessor = DataReadinessAssessor()

    result = assessor.assess(
        quality_report={
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0,
            }
        },
        semantic_quality={
            "summary": {
                "review": 0,
            }
        },
        validation={
            "status": "success",
        },
        audit={
            "status": "success",
        },
        relationship_report={
            "relationship_count": 2,
        },
        integrity_reports=[],
    )

    assert result["status"] == "success"
    assert result["readiness_score"] == 100.0
    assert result["readiness_status"] == "ready"


def test_warning_dataset():

    assessor = DataReadinessAssessor()

    result = assessor.assess(
        quality_report={
            "summary": {
                "critical": 0,
                "warning": 2,
                "info": 1,
            }
        },
        semantic_quality={
            "summary": {
                "review": 0,
            }
        },
        validation={
            "status": "success",
        },
        audit={
            "status": "success",
        },
    )

    assert result["readiness_score"] == 90.0
    assert result["readiness_status"] == "ready"


def test_review_required():

    assessor = DataReadinessAssessor()

    result = assessor.assess(
        quality_report={
            "summary": {
                "critical": 0,
                "warning": 1,
                "info": 0,
            }
        },
        semantic_quality={
            "summary": {
                "review": 2,
            }
        },
        validation={
            "status": "success",
        },
        audit={
            "status": "success",
        },
    )

    assert result["readiness_score"] == 85.0
    assert result["readiness_status"] == "ready_with_warnings"


def test_critical_issue_not_ready():

    assessor = DataReadinessAssessor()

    result = assessor.assess(
        quality_report={
            "summary": {
                "critical": 1,
                "warning": 0,
                "info": 0,
            }
        },
        semantic_quality={
            "summary": {
                "review": 0,
            }
        },
        validation={
            "status": "success",
        },
        audit={
            "status": "success",
        },
    )

    assert result["readiness_score"] == 80.0
    assert result["readiness_status"] == "not_ready"


def test_integrity_violation():

    assessor = DataReadinessAssessor()

    result = assessor.assess(
        quality_report={
            "summary": {
                "critical": 0,
                "warning": 0,
                "info": 0,
            }
        },
        semantic_quality={
            "summary": {
                "review": 0,
            }
        },
        validation={
            "status": "success",
        },
        audit={
            "status": "success",
        },
        relationship_report={
            "relationship_count": 1,
        },
        integrity_reports=[
            {
                "orphan_values": 2,
            }
        ],
    )

    assert result["readiness_score"] == 90.0
    assert result["relationships"]["integrity_violations"] == 2