from typing import Any


class DataReadinessAssessor:
    """
    Deterministically evaluates whether datasets are ready
    for downstream analysis or machine-learning workflows.

    The assessor does not modify data and does not use an LLM.
    """

    def assess(
        self,
        quality_report: dict[str, Any],
        semantic_quality: dict[str, Any],
        validation: dict[str, Any],
        audit: dict[str, Any],
        relationship_report: dict[str, Any] | None = None,
        integrity_reports: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:

        quality_report = quality_report or {}
        semantic_quality = semantic_quality or {}
        validation = validation or {}
        audit = audit or {}

        relationship_report = relationship_report or {}
        integrity_reports = integrity_reports or []

        # --------------------------------------------------------
        # Quality evidence
        # --------------------------------------------------------

        quality_summary = quality_report.get(
            "summary",
            {},
        )

        critical = int(
            quality_summary.get("critical", 0)
        )

        warnings = int(
            quality_summary.get("warning", 0)
        )

        info = int(
            quality_summary.get("info", 0)
        )

        # --------------------------------------------------------
        # Semantic quality interpretation
        # --------------------------------------------------------

        semantic_summary = semantic_quality.get(
            "summary",
            {},
        )

        review_count = int(
            semantic_summary.get("review", 0)
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        validation_status = validation.get(
            "status",
            "unknown",
        )

        validation_passed = (
            validation_status == "success"
        )

        # --------------------------------------------------------
        # Preparation audit
        # --------------------------------------------------------

        audit_status = audit.get(
            "status",
            "success",
        )

        audit_passed = (
            audit_status == "success"
        )

        # --------------------------------------------------------
        # Relationship integrity
        # --------------------------------------------------------

        integrity_violations = sum(
            int(
                report.get(
                    "orphan_values",
                    0,
                )
            )
            for report in integrity_reports
        )

        relationship_count = int(
            relationship_report.get(
                "relationship_count",
                0,
            )
        )

        # --------------------------------------------------------
        # Score
        # --------------------------------------------------------

        score = 100.0

        # Critical issues have the largest impact.
        score -= critical * 20

        # Warnings indicate unresolved quality concerns.
        score -= warnings * 5

        # Semantic issues requiring human review.
        score -= review_count * 5

        # Referential integrity problems.
        score -= min(
            integrity_violations * 5,
            20,
        )

        # Failed validation/audit is a major concern.
        if not validation_passed:
            score -= 15

        if not audit_passed:
            score -= 10

        score = max(
            0.0,
            min(100.0, score),
        )

        score = round(score, 2)

        # --------------------------------------------------------
        # Determine readiness status
        # --------------------------------------------------------

        if (
            critical > 0
            or not validation_passed
            or not audit_passed
        ):
            status = "not_ready"

        elif score >= 90 and review_count == 0:
            status = "ready"

        elif score >= 75:
            status = "ready_with_warnings"

        else:
            status = "needs_review"

        # --------------------------------------------------------
        # Build recommendations
        # --------------------------------------------------------

        recommendations = []

        if critical > 0:
            recommendations.append(
                "Resolve critical data-quality issues."
            )

        if warnings > 0:
            recommendations.append(
                "Review remaining data-quality warnings."
            )

        if review_count > 0:
            recommendations.append(
                "Review quality findings requiring human confirmation."
            )

        if integrity_violations > 0:
            recommendations.append(
                "Resolve referential-integrity violations."
            )

        if not validation_passed:
            recommendations.append(
                "Resolve preparation validation failures."
            )

        if not audit_passed:
            recommendations.append(
                "Resolve preparation audit failures."
            )

        if not recommendations:
            recommendations.append(
                "Dataset is ready for downstream analysis."
            )

        # --------------------------------------------------------
        # Final report
        # --------------------------------------------------------

        return {
            "status": "success",
            "readiness_score": score,
            "readiness_status": status,
            "quality": {
                "critical": critical,
                "warnings": warnings,
                "info": info,
            },
            "semantic_quality": {
                "review_required": review_count,
            },
            "relationships": {
                "discovered": relationship_count,
                "integrity_violations": integrity_violations,
            },
            "validation": {
                "passed": validation_passed,
            },
            "audit": {
                "passed": audit_passed,
            },
            "recommendations": recommendations,
        }