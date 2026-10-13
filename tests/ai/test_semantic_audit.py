from __future__ import annotations

import pandas as pd
import pytest

from src.ai.semantic_audit import (
    SemanticAuditError,
    audit_llm_reasoning,
)
from src.ai.semantic_audit import (
    SemanticAuditError,
    audit_llm_reasoning,
    _find_unsupported_claims,
)

def _context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "DEC-001",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "CUSTOMER-001",
                "direction": "DECREASING",
                "evidence_level": "HIGH",
                "signal_confidence": "HIGH",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "active_months": 4,
                "total_orders": 7,
                "lifetime_revenue": 809.21,
                "transition_count": 3,
                "direction_consistency": 1.0,
                "recommended_action": "Prioritize customer value-risk intervention.",
            }
        ]
    )


def _reasoning(
    *,
    executive_summary: str = "High-value repeat customer shows decreasing behavior.",
    evidence_used: str = (
        "Three behavioral transitions were observed with directional consistency of 1.0."
    ),
    reasoning: str = (
        "The observed transitions provide strong evidence of a consistent decreasing pattern."
    ),
    recommended_next_step: str = "Prioritize customer value-risk intervention.",
    risk_or_caveat: str = (
        "The observational data does not establish causality or intervention effectiveness."
    ),
    human_review_required: str = "YES",
    grounding_status: str = "GROUNDED",
    decision_id: str = "DEC-001",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": decision_id,
                "executive_summary": executive_summary,
                "evidence_used": evidence_used,
                "reasoning": reasoning,
                "recommended_next_step": recommended_next_step,
                "risk_or_caveat": risk_or_caveat,
                "human_review_required": human_review_required,
                "grounding_status": grounding_status,
            }
        ]
    )


def test_valid_reasoning_passes() -> None:
    result = audit_llm_reasoning(_context(), _reasoning())

    assert result is True


def test_unsupported_churn_claim_is_rejected() -> None:
    reasoning = _reasoning(
        reasoning="The customer is likely to churn because of the decreasing behavior."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_unsupported_causal_claim_is_rejected() -> None:
    reasoning = _reasoning(
        reasoning="The intervention caused the customer's behavior to decline."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_guaranteed_outcome_claim_is_rejected() -> None:
    reasoning = _reasoning(
        reasoning="This intervention will definitely improve customer value."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_customer_motivation_claim_is_rejected() -> None:
    reasoning = _reasoning(
        executive_summary="The customer is dissatisfied and wants to leave."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_empty_reasoning_is_rejected() -> None:
    reasoning = _reasoning(reasoning="   ")

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_missing_context_column_is_rejected() -> None:
    context = _context().drop(columns=["direction"])

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(context, _reasoning())


def test_missing_reasoning_column_is_rejected() -> None:
    reasoning = _reasoning().drop(columns=["reasoning"])

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_unknown_decision_id_is_rejected() -> None:
    reasoning = _reasoning(decision_id="DEC-UNKNOWN")

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_all_governed_decisions_must_be_covered() -> None:
    context = pd.concat(
        [
            _context(),
            _context().assign(
                decision_id="DEC-002",
                entity_id="CUSTOMER-002",
            ),
        ],
        ignore_index=True,
    )

    reasoning = _reasoning()

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(context, reasoning)


def test_unsupported_engagement_claim_is_rejected() -> None:
    reasoning = _reasoning(
        executive_summary="The customer shows positive engagement with the product."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_unsupported_significance_claim_is_rejected() -> None:
    reasoning = _reasoning(
        reasoning="The customer shows a statistically significant decline."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_unsupported_confirmation_claim_is_rejected() -> None:
    reasoning = _reasoning(
        reasoning="The observed behavior confirms the customer's declining intent."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)


def test_unsupported_proof_claim_is_rejected() -> None:
    reasoning = _reasoning(
        reasoning="The observed transitions prove that the customer will leave."
    )

    with pytest.raises(SemanticAuditError):
        audit_llm_reasoning(_context(), reasoning)

def test_confirmation_rule_does_not_match_unrelated_substrings() -> None:
    findings = _find_unsupported_claims(
        "The evidence provides useful context for further review."
    )

    assert not any(
        category == "unsupported_confirmation"
        for category, _ in findings
    )


def test_confirmation_rule_still_rejects_confirmation_claims() -> None:
    findings = _find_unsupported_claims(
        "The evidence confirms the observed pattern."
    )

    assert (
        "unsupported_confirmation",
        "confirm",
    ) in findings