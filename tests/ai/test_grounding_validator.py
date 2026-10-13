import pandas as pd
import pytest

from src.ai.grounding_validator import (
    GroundingValidationError,
    validate_llm_grounding,
)


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "DEC-001",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-001",
                "signal": "High-value decreasing customer.",
                "direction": "DECREASING",
                "priority": "CRITICAL",
                "queue_rank": 1,
                "decision_readiness": "ACTION_CANDIDATE",
                "evidence_level": "HIGH",
                "signal_confidence": "HIGH",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "active_months": 4,
                "total_orders": 7,
                "lifetime_revenue": 809.21,
                "transition_count": 3,
                "direction_consistency": 1.0,
                "evidence_basis": (
                    "Three or more behavioral transitions observed."
                ),
                "recommended_action": (
                    "Prioritize customer value-risk intervention."
                ),
            }
        ]
    )


def _reasoning() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "DEC-001",
                "executive_summary": (
                    "High-value repeat customer shows decreasing behavior."
                ),
                "evidence_used": (
                    "HIGH evidence with 3 behavioral transitions, "
                    "direction consistency of 1.0, and HIGH confidence."
                ),
                "reasoning": (
                    "The observed decreasing behavior is supported by "
                    "multiple consistent transitions."
                ),
                "recommended_next_step": (
                    "Prioritize customer value-risk intervention."
                ),
                "risk_or_caveat": (
                    "Observational data does not establish causality "
                    "or intervention effectiveness."
                ),
                "human_review_required": "YES",
                "grounding_status": "GROUNDED",
            }
        ]
    )


def test_valid_grounding_passes():
    result = validate_llm_grounding(
        _context(),
        _reasoning(),
    )

    assert result is True


def test_decision_id_mismatch_is_rejected():
    reasoning = _reasoning().copy()
    reasoning.loc[0, "decision_id"] = "DEC-999"

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_recommendation_change_is_rejected():
    reasoning = _reasoning().copy()
    reasoning.loc[0, "recommended_next_step"] = (
        "Invent a new intervention."
    )

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_human_review_change_is_rejected():
    reasoning = _reasoning().copy()
    reasoning.loc[0, "human_review_required"] = "NO"

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_invalid_grounding_status_is_rejected():
    reasoning = _reasoning().copy()
    reasoning.loc[0, "grounding_status"] = "GROUNDED_WITH_ASSUMPTIONS"

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_missing_context_column_is_rejected():
    context = _context().drop(columns=["priority"])

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(context, _reasoning())


def test_missing_reasoning_column_is_rejected():
    reasoning = _reasoning().drop(columns=["reasoning"])

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_duplicate_decision_ids_are_rejected():
    reasoning = pd.concat(
        [_reasoning(), _reasoning()],
        ignore_index=True,
    )

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_null_reasoning_values_are_rejected():
    reasoning = _reasoning().copy()
    reasoning.loc[0, "reasoning"] = None

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_action_candidate_requires_human_review():
    reasoning = _reasoning().copy()
    reasoning.loc[0, "human_review_required"] = "NO"

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)


def test_recommendation_must_match_exactly():
    reasoning = _reasoning().copy()
    reasoning.loc[0, "recommended_next_step"] = (
        "Prioritize customer value-risk intervention"
    )

    with pytest.raises(GroundingValidationError):
        validate_llm_grounding(_context(), reasoning)