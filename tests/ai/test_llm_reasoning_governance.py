from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ai.llm_reasoning import (
    LLMReasoningError,
    build_llm_reasoning,
)


class FakeProvider:
    """Deterministic provider used to test governance integration."""

    def __init__(self, response: dict):
        self.response = response

    def generate_json(self, prompt: str) -> str:
        return json.dumps(self.response)


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "DEC-001",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "CUSTOMER-001",
                "signal": "customer value-risk",
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
                    "Three or more behavioral transitions observed "
                    "(transition_count=3) with directional consistency "
                    "of 1.0."
                ),
                "recommended_action": (
                    "Prioritize customer value-risk intervention."
                ),
            }
        ]
    )


def _valid_response() -> dict:
    return {
        "decision_id": "DEC-001",
        "executive_summary": (
            "This high-value repeat customer shows decreasing behavior "
            "with strong longitudinal evidence."
        ),
        "evidence_used": (
            "The decision is supported by three behavioral transitions "
            "with directional consistency of 1.0."
        ),
        "reasoning": (
            "The observed transitions consistently indicate decreasing "
            "customer value behavior."
        ),
        "recommended_next_step": (
            "Prioritize customer value-risk intervention."
        ),
        "risk_or_caveat": (
            "The observational data does not establish causality or "
            "intervention effectiveness."
        ),
        "human_review_required": "YES",
        "grounding_status": "GROUNDED",
    }


def test_valid_llm_output_passes_governance():
    provider = FakeProvider(_valid_response())

    result = build_llm_reasoning(
        _context(),
        provider=provider,
    )

    assert result.shape == (1, 8)
    assert result.loc[0, "decision_id"] == "DEC-001"


def test_unsupported_churn_claim_is_rejected():
    response = _valid_response()
    response["reasoning"] = (
        "The customer is showing a clear trend toward churn."
    )

    provider = FakeProvider(response)

    with pytest.raises(LLMReasoningError):
        build_llm_reasoning(
            _context(),
            provider=provider,
        )


def test_causal_claim_is_rejected():
    response = _valid_response()
    response["reasoning"] = (
        "The intervention caused the customer's behavior to decrease."
    )

    provider = FakeProvider(response)

    with pytest.raises(LLMReasoningError):
        build_llm_reasoning(
            _context(),
            provider=provider,
        )


def test_recommendation_change_is_rejected():
    response = _valid_response()
    response["recommended_next_step"] = (
        "Offer a discount immediately."
    )

    provider = FakeProvider(response)

    with pytest.raises(LLMReasoningError):
        build_llm_reasoning(
            _context(),
            provider=provider,
        )