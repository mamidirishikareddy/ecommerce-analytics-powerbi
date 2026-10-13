import pandas as pd
import pytest

from src.ai.decision_reasoning import (
    AIDecisionReasoningError,
    AIDecisionReasoningRecord,
    AI_REASONING_COLUMNS,
    build_ai_reasoning_frame,
    validate_ai_reasoning_record,
)


def _record(
    decision_id: str = "DEC-001",
) -> AIDecisionReasoningRecord:
    return AIDecisionReasoningRecord(
        decision_id=decision_id,
        executive_summary=(
            "High-value customer shows a consistent decreasing pattern."
        ),
        evidence_used=(
            "HIGH evidence with three behavioral transitions and "
            "direction consistency of 1.0."
        ),
        reasoning=(
            "The customer has repeated purchases and all observed "
            "behavioral transitions indicate declining value."
        ),
        recommended_next_step=(
            "Prioritize customer value-risk intervention."
        ),
        risk_or_caveat=(
            "Observed behavior is based only on the available dataset "
            "window and does not establish causality."
        ),
        human_review_required="YES",
        grounding_status="GROUNDED",
    )


def test_valid_record_passes_validation():
    validate_ai_reasoning_record(_record())


def test_record_has_expected_fields():
    record = _record()

    assert list(record.__dataclass_fields__.keys()) == AI_REASONING_COLUMNS


def test_build_reasoning_frame_shape():
    result = build_ai_reasoning_frame(
        [_record("DEC-001"), _record("DEC-002")]
    )

    assert result.shape == (2, 8)


def test_build_reasoning_frame_columns():
    result = build_ai_reasoning_frame([_record()])

    assert list(result.columns) == AI_REASONING_COLUMNS


def test_preserves_decision_ids():
    result = build_ai_reasoning_frame(
        [_record("DEC-001"), _record("DEC-002")]
    )

    assert result["decision_id"].tolist() == [
        "DEC-001",
        "DEC-002",
    ]


def test_preserves_reasoning_content():
    result = build_ai_reasoning_frame([_record()])

    row = result.iloc[0]

    assert "decreasing pattern" in row["executive_summary"]
    assert "three behavioral transitions" in row["evidence_used"]
    assert "causality" in row["risk_or_caveat"]


def test_grounding_status_is_restricted():
    record = _record()

    invalid = AIDecisionReasoningRecord(
        **{
            **record.__dict__,
            "grounding_status": "UNCERTAIN",
        }
    )

    with pytest.raises(AIDecisionReasoningError):
        validate_ai_reasoning_record(invalid)


def test_human_review_status_is_restricted():
    record = _record()

    invalid = AIDecisionReasoningRecord(
        **{
            **record.__dict__,
            "human_review_required": "MAYBE",
        }
    )

    with pytest.raises(AIDecisionReasoningError):
        validate_ai_reasoning_record(invalid)


def test_empty_string_is_rejected():
    record = _record()

    invalid = AIDecisionReasoningRecord(
        **{
            **record.__dict__,
            "reasoning": "",
        }
    )

    with pytest.raises(AIDecisionReasoningError):
        validate_ai_reasoning_record(invalid)


def test_null_value_is_rejected():
    record = _record()

    invalid = AIDecisionReasoningRecord(
        **{
            **record.__dict__,
            "risk_or_caveat": None,
        }
    )

    with pytest.raises(AIDecisionReasoningError):
        validate_ai_reasoning_record(invalid)


def test_non_record_is_rejected():
    with pytest.raises(AIDecisionReasoningError):
        validate_ai_reasoning_record({"decision_id": "DEC-001"})


def test_empty_record_list_is_rejected():
    with pytest.raises(AIDecisionReasoningError):
        build_ai_reasoning_frame([])


def test_duplicate_decision_ids_are_rejected():
    with pytest.raises(AIDecisionReasoningError):
        build_ai_reasoning_frame(
            [_record("DEC-001"), _record("DEC-001")]
        )


def test_dataframe_contains_no_nulls():
    result = build_ai_reasoning_frame([_record()])

    assert isinstance(result, pd.DataFrame)
    assert not result.isna().any().any()