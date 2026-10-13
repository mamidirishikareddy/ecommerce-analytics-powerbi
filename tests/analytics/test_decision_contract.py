import pytest

from src.analytics.decision_contract import (
    DECISION_CONTRACT_FIELDS,
    DecisionRecord,
    validate_decision_record,
)


def _valid_record():
    return DecisionRecord(
        decision_id="DEC-001",
        decision_type="CUSTOMER_RETENTION",
        entity_id="customer-001",
        signal="Repeat customer showing decreasing revenue",
        direction="DECREASING",
        priority="HIGH",
        evidence_level="MEDIUM",
        decision_readiness="REVIEW",
        evidence_basis="Two behavioral transitions observed.",
        recommended_action="Review retention intervention.",
    )


def test_valid_decision_record_passes():
    record = _valid_record()

    validate_decision_record(record)


def test_contract_contains_expected_fields():
    assert DECISION_CONTRACT_FIELDS == [
        "decision_id",
        "decision_type",
        "entity_id",
        "signal",
        "direction",
        "priority",
        "evidence_level",
        "decision_readiness",
        "evidence_basis",
        "recommended_action",
    ]


@pytest.mark.parametrize(
    "priority",
    [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ],
)
def test_valid_priorities_pass(priority):
    record = DecisionRecord(
        **{
            **_valid_record().__dict__,
            "priority": priority,
        }
    )

    validate_decision_record(record)


@pytest.mark.parametrize(
    "readiness",
    [
        "ACTION_CANDIDATE",
        "REVIEW",
        "OBSERVE",
        "NOT_READY",
    ],
)
def test_valid_readiness_values_pass(readiness):
    record = DecisionRecord(
        **{
            **_valid_record().__dict__,
            "decision_readiness": readiness,
        }
    )

    validate_decision_record(record)


@pytest.mark.parametrize(
    "priority",
    [
        "URGENT",
        "P1",
        "",
        "UNKNOWN",
    ],
)
def test_invalid_priority_raises(priority):
    record = DecisionRecord(
        **{
            **_valid_record().__dict__,
            "priority": priority,
        }
    )

    with pytest.raises(ValueError):
        validate_decision_record(record)


@pytest.mark.parametrize(
    "readiness",
    [
        "ACTION",
        "APPROVED",
        "",
        "UNKNOWN",
    ],
)
def test_invalid_readiness_raises(readiness):
    record = DecisionRecord(
        **{
            **_valid_record().__dict__,
            "decision_readiness": readiness,
        }
    )

    with pytest.raises(ValueError):
        validate_decision_record(record)


@pytest.mark.parametrize(
    "field_name",
    DECISION_CONTRACT_FIELDS,
)
def test_empty_string_required_field_raises(field_name):
    values = _valid_record().__dict__
    values[field_name] = ""

    record = DecisionRecord(**values)

    with pytest.raises(ValueError):
        validate_decision_record(record)


@pytest.mark.parametrize(
    "field_name",
    DECISION_CONTRACT_FIELDS,
)
def test_none_required_field_raises(field_name):
    values = _valid_record().__dict__
    values[field_name] = None

    record = DecisionRecord(**values)

    with pytest.raises(ValueError):
        validate_decision_record(record)