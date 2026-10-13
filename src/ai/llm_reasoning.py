
from __future__ import annotations

import json

import pandas as pd

from src.ai.decision_input import (
    AIDecisionInputError,
    build_ai_decision_input,
)
from src.ai.decision_reasoning import (
    AIDecisionReasoningError,
    AIDecisionReasoningRecord,
    build_ai_reasoning_frame,
    validate_ai_reasoning_record,
)
from src.ai.groq_provider import (
    GroqProviderError,
    GroqReasoningProvider,
)
from src.ai.grounding_validator import (
    validate_llm_grounding,
)
from src.ai.semantic_audit import (
    SemanticAuditError,
    audit_llm_reasoning,
)


class LLMReasoningError(RuntimeError):
    """Raised when governed LLM reasoning fails."""


REQUIRED_COLUMNS = {
    "decision_id",
    "decision_type",
    "entity_id",
    "signal",
    "direction",
    "priority",
    "queue_rank",
    "decision_readiness",
    "evidence_level",
    "signal_confidence",
    "customer_lifecycle",
    "value_segment",
    "active_months",
    "total_orders",
    "lifetime_revenue",
    "transition_count",
    "direction_consistency",
    "evidence_basis",
    "recommended_action",
}

LLM_OUTPUT_FIELDS = [
    "decision_id",
    "executive_summary",
    "evidence_used",
    "reasoning",
    "recommended_next_step",
    "risk_or_caveat",
    "human_review_required",
    "grounding_status",
]


def _expected_human_review(row: pd.Series) -> str:
    """Return the governed human-review value for a decision."""
    return (
        "YES"
        if row["decision_readiness"] in (
            "REVIEW",
            "ACTION_CANDIDATE",
        )
        else "NO"
    )


def _validate_input(ai_input: pd.DataFrame) -> None:
    if not isinstance(ai_input, pd.DataFrame):
        raise LLMReasoningError(
            "ai_input must be a pandas DataFrame."
        )

    missing = REQUIRED_COLUMNS - set(ai_input.columns)
    if missing:
        raise LLMReasoningError(
            f"AI input is missing required columns: {sorted(missing)}"
        )

    if ai_input.empty:
        raise LLMReasoningError("AI input must not be empty.")

    if ai_input[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise LLMReasoningError(
            "AI input contains null values in required columns."
        )

    if ai_input["decision_id"].duplicated().any():
        raise LLMReasoningError(
            "AI input contains duplicate decision_id values."
        )


def _build_prompt(row: pd.Series) -> str:
    """Build the primary governed reasoning prompt."""
    return f"""
You are a constrained decision-reasoning assistant.

Explain ONLY the supplied governed decision.

GROUNDING
---------
Use only the supplied evidence.

Do not invent facts, metrics, causes, motivations, outcomes,
statistical significance, sentiment, intent, or intervention effects.
Do not make causal claims.

Do not infer or mention:
- churn or likelihood to leave
- dissatisfaction or satisfaction
- customer intent or motivation
- engagement
- statistical significance

Do not claim evidence confirms, proves, guarantees, or definitely
establishes an outcome.

Describe only observed behavioral patterns and the available
evidence strength.

GOVERNED VALUES
---------------
Do not change any supplied governed value.

RECOMMENDATION
--------------
recommended_next_step MUST be an exact character-for-character
copy of recommended_action. Do not paraphrase it.

HUMAN REVIEW
------------
human_review_required MUST be exactly:
{_expected_human_review(row)}

Do not derive, reinterpret, or change this value.

GROUNDING STATUS
----------------
grounding_status MUST be exactly one of:
GROUNDED
INSUFFICIENT_EVIDENCE

Do not use any other value.

RISK / CAVEAT
-------------
State that observational data does not establish causality or
intervention effectiveness.

OUTPUT
------
Return ONLY valid JSON with exactly these eight keys:
{json.dumps(LLM_OUTPUT_FIELDS)}

GOVERNED DECISION
-----------------
decision_id: {row["decision_id"]}
decision_type: {row["decision_type"]}
entity_id: {row["entity_id"]}
signal: {row["signal"]}
direction: {row["direction"]}
priority: {row["priority"]}
queue_rank: {row["queue_rank"]}
decision_readiness: {row["decision_readiness"]}
evidence_level: {row["evidence_level"]}
signal_confidence: {row["signal_confidence"]}
customer_lifecycle: {row["customer_lifecycle"]}
value_segment: {row["value_segment"]}
active_months: {row["active_months"]}
total_orders: {row["total_orders"]}
lifetime_revenue: {row["lifetime_revenue"]}
transition_count: {row["transition_count"]}
direction_consistency: {row["direction_consistency"]}
evidence_basis: {row["evidence_basis"]}
recommended_action: {row["recommended_action"]}
""".strip()


def _build_recommendation_retry_prompt(row: pd.Series) -> str:
    """Build a targeted recommendation-correction prompt."""
    return f"""
CORRECTION REQUIRED.

Return the exact governed recommendation below.
Do not paraphrase, shorten, expand, or change punctuation.

recommended_action:
{row["recommended_action"]}

recommended_next_step MUST equal recommended_action exactly.

HUMAN REVIEW
------------
human_review_required MUST be exactly:
{_expected_human_review(row)}

Do not derive, reinterpret, or change this value.

GROUNDING STATUS
----------------
grounding_status MUST be exactly one of:
GROUNDED
INSUFFICIENT_EVIDENCE

Do not use any other value.

Do not introduce unsupported claims about churn, satisfaction,
motivation, intent, engagement, significance, causality, or outcomes.

Return ONLY valid JSON with exactly these eight keys:
{json.dumps(LLM_OUTPUT_FIELDS)}

decision_id: {row["decision_id"]}
decision_type: {row["decision_type"]}
entity_id: {row["entity_id"]}
direction: {row["direction"]}
priority: {row["priority"]}
decision_readiness: {row["decision_readiness"]}
evidence_level: {row["evidence_level"]}
signal_confidence: {row["signal_confidence"]}
customer_lifecycle: {row["customer_lifecycle"]}
value_segment: {row["value_segment"]}
active_months: {row["active_months"]}
total_orders: {row["total_orders"]}
lifetime_revenue: {row["lifetime_revenue"]}
transition_count: {row["transition_count"]}
direction_consistency: {row["direction_consistency"]}
evidence_basis: {row["evidence_basis"]}
""".strip()


def _build_semantic_retry_prompt(row: pd.Series) -> str:
    """Build a targeted semantic and contract-correction prompt."""
    return f"""
CORRECTION REQUIRED.

Rewrite the explanation to remove unsupported semantic claims.
Use only the supplied evidence.

Do not infer or mention:
- churn or likelihood to leave
- dissatisfaction or satisfaction
- customer intent or motivation
- engagement
- statistical significance

Do not claim that evidence confirms, proves, guarantees, or
definitely establishes an outcome.
Do not make causal claims.

Describe only observed behavioral patterns and evidence strength.

SAFE WORDING EXAMPLES
---------------------
Prefer factual wording such as:
- "The observed transitions show a decreasing behavioral direction."
- "The evidence is limited to the observed transitions."
- "Additional behavioral history may help assess whether the pattern persists."
- "The observations do not establish causality or intervention effectiveness."

Do not use the terms "engagement", "engaged", "churn", "confirm",
"confirmed", "confirms", "confirming", "prove", "proves", "proven",
"proof", "guarantee", or "guaranteed" anywhere in the generated text.

Do not introduce new facts or infer customer intent, satisfaction,
future behavior, or intervention outcomes.

RECOMMENDATION
--------------
recommended_next_step MUST equal this value exactly:
{row["recommended_action"]}

HUMAN REVIEW
------------
human_review_required MUST be exactly:
{_expected_human_review(row)}

Do not derive, reinterpret, or change this value.

GROUNDING STATUS
----------------
grounding_status MUST be exactly one of:
GROUNDED
INSUFFICIENT_EVIDENCE

Do not use any other value, including EVIDENCE_BASED.

Return ONLY valid JSON with exactly these eight keys:
{json.dumps(LLM_OUTPUT_FIELDS)}

decision_id: {row["decision_id"]}
decision_type: {row["decision_type"]}
entity_id: {row["entity_id"]}
direction: {row["direction"]}
priority: {row["priority"]}
decision_readiness: {row["decision_readiness"]}
evidence_level: {row["evidence_level"]}
signal_confidence: {row["signal_confidence"]}
customer_lifecycle: {row["customer_lifecycle"]}
value_segment: {row["value_segment"]}
active_months: {row["active_months"]}
total_orders: {row["total_orders"]}
lifetime_revenue: {row["lifetime_revenue"]}
transition_count: {row["transition_count"]}
direction_consistency: {row["direction_consistency"]}
evidence_basis: {row["evidence_basis"]}
""".strip()


def _parse_response(
    response: str,
    expected_decision_id: str,
) -> AIDecisionReasoningRecord:
    """Parse a JSON response and enforce the exact output schema."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise LLMReasoningError(
            "LLM response was not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise LLMReasoningError(
            "LLM response must be a JSON object."
        )

    missing = set(LLM_OUTPUT_FIELDS) - set(payload)
    if missing:
        raise LLMReasoningError(
            f"LLM response is missing required fields: {sorted(missing)}"
        )

    unexpected = set(payload) - set(LLM_OUTPUT_FIELDS)
    if unexpected:
        raise LLMReasoningError(
            f"LLM response contains unexpected fields: {sorted(unexpected)}"
        )

    if str(payload["decision_id"]) != str(expected_decision_id):
        raise LLMReasoningError(
            "LLM response changed the governed decision_id."
        )

    try:
        return AIDecisionReasoningRecord(
            decision_id=str(payload["decision_id"]),
            executive_summary=str(payload["executive_summary"]),
            evidence_used=str(payload["evidence_used"]),
            reasoning=str(payload["reasoning"]),
            recommended_next_step=str(
                payload["recommended_next_step"]
            ),
            risk_or_caveat=str(payload["risk_or_caveat"]),
            human_review_required=str(
                payload["human_review_required"]
            ),
            grounding_status=str(payload["grounding_status"]),
        )
    except (TypeError, ValueError) as exc:
        raise LLMReasoningError(
            f"Invalid LLM reasoning record: {exc}"
        ) from exc


def _validate_record_contract(
    record: AIDecisionReasoningRecord,
) -> None:
    """Validate allowed values and non-empty reasoning fields."""
    try:
        validate_ai_reasoning_record(record)
    except AIDecisionReasoningError as exc:
        raise LLMReasoningError(
            f"Invalid LLM reasoning record: {exc}"
        ) from exc


def _recommendation_matches(
    record: AIDecisionReasoningRecord,
    row: pd.Series,
) -> bool:
    return (
        record.recommended_next_step
        == str(row["recommended_action"])
    )


def _build_reasoning_frame(
    records: list[AIDecisionReasoningRecord],
) -> pd.DataFrame:
    try:
        return build_ai_reasoning_frame(records)
    except AIDecisionReasoningError as exc:
        raise LLMReasoningError(
            "Failed to construct governed LLM reasoning output: "
            f"{exc}"
        ) from exc


def _validate_governance(
    governed_input: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> None:
    try:
        validate_llm_grounding(
            context=governed_input,
            reasoning=reasoning,
        )
    except Exception as exc:
        raise LLMReasoningError(
            f"LLM grounding validation failed: {exc}"
        ) from exc

    try:
        audit_llm_reasoning(
            context=governed_input,
            reasoning=reasoning,
        )
    except SemanticAuditError as exc:
        raise LLMReasoningError(
            f"LLM semantic audit failed: {exc}"
        ) from exc


def _find_semantically_invalid_decisions(
    governed_input: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> set[str]:
    """Identify only decisions that fail semantic auditing."""
    invalid_ids: set[str] = set()

    for _, context_row in governed_input.iterrows():
        decision_id = str(context_row["decision_id"])

        context_one = governed_input[
            governed_input["decision_id"].astype(str) == decision_id
        ].copy()

        reasoning_one = reasoning[
            reasoning["decision_id"].astype(str) == decision_id
        ].copy()

        try:
            audit_llm_reasoning(
                context=context_one,
                reasoning=reasoning_one,
            )
        except SemanticAuditError:
            invalid_ids.add(decision_id)

    return invalid_ids


def _generate_record(
    row: pd.Series,
    provider: GroqReasoningProvider,
) -> AIDecisionReasoningRecord:
    """Generate a record with one contract-correction attempt."""
    try:
        raw_response = provider.generate_json(_build_prompt(row))
    except GroqProviderError as exc:
        raise LLMReasoningError(
            f"Groq reasoning request failed: {exc}"
        ) from exc

    record = _parse_response(
        raw_response,
        expected_decision_id=str(row["decision_id"]),
    )

    try:
        _validate_record_contract(record)
    except LLMReasoningError:
        try:
            corrected_response = provider.generate_json(
                _build_semantic_retry_prompt(row)
            )
        except GroqProviderError as exc:
            raise LLMReasoningError(
                f"Groq contract-correction request failed: {exc}"
            ) from exc

        record = _parse_response(
            corrected_response,
            expected_decision_id=str(row["decision_id"]),
        )

        _validate_record_contract(record)

    if not _recommendation_matches(record, row):
        try:
            retry_response = provider.generate_json(
                _build_recommendation_retry_prompt(row)
            )
        except GroqProviderError as exc:
            raise LLMReasoningError(
                f"Groq recommendation-correction request failed: {exc}"
            ) from exc

        record = _parse_response(
            retry_response,
            expected_decision_id=str(row["decision_id"]),
        )
        _validate_record_contract(record)

        if not _recommendation_matches(record, row):
            raise LLMReasoningError(
                "LLM changed the governed recommended action for "
                f"decision_id={row['decision_id']} after corrective retry."
            )

    if record.human_review_required != _expected_human_review(row):
        raise LLMReasoningError(
            "LLM changed the governed human-review requirement for "
            f"decision_id={row['decision_id']}."
        )

    return record


def build_llm_reasoning(
    ai_input: pd.DataFrame,
    provider: GroqReasoningProvider | None = None,
) -> pd.DataFrame:
    """
    Generate and govern LLM reasoning.

    Invalid enum values receive one contract-correction attempt.
    Recommendation mismatches receive one recommendation retry.
    Semantic audit failures receive one targeted semantic retry.
    All unresolved failures are rejected.
    """
    _validate_input(ai_input)

    try:
        governed_input = build_ai_decision_input(ai_input)
    except AIDecisionInputError as exc:
        raise LLMReasoningError(
            f"Invalid governed AI input: {exc}"
        ) from exc

    provider = provider or GroqReasoningProvider()

    records = [
        _generate_record(row, provider)
        for _, row in governed_input.iterrows()
    ]

    reasoning = _build_reasoning_frame(records)

    try:
        _validate_governance(governed_input, reasoning)
        return reasoning
    except LLMReasoningError as first_error:
        if "LLM semantic audit failed:" not in str(first_error):
            raise

    invalid_ids = _find_semantically_invalid_decisions(
        governed_input,
        reasoning,
    )

    if not invalid_ids:
        raise LLMReasoningError(
            "Semantic audit failed but no individual invalid decision "
            "could be identified."
        )

    corrected_records = list(records)

    for index, (_, row) in enumerate(governed_input.iterrows()):
        decision_id = str(row["decision_id"])

        if decision_id not in invalid_ids:
            continue

        try:
            retry_response = provider.generate_json(
                _build_semantic_retry_prompt(row)
            )
        except GroqProviderError as exc:
            raise LLMReasoningError(
                f"Groq semantic-correction request failed: {exc}"
            ) from exc

        corrected_record = _parse_response(
            retry_response,
            expected_decision_id=decision_id,
        )
        _validate_record_contract(corrected_record)

        if not _recommendation_matches(corrected_record, row):
            raise LLMReasoningError(
                "LLM changed the governed recommendation during "
                f"semantic correction for decision_id={decision_id}."
            )

        if (
            corrected_record.human_review_required
            != _expected_human_review(row)
        ):
            raise LLMReasoningError(
                "LLM changed the governed human-review requirement "
                f"during semantic correction for decision_id={decision_id}."
            )

        corrected_records[index] = corrected_record

    corrected_reasoning = _build_reasoning_frame(corrected_records)
    _validate_governance(governed_input, corrected_reasoning)

    return corrected_reasoning
