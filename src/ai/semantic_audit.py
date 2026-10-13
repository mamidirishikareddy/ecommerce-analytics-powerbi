from __future__ import annotations

import re

import pandas as pd


class SemanticAuditError(ValueError):
    """Raised when LLM reasoning contains unsupported semantic claims."""


CONTEXT_REQUIRED_COLUMNS = {
    "decision_id",
    "decision_type",
    "entity_id",
    "direction",
    "evidence_level",
    "signal_confidence",
    "customer_lifecycle",
    "value_segment",
    "active_months",
    "total_orders",
    "lifetime_revenue",
    "transition_count",
    "direction_consistency",
    "recommended_action",
}


REASONING_REQUIRED_COLUMNS = {
    "decision_id",
    "executive_summary",
    "evidence_used",
    "reasoning",
    "recommended_next_step",
    "risk_or_caveat",
    "human_review_required",
    "grounding_status",
}


AUDITED_TEXT_COLUMNS = (
    "executive_summary",
    "evidence_used",
    "reasoning",
    "recommended_next_step",
    "risk_or_caveat",
)


# These patterns identify semantic claims that cannot be supported
# by the governed observational evidence.
UNSUPPORTED_CLAIM_PATTERNS = {
    "causality": (
        "caused",
        "causes",
        "because of the intervention",
        "resulted from the intervention",
    ),
    "churn": (
        "churn",
        "churning",
    ),
    "guaranteed_outcome": (
        "will definitely",
        "will certainly",
        "guaranteed",
        "guarantee",
        "certainly improve",
    ),
    "customer_motivation": (
        "dissatisfied",
        "unhappy",
        "frustrated",
        "does not like",
        "doesn't like",
        "wants to leave",
        "intends to leave",
    ),
    "unsupported_engagement": (
        "engagement",
        "engaged",
    ),
    "unsupported_significance": (
        "statistically significant",
        "significant decline",
        "significant increase",
        "significantly declining",
        "significantly increasing",
        "significantly improved",
        "significantly worsened",
    ),
    "unsupported_confirmation": (
        "confirming",
        "confirmed",
        "confirms",
        "confirm",
    ),
    "unsupported_proof": (
        "proves",
        "prove",
        "proven",
        "proof that",
    ),
}


def _validate_frame(
    frame: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise SemanticAuditError(
            f"{frame_name} must be a pandas DataFrame."
        )

    missing = required_columns - set(frame.columns)

    if missing:
        raise SemanticAuditError(
            f"{frame_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise SemanticAuditError(
            f"{frame_name} must not be empty."
        )

    if frame[list(required_columns)].isnull().any().any():
        raise SemanticAuditError(
            f"{frame_name} contains null values in required columns."
        )

    for column in required_columns:
        if column in AUDITED_TEXT_COLUMNS:
            values = frame[column].astype("string").str.strip()

            if values.eq("").any():
                raise SemanticAuditError(
                    f"{frame_name}.{column} contains empty text values."
                )


def _validate_decision_coverage(
    context: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> None:
    context_ids = context["decision_id"].astype(str)
    reasoning_ids = reasoning["decision_id"].astype(str)

    if context_ids.duplicated().any():
        duplicates = sorted(
            context_ids[context_ids.duplicated()].unique()
        )

        raise SemanticAuditError(
            f"Context contains duplicate decision IDs: {duplicates}"
        )

    if reasoning_ids.duplicated().any():
        duplicates = sorted(
            reasoning_ids[reasoning_ids.duplicated()].unique()
        )

        raise SemanticAuditError(
            f"Reasoning contains duplicate decision IDs: {duplicates}"
        )

    context_set = set(context_ids)
    reasoning_set = set(reasoning_ids)

    missing = context_set - reasoning_set
    unexpected = reasoning_set - context_set

    if missing:
        raise SemanticAuditError(
            f"Reasoning is missing decisions: {sorted(missing)}"
        )

    if unexpected:
        raise SemanticAuditError(
            "Reasoning contains unknown decision IDs: "
            f"{sorted(unexpected)}"
        )


def _find_unsupported_claims(
    text: str,
) -> list[tuple[str, str]]:
    normalized = re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )

    findings: list[tuple[str, str]] = []

    for category, patterns in UNSUPPORTED_CLAIM_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in normalized:
                findings.append(
                    (
                        category,
                        pattern,
                    )
                )

    return findings


def _validate_claims(
    reasoning: pd.DataFrame,
) -> None:
    findings: list[tuple[str, str, str]] = []

    for _, row in reasoning.iterrows():
        decision_id = str(row["decision_id"])

        for column in AUDITED_TEXT_COLUMNS:
            text = str(row[column])

            for category, pattern in _find_unsupported_claims(text):
                findings.append(
                    (
                        decision_id,
                        column,
                        f"{category}: '{pattern}'",
                    )
                )

    if findings:
        details = "; ".join(
            (
                f"decision_id={decision_id}, "
                f"field={field}, "
                f"claim={claim}"
            )
            for decision_id, field, claim in findings
        )

        raise SemanticAuditError(
            "Unsupported semantic claims detected: "
            f"{details}"
        )


def _validate_recommendation_preservation(
    context: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> None:
    context_actions = (
        context[
            [
                "decision_id",
                "recommended_action",
            ]
        ]
        .copy()
        .set_index("decision_id")
    )

    reasoning_actions = (
        reasoning[
            [
                "decision_id",
                "recommended_next_step",
            ]
        ]
        .copy()
        .set_index("decision_id")
    )

    for decision_id in context_actions.index:
        expected = str(
            context_actions.loc[
                decision_id,
                "recommended_action",
            ]
        )

        actual = str(
            reasoning_actions.loc[
                decision_id,
                "recommended_next_step",
            ]
        )

        if actual != expected:
            raise SemanticAuditError(
                "LLM changed the governed recommendation for "
                f"decision_id={decision_id}. "
                f"Expected exact recommendation {expected!r}; "
                f"received {actual!r}."
            )


def audit_llm_reasoning(
    context: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> bool:
    """
    Perform semantic governance checks on LLM-generated reasoning.

    Checks:
    - required governed context fields are present;
    - required reasoning fields are present;
    - required fields contain no null values;
    - audited text fields are not empty;
    - every governed decision is covered exactly once;
    - no unsupported semantic claims are introduced;
    - the governed recommendation is preserved verbatim.

    Returns
    -------
    bool
        True when all semantic governance checks pass.

    Raises
    ------
    SemanticAuditError
        If any semantic governance condition fails.
    """

    _validate_frame(
        context,
        CONTEXT_REQUIRED_COLUMNS,
        "context",
    )

    _validate_frame(
        reasoning,
        REASONING_REQUIRED_COLUMNS,
        "reasoning",
    )

    _validate_decision_coverage(
        context,
        reasoning,
    )

    _validate_claims(
        reasoning,
    )

    _validate_recommendation_preservation(
        context,
        reasoning,
    )

    return True