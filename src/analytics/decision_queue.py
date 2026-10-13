from __future__ import annotations

import pandas as pd


class DecisionQueueError(ValueError):
    """Raised when a governed decision queue cannot be built."""


REQUIRED_COLUMNS = {
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
}


QUEUE_READINESS = {
    "ACTION_CANDIDATE",
    "REVIEW",
}

PRIORITY_ORDER = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
}


def build_decision_queue(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an operational queue from governed decision candidates.

    Only ACTION_CANDIDATE and REVIEW decisions enter the queue.
    NOT_READY and OBSERVE decisions remain outside the operational queue.

    The queue does not create new evidence or scoring. It only prioritizes
    already-governed decisions using priority and decision readiness.
    """
    missing_columns = REQUIRED_COLUMNS - set(decisions.columns)

    if missing_columns:
        raise DecisionQueueError(
            "Decisions are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if decisions.empty:
        raise DecisionQueueError(
            "Decisions cannot be empty."
        )

    if decisions["decision_id"].duplicated().any():
        raise DecisionQueueError(
            "Decisions must contain unique decision IDs."
        )

    queue = decisions[
        decisions["decision_readiness"].isin(QUEUE_READINESS)
    ].copy()

    if queue.empty:
        return pd.DataFrame(
            columns=[
                *decisions.columns,
                "queue_rank",
            ]
        )

    invalid_priorities = set(queue["priority"]) - set(PRIORITY_ORDER)

    if invalid_priorities:
        raise DecisionQueueError(
            "Queue contains unsupported priorities: "
            f"{sorted(invalid_priorities)}"
        )

    queue["_priority_order"] = queue["priority"].map(PRIORITY_ORDER)

    queue["_readiness_order"] = queue[
        "decision_readiness"
    ].map(
        {
            "ACTION_CANDIDATE": 1,
            "REVIEW": 2,
        }
    )

    queue = queue.sort_values(
        by=[
            "_priority_order",
            "_readiness_order",
            "decision_id",
        ],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    queue["queue_rank"] = range(1, len(queue) + 1)

    queue = queue.drop(
        columns=[
            "_priority_order",
            "_readiness_order",
        ]
    )

    return queue