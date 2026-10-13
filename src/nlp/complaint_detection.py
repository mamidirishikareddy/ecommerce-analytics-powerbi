from __future__ import annotations

import re
import unicodedata
import pandas as pd
import unicodedata
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_PATH = Path(
    "configs/complaint_evidence.yaml"
)


OUTPUT_COLUMNS = [
    "complaint_flag",
    "complaint_confidence",
    "complaint_theme",
    "complaint_theme_confidence",
    "complaint_detection_method",
]


class ComplaintDetectionError(Exception):
    """Raised when complaint detection violates its governed contract."""


# =============================================================================
# REGISTRY
# =============================================================================


def load_complaint_registry(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load and validate the governed complaint evidence registry."""

    path = Path(registry_path)

    if not path.exists():
        raise ComplaintDetectionError(
            f"Complaint evidence registry not found: {path}"
        )

    try:
        registry = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
    except yaml.YAMLError as exc:
        raise ComplaintDetectionError(
            f"Invalid complaint evidence YAML: {exc}"
        ) from exc

    if not isinstance(registry, dict):
        raise ComplaintDetectionError(
            "Complaint evidence registry must be a mapping."
        )

    if "complaint_evidence" not in registry:
        raise ComplaintDetectionError(
            "Complaint evidence registry missing "
            "'complaint_evidence' root."
        )

    config = registry["complaint_evidence"]

    required_sections = [
        "version",
        "taxonomy",
        "negation",
        "contextual_rules",
        "confidence_method",
        "theme_selection",
        "output_fields",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise ComplaintDetectionError(
            "Complaint evidence registry missing sections: "
            f"{missing_sections}"
        )

    taxonomy = config["taxonomy"]

    if not isinstance(taxonomy, dict):
        raise ComplaintDetectionError(
            "Complaint taxonomy must be a mapping."
        )

    required_taxonomy = {
        "delivery",
        "product_quality",
        "wrong_or_missing_item",
        "seller_service",
        "order_or_transaction",
        "product_information",
        "packaging",
        "other",
        "unknown",
    }

    missing_taxonomy = (
        required_taxonomy
        - set(taxonomy.keys())
    )

    if missing_taxonomy:
        raise ComplaintDetectionError(
            "Complaint taxonomy missing categories: "
            f"{sorted(missing_taxonomy)}"
        )

    for category, definition in taxonomy.items():

        if not isinstance(definition, dict):
            raise ComplaintDetectionError(
                f"Taxonomy category '{category}' must be a mapping."
            )

        for field in [
            "strong_evidence",
            "supporting_evidence",
        ]:
            if field not in definition:
                raise ComplaintDetectionError(
                    f"Taxonomy category '{category}' "
                    f"missing '{field}'."
                )

    confidence_method = config[
        "confidence_method"
    ]

    thresholds = confidence_method.get(
        "thresholds"
    )

    if not thresholds:
        raise ComplaintDetectionError(
            "Complaint confidence thresholds are missing."
        )

    complaint_threshold = float(
        thresholds["complaint"]["minimum"]
    )

    unknown_minimum = float(
        thresholds["unknown"]["minimum"]
    )

    unknown_maximum = float(
        thresholds["unknown"]["maximum_exclusive"]
    )

    non_complaint_maximum = float(
        thresholds["non_complaint"]["maximum_exclusive"]
    )

    if not (
        0 <= non_complaint_maximum
        <= unknown_minimum
        < unknown_maximum
        <= complaint_threshold
        <= 1
    ):
        raise ComplaintDetectionError(
            "Complaint confidence thresholds are inconsistent."
        )

    theme_selection = config[
        "theme_selection"
    ]

    minimum_theme_score = float(
        theme_selection["minimum_theme_score"]
    )

    ambiguity_margin = float(
        theme_selection["ambiguity_margin"]
    )

    if not (
        0 <= minimum_theme_score <= 1
    ):
        raise ComplaintDetectionError(
            "minimum_theme_score must be between 0 and 1."
        )

    if not (
        0 <= ambiguity_margin <= 1
    ):
        raise ComplaintDetectionError(
            "ambiguity_margin must be between 0 and 1."
        )

    return config


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================


def normalize_text(
    text: Any,
) -> str:
    """
    Normalize review text without translating or rewriting its meaning.

    Operations:
    - NULL/non-string -> empty string
    - Unicode normalization
    - lowercase
    - whitespace normalization
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_phrase(
    phrase: str,
) -> str:
    """Normalize a registry phrase using the same text normalization."""

    return normalize_text(
        phrase
    )


# =============================================================================
# EVIDENCE MATCHING
# =============================================================================


def phrase_present(
    text: str,
    phrase: str,
) -> bool:
    """
    Check whether a normalized evidence phrase occurs in normalized text.

    Word-boundary matching is used where possible while preserving phrases
    containing punctuation or accented characters.
    """

    phrase = normalize_phrase(
        phrase
    )

    if not phrase:
        return False

    escaped = re.escape(
        phrase
    )

    pattern = (
        rf"(?<!\w)"
        rf"{escaped}"
        rf"(?!\w)"
    )

    return re.search(
        pattern,
        text,
        flags=re.UNICODE,
    ) is not None


def collect_matches(
    text: str,
    phrases: list[str],
) -> list[str]:
    """Return all registry phrases found in text."""

    matches: list[str] = []

    for phrase in phrases:

        if phrase_present(
            text,
            phrase,
        ):
            matches.append(
                phrase
            )

    return matches


# =============================================================================
# NEGATION
# =============================================================================


def has_negation_evidence(
    text: str,
    registry: dict[str, Any],
) -> list[str]:
    """Return explicit negation patterns found in the review text."""

    patterns = registry[
        "negation"
    ].get(
        "patterns",
        [],
    )

    return collect_matches(
        text,
        patterns,
    )


# =============================================================================
# THEME EVIDENCE
# =============================================================================


def calculate_theme_evidence(
    text: str,
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Calculate deterministic evidence for each complaint theme.

    Strong evidence takes precedence over supporting evidence.

    The returned score is an evidence score, not a calibrated probability.
    """

    taxonomy = registry[
        "taxonomy"
    ]

    evidence: dict[str, dict[str, Any]] = {}

    for theme, definition in taxonomy.items():

        strong_matches = collect_matches(
            text,
            definition.get(
                "strong_evidence",
                [],
            ),
        )

        supporting_matches = collect_matches(
            text,
            definition.get(
                "supporting_evidence",
                [],
            ),
        )

        strong_count = len(
            strong_matches
        )

        supporting_count = len(
            supporting_matches
        )

        if strong_count:
            score = 1.0

            evidence_strength = (
                "explicit_problem"
            )

        elif supporting_count:
            score = min(
                0.30
                + (
                    0.10
                    * supporting_count
                ),
                0.60,
            )

            evidence_strength = (
                "supporting_evidence"
            )

        else:
            score = 0.0

            evidence_strength = (
                "none"
            )

        evidence[theme] = {
            "score": score,
            "strong_matches": strong_matches,
            "supporting_matches": supporting_matches,
            "evidence_strength": evidence_strength,
        }

    return evidence


# =============================================================================
# CLASSIFICATION
# =============================================================================


def classify_complaint(
    review_text: Any,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """
    Classify one review into complaint/non-complaint/unknown.

    Important:
    - sentiment is not an input to this function
    - review_score is not an input
    - text evidence drives the classification
    """

    text = normalize_text(
        review_text
    )

    # -------------------------------------------------------------------------
    # Textless population
    # -------------------------------------------------------------------------

    if not text:

        return {
            "complaint_flag": None,
            "complaint_confidence": None,
            "complaint_theme": None,
            "complaint_theme_confidence": None,
            "complaint_detection_method": "unclassified",
        }

    # -------------------------------------------------------------------------
    # Negation evidence
    # -------------------------------------------------------------------------

    negation_matches = has_negation_evidence(
        text,
        registry,
    )

    # -------------------------------------------------------------------------
    # Theme evidence
    # -------------------------------------------------------------------------

    evidence = calculate_theme_evidence(
        text,
        registry,
    )

    # Exclude "unknown" from direct evidence competition.
    candidate_themes = {
        theme: details
        for theme, details in evidence.items()
        if theme != "unknown"
    }

    # -------------------------------------------------------------------------
    # Negation protection
    #
    # If the review explicitly says that there was no problem, do not allow
    # generic supporting evidence to create a complaint.
    #
    # Strong explicit problem phrases remain eligible because the registry
    # treats them as direct evidence.
    # -------------------------------------------------------------------------

    if negation_matches:

     for theme, details in candidate_themes.items():

            # Explicit negation suppresses supporting/weak evidence.
            # Strong explicit problem evidence is retained because a review
            # may contain both a negated statement and a real complaint.
            if (
                details["evidence_strength"]
                == "supporting_evidence"
            ):
                details["score"] = 0.0

                details["supporting_matches"] = []

                details["evidence_strength"] = (
                    "negated"
                )

    # -------------------------------------------------------------------------
    # Rank themes by evidence score
    # -------------------------------------------------------------------------

    ranked = sorted(
        candidate_themes.items(),
        key=lambda item: (
            item[1]["score"],
            len(item[1]["strong_matches"]),
            len(item[1]["supporting_matches"]),
        ),
        reverse=True,
    )

    best_theme = None
    best_score = 0.0
    second_score = 0.0

    if ranked:

        best_theme, best_details = ranked[0]

        best_score = float(
            best_details["score"]
        )

        if len(ranked) > 1:
            second_score = float(
                ranked[1][1]["score"]
            )

    # -------------------------------------------------------------------------
    # Confidence thresholds
    # -------------------------------------------------------------------------

    thresholds = registry[
        "confidence_method"
    ][
        "thresholds"
    ]

    complaint_minimum = float(
        thresholds["complaint"]["minimum"]
    )

    unknown_minimum = float(
        thresholds["unknown"]["minimum"]
    )

    # -------------------------------------------------------------------------
    # No meaningful evidence
    # -------------------------------------------------------------------------

    if best_score < unknown_minimum:

        return {
            "complaint_flag": False,
            "complaint_confidence": round(
                best_score,
                6,
            ),
            "complaint_theme": None,
            "complaint_theme_confidence": None,
            "complaint_detection_method": "rule_based",
        }

    # -------------------------------------------------------------------------
    # Ambiguous evidence
    # -------------------------------------------------------------------------

    ambiguity_margin = float(
        registry[
            "theme_selection"
        ][
            "ambiguity_margin"
        ]
    )

    if (
        best_score >= complaint_minimum
        and (
            best_score
            - second_score
            < ambiguity_margin
        )
        and second_score > 0
    ):

        return {
            "complaint_flag": None,
            "complaint_confidence": round(
                best_score,
                6,
            ),
            "complaint_theme": "unknown",
            "complaint_theme_confidence": round(
                best_score,
                6,
            ),
            "complaint_detection_method": "rule_based",
        }

    # -------------------------------------------------------------------------
    # Confirmed complaint
    # -------------------------------------------------------------------------

    if best_score >= complaint_minimum:

        return {
            "complaint_flag": True,
            "complaint_confidence": round(
                best_score,
                6,
            ),
            "complaint_theme": best_theme,
            "complaint_theme_confidence": round(
                best_score,
                6,
            ),
            "complaint_detection_method": "rule_based",
        }

    # -------------------------------------------------------------------------
    # Uncertain / unknown
    # -------------------------------------------------------------------------

    return {
        "complaint_flag": None,
        "complaint_confidence": round(
            best_score,
            6,
        ),
        "complaint_theme": "unknown",
        "complaint_theme_confidence": round(
            best_score,
            6,
        ),
        "complaint_detection_method": "rule_based",
    }


# =============================================================================
# DATAFRAME TRANSFORMATION
# =============================================================================


def detect_complaints(
    df,
    registry: dict[str, Any],
):
    """
    Apply complaint detection to a Customer Voice dataframe.

    Required input:
        review_id
        order_id
        review_text
        has_review_text

    Existing columns are preserved.
    Five governed complaint fields are appended.
    """

    required_columns = [
        "review_id",
        "order_id",
        "review_text",
        "has_review_text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ComplaintDetectionError(
            "Complaint detection input missing columns: "
            f"{missing_columns}"
        )

    result = df.copy()

    duplicate_count = int(
        result.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ComplaintDetectionError(
            "Complaint detection input contains "
            f"{duplicate_count:,} duplicate review keys."
        )

    records = [
        classify_complaint(
            review_text=value,
            registry=registry,
        )
        for value in result[
            "review_text"
        ]
    ]

    complaint_df = pd.DataFrame(
        records,
        index=result.index,
    )

    for column in OUTPUT_COLUMNS:

        result[column] = complaint_df[
            column
        ]

    # -------------------------------------------------------------------------
    # Population validation
    # -------------------------------------------------------------------------

    textless_mask = (
        ~result["has_review_text"]
    )

    text_bearing_mask = (
        result["has_review_text"]
    )

    textless_complaint_values = (
        result.loc[
            textless_mask,
            "complaint_flag",
        ]
        .notna()
        .sum()
    )

    if int(
        textless_complaint_values
    ):
        raise ComplaintDetectionError(
            "Textless reviews received complaint_flag."
        )

    textless_theme_values = (
        result.loc[
            textless_mask,
            "complaint_theme",
        ]
        .notna()
        .sum()
    )

    if int(
        textless_theme_values
    ):
        raise ComplaintDetectionError(
            "Textless reviews received complaint_theme."
        )

    # -------------------------------------------------------------------------
    # Text-bearing reviews must have an explicit classification state.
    #
    # True = complaint
    # False = non-complaint
    # NULL = genuine uncertainty.
    # -------------------------------------------------------------------------

    invalid_text_classification = (
        result.loc[
            text_bearing_mask,
            "complaint_flag",
        ]
        .isna()
        .sum()
    )

    # NULL is permitted only for genuinely ambiguous classifications.
    # Therefore we do not reject it here.
    #
    # We do require that ambiguous rows explicitly carry the unknown theme.
    ambiguous_mask = (
        text_bearing_mask
        & result["complaint_flag"].isna()
    )

    ambiguous_without_unknown_theme = (
        ambiguous_mask
        & result["complaint_theme"].ne(
            "unknown"
        )
    ).sum()

    if int(
        ambiguous_without_unknown_theme
    ):
        raise ComplaintDetectionError(
            "Ambiguous complaint classifications must use "
            "complaint_theme='unknown'."
        )

    # -------------------------------------------------------------------------
    # Confidence validation
    # -------------------------------------------------------------------------

    confidence = result[
        "complaint_confidence"
    ]

    invalid_confidence = (
        confidence.notna()
        & (
            (confidence < 0)
            | (confidence > 1)
        )
    )

    if int(
        invalid_confidence.sum()
    ):
        raise ComplaintDetectionError(
            "Complaint confidence contains values outside [0, 1]."
        )

    theme_confidence = result[
        "complaint_theme_confidence"
    ]

    invalid_theme_confidence = (
        theme_confidence.notna()
        & (
            (theme_confidence < 0)
            | (theme_confidence > 1)
        )
    )

    if int(
        invalid_theme_confidence.sum()
    ):
        raise ComplaintDetectionError(
            "Complaint theme confidence contains "
            "values outside [0, 1]."
        )

    # -------------------------------------------------------------------------
    # Allowed theme validation
    # -------------------------------------------------------------------------

    allowed_themes = set(
        registry[
            "taxonomy"
        ].keys()
    )

    observed_themes = set(
        result.loc[
            result["complaint_theme"].notna(),
            "complaint_theme",
        ].astype(str)
    )

    unexpected_themes = (
        observed_themes
        - allowed_themes
    )

    if unexpected_themes:
        raise ComplaintDetectionError(
            "Unexpected complaint themes: "
            f"{sorted(unexpected_themes)}"
        )

    return result


# =============================================================================
# SIMPLE SINGLE-REVIEW API
# =============================================================================


def classify_review(
    review_text: str | None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Convenience API for classifying a single review."""

    registry = load_complaint_registry(
        registry_path
    )

    return classify_complaint(
        review_text=review_text,
        registry=registry,
    )