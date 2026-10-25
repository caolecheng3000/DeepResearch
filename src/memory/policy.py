"""Deterministic memory relations; unknown paraphrases are retained conservatively.

Structured facts use metadata['fact'] = {subject, predicate, value, scope?, valid_from?}.
scope includes qualifiers such as product version, population or measurement units.
Free text rules only recognize matching assertion frames, not general entailment.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone


def timestamp(value):
    if value is None:
        return None
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        result = parsed.timestamp()
    else:
        raise ValueError("Expected a Unix timestamp or ISO-8601 date")
    if not math.isfinite(result):
        raise ValueError("Timestamp must be finite")
    return result


def canonical(value):
    if isinstance(value, str):
        return re.sub(r"\s+", "", value).casefold()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fact(entry):
    value = entry.metadata.get("fact")
    if value is None:
        return None
    if not isinstance(value, dict) or not all(k in value for k in ("subject", "predicate", "value")):
        raise ValueError("fact requires subject, predicate and value")
    if not isinstance(value["subject"], str) or not value["subject"].strip():
        raise ValueError("fact.subject must be a nonempty string")
    if not isinstance(value["predicate"], str) or not value["predicate"].strip():
        raise ValueError("fact.predicate must be a nonempty string")
    key = (canonical(value["subject"]), canonical(value["predicate"]), canonical(value.get("scope", {})))
    return key, canonical(value["value"]), timestamp(value.get("valid_from"))


_NEGATIONS = (
    (r"does\s+not\s+support|doesn't\s+support|cannot\s+support", "supports"),
    (r"do\s+not\s+support|don't\s+support", "support"),
    (r"不支持|未支持", "支持"),
    (r"不允许|未允许", "允许"),
    (r"不具备|未具备", "具备"),
    (r"不能|无法", "可以"),
)


def assertion(text):
    text = text.strip().casefold()
    negative = False
    for pattern, positive in _NEGATIONS:
        text, count = re.subn(pattern, positive, text)
        negative = negative or count > 0
    return re.sub(r"\s+", "", text), negative


def relation(left, right, similarity, conflict_threshold):
    """Return equivalent/conflict/related/distinct plus a human-readable reason."""
    a, b = fact(left), fact(right)
    if (a is None) != (b is None):
        return "related", "partially specified fact scope; retain both"
    if a is not None and b is not None:
        if a[0] != b[0]:
            return "distinct", "different fact identity or scope"
        if a[1] == b[1]:
            return "equivalent", "same structured fact and value"
        return "conflict", "same structured fact with different values"
    if canonical(left.claim) == canonical(right.claim):
        return "equivalent", "identical text"
    if similarity < conflict_threshold:
        return "distinct", "below candidate threshold"
    frame_a, neg_a = assertion(left.claim)
    frame_b, neg_b = assertion(right.claim)
    if frame_a == frame_b and neg_a != neg_b:
        return "conflict", "opposite polarity in the same assertion frame"
    # Limit numeric checks to explicit measurement/value predicates. Digits in
    # model names or entity IDs (e.g. product1 vs product2) are not value claims.
    number_claim = r"(为|是|达到|处理|成本|价格|等于|约|equals|handles|costs)([-+]?\d+(?:\.\d+)?)"
    values_a = [value for _, value in re.findall(number_claim, frame_a)]
    values_b = [value for _, value in re.findall(number_claim, frame_b)]
    if (values_a and values_b
            and re.sub(number_claim, r"\1#", frame_a) == re.sub(number_claim, r"\1#", frame_b)
            and values_a != values_b):
        return "conflict", "different numbers in the same assertion frame"
    return "related", "semantic candidate without confirmed equivalence"


def authority(entry):
    trust = float(entry.metadata.get("source_trust", 0.5))
    if not math.isfinite(trust) or not 0 <= trust <= 1:
        raise ValueError("source_trust must be between 0 and 1")
    return trust, {"primary": 3, "secondary": 2, "inference": 1}.get(entry.evidence_type, 0)


def prefer(left, right):
    """1 prefers left, -1 prefers right, 0 keeps both as disputed."""
    a, b = fact(left), fact(right)
    if a and b and a[0] == b[0] and a[2] is not None and b[2] is not None and a[2] != b[2]:
        return (1 if a[2] > b[2] else -1), "newer explicit fact valid_from"
    rank_a, rank_b = authority(left), authority(right)
    if rank_a != rank_b:
        return (1 if rank_a > rank_b else -1), "source_trust then evidence_type"
    return 0, "equal source authority; manual verification required"
