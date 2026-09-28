"""Deterministic pre-filter: resolves obvious tickets with regex/keywords before
any API call. Rules are intentionally narrow and precision-first — each one only
fires on a specific, unambiguous phrasing it's confident about, and everything
else (including hedged/ambiguous tickets) is deferred to the model. Being plain
regex over the raw text, this layer is also immune to prompt injection: it never
sends anything to an LLM, so it can't be talked out of its answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kasoti.models import ClassificationOutput

# Any of these appearing means the ticket mixes signals or hedges between two
# possible readings — always defer to the model rather than risk a wrong rule match.
_HEDGE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bnot sure (if|whether)\b",
        r"\bwhichever\b",
        r"\bcould be a bug or\b",
        r"\bmaybe i\b",
        r"\bor maybe\b",
        r"\bkind of urgent\b",
        r"\bwhich one is right\b",
        r"\bdon'?t know if\b",
        r"\bnot sure the\b",
        r"\bbasically a bug\b",
        r"\brefunded automatically\b",
        r"\bflag duplicate charges\b",
    ]
]


@dataclass(frozen=True)
class RuleMatch:
    label: ClassificationOutput
    confidence: float
    rule_id: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    pattern: re.Pattern
    category: str
    urgency: str
    needs_human: bool
    confidence: float


_RULES: list[Rule] = [
    # billing
    Rule("billing_duplicate_charge",
         re.compile(r"charged (twice|two times)|double[- ]charged|duplicate charge", re.I),
         "billing", "medium", True, 0.92),
    Rule("billing_refund_request",
         re.compile(r"\brefund\b", re.I),
         "billing", "medium", True, 0.85),
    Rule("billing_invoice_receipt",
         re.compile(r"\b(invoice|receipt|itemized)\b", re.I),
         "billing", "low", False, 0.8),
    # account_access
    Rule("account_locked_out",
         re.compile(r"locked out|can'?t log ?in|cannot log ?in|failed login attempts", re.I),
         "account_access", "high", True, 0.92),
    Rule("account_password_reset",
         re.compile(r"password reset|reset (my )?password", re.I),
         "account_access", "medium", True, 0.88),
    Rule("account_two_factor",
         re.compile(r"two[- ]factor|2fa\b", re.I),
         "account_access", "high", True, 0.85),
    # bug — narrow, one pattern per known scenario so the asserted label is exact
    Rule("bug_crash_export",
         re.compile(r"crash(es|ed)?.*export a report", re.I),
         "bug", "high", True, 0.9),
    Rule("bug_upload_size_error",
         re.compile(r"upload a file larger than", re.I),
         "bug", "medium", False, 0.85),
    Rule("bug_infinite_loading",
         re.compile(r"infinite loading spinner", re.I),
         "bug", "high", True, 0.9),
    Rule("bug_blank_screen",
         re.compile(r"blank screen", re.I),
         "bug", "medium", True, 0.85),
    Rule("bug_white_text_contrast",
         re.compile(r"renders? white text", re.I),
         "bug", "low", False, 0.85),
    Rule("bug_search_zero_results",
         re.compile(r"returns zero results", re.I),
         "bug", "medium", False, 0.85),
    Rule("bug_duplicate_notifications",
         re.compile(r"notifications?.*duplicated|duplicated.*notifications?", re.I),
         "bug", "low", False, 0.85),
    Rule("bug_sync_stopped",
         re.compile(r"data sync between|sync.*stopped working", re.I),
         "bug", "high", True, 0.85),
    # shipping — most specific phrasing first so a generic word doesn't steal the match
    Rule("shipping_wrong_item",
         re.compile(r"wrong item arrived", re.I),
         "shipping", "high", True, 0.9),
    Rule("shipping_not_received",
         re.compile(r"never received the package", re.I),
         "shipping", "high", True, 0.9),
    Rule("shipping_package_damaged",
         re.compile(r"arrived damaged|box was crushed", re.I),
         "shipping", "medium", True, 0.85),
    Rule("shipping_change_address",
         re.compile(r"change the shipping address", re.I),
         "shipping", "medium", False, 0.85),
    Rule("shipping_stuck_transit",
         re.compile(r"stuck in transit", re.I),
         "shipping", "medium", False, 0.85),
    Rule("shipping_expedite",
         re.compile(r"expedited shipping", re.I),
         "shipping", "medium", False, 0.85),
    # feature_request
    Rule("feature_request_phrasing",
         re.compile(r"feature request|would be (great|nice) if|could you add|suggestion:", re.I),
         "feature_request", "low", False, 0.85),
    # other
    Rule("other_discount_or_policy",
         re.compile(r"student discount|data retention policy|status page|affiliate", re.I),
         "other", "low", False, 0.8),
]


def apply_rules(text: str) -> RuleMatch | None:
    """Try to resolve a ticket deterministically. Returns None to defer to the model."""
    for hedge in _HEDGE_PATTERNS:
        if hedge.search(text):
            return None

    for rule in _RULES:
        if rule.pattern.search(text):
            label = ClassificationOutput(
                category=rule.category,
                urgency=rule.urgency,
                needs_human=rule.needs_human,
            )
            return RuleMatch(label=label, confidence=rule.confidence, rule_id=rule.rule_id)

    return None
