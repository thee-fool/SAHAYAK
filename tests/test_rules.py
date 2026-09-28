import pytest

from kasoti.rules import apply_rules


@pytest.mark.parametrize(
    ("text", "expected_category", "expected_urgency", "expected_needs_human"),
    [
        ("I was charged twice for the same order, please refund me.", "billing", "medium", True),
        ("Could I get an itemized receipt for my last order?", "billing", "low", False),
        (
            "I'm locked out of my account after too many failed login attempts.",
            "account_access", "high", True,
        ),
        ("My password reset email never arrived, checked spam already.", "account_access", "medium", True),
        ("The app shows a blank screen after the latest update.", "bug", "medium", True),
        ("Dark mode renders white text on a white background, hard to read.", "bug", "low", False),
        ("Wrong item arrived for my order, I got something else entirely.", "shipping", "high", True),
        ("Can you change the shipping address on my order? It hasn't shipped yet.", "shipping", "medium", False),
        ("Would be great if the app supported exporting data to CSV.", "feature_request", "low", False),
        ("Do you offer a student discount for the premium plan?", "other", "low", False),
    ],
)
def test_rules_fire_on_clear_signals(text, expected_category, expected_urgency, expected_needs_human):
    match = apply_rules(text)
    assert match is not None, f"expected a rule to fire on: {text!r}"
    assert match.label.category == expected_category
    assert match.label.urgency == expected_urgency
    assert match.label.needs_human is expected_needs_human
    assert match.rule_id
    assert 0.0 < match.confidence <= 1.0


@pytest.mark.parametrize(
    "text",
    [
        "Not sure if this is a bug or a billing issue, could be either honestly.",
        "I want a refund or my access back, whichever is faster.",
        "This might just be a display quirk, or maybe I misconfigured something.",
        "The tracking page and the app show two different delivery dates, which one is right?",
        "Just a general question about your company, nothing urgent here.",
    ],
)
def test_rules_defer_on_ambiguous_or_unmatched_text(text):
    assert apply_rules(text) is None
