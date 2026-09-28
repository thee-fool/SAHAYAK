import asyncio

import pytest

from kasoti.models import PromptConfig, Ticket
from kasoti.runner import classify_ticket


class _FakeBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(input_tokens=12, output_tokens=6)


class _FakeMessages:
    def __init__(self, text: str):
        self._text = text
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text: str):
        self.messages = _FakeMessages(text)


def _ticket(text: str = "something that no rule matches at all") -> Ticket:
    return Ticket(
        id="t-1",
        text=text,
        label_category="bug",
        label_urgency="medium",
        label_needs_human=True,
    )


def _prompt_config(version: str) -> PromptConfig:
    return PromptConfig(version=version, model="test-model", system_prompt="classify the ticket")


@pytest.mark.asyncio
async def test_cache_hit_avoids_a_second_api_call(tmp_path):
    ticket = _ticket()
    prompt_config = _prompt_config("cache-test")
    client = _FakeClient('{"category": "bug", "urgency": "medium", "needs_human": true}')
    semaphore = asyncio.Semaphore(1)
    cache_dir = tmp_path / "cache"

    record1, _ = await classify_ticket(
        ticket, prompt_config, mock=False, cache_dir=cache_dir, client=client, semaphore=semaphore
    )
    record2, _ = await classify_ticket(
        ticket, prompt_config, mock=False, cache_dir=cache_dir, client=client, semaphore=semaphore
    )

    assert client.messages.call_count == 1, "second identical call should be served from the cache"
    assert record1.pred_category == record2.pred_category == "bug"
    assert record1.source == record2.source == "api"
    assert list(cache_dir.glob("*.json")), "expected a cache file to have been written"


@pytest.mark.asyncio
async def test_schema_violation_is_counted_not_raised(tmp_path):
    ticket = _ticket()
    prompt_config = _prompt_config("schema-test")
    # Missing required fields / malformed key — should fail ClassificationOutput validation.
    client = _FakeClient('{"category": "bug", "urgenccy": "medium"}')
    semaphore = asyncio.Semaphore(1)

    record, raw_log = await classify_ticket(
        ticket, prompt_config, mock=False, cache_dir=tmp_path / "cache", client=client, semaphore=semaphore
    )

    assert record.schema_valid is False
    assert raw_log["schema_valid"] is False
    # A safe fallback label is used instead of propagating the exception.
    assert record.pred_category == "other"
    assert record.pred_urgency == "low"
    assert record.pred_needs_human is False


@pytest.mark.asyncio
async def test_rule_match_short_circuits_the_api_entirely(tmp_path):
    ticket = _ticket(text="I'm locked out of my account after too many failed login attempts.")
    prompt_config = _prompt_config("rule-test")
    client = _FakeClient('{"category": "bug", "urgency": "low", "needs_human": false}')
    semaphore = asyncio.Semaphore(1)

    record, raw_log = await classify_ticket(
        ticket, prompt_config, mock=False, cache_dir=tmp_path / "cache", client=client, semaphore=semaphore
    )

    assert client.messages.call_count == 0
    assert record.source == "rule"
    assert record.pred_category == "account_access"
    assert raw_log["source"] == "rule"
