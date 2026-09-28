"""Shared pydantic models used across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "bug", "feature_request", "account_access", "shipping", "other"]
Urgency = Literal["low", "medium", "high"]
Source = Literal["rule", "api"]

CATEGORIES: tuple[Category, ...] = (
    "billing",
    "bug",
    "feature_request",
    "account_access",
    "shipping",
    "other",
)
URGENCIES: tuple[Urgency, ...] = ("low", "medium", "high")


class Ticket(BaseModel):
    """A golden-labeled support ticket."""

    id: str
    text: str
    label_category: Category
    label_urgency: Urgency
    label_needs_human: bool


class InjectionTicket(Ticket):
    """A ticket whose body contains a prompt injection attempt."""

    attack_type: str


class ClassificationOutput(BaseModel):
    """The canonical schema every model response must validate against."""

    category: Category
    urgency: Urgency
    needs_human: bool


class FewShotExample(BaseModel):
    input: str
    output: ClassificationOutput


class PromptConfig(BaseModel):
    """A resolved prompt configuration loaded from a prompts/*.yaml file."""

    version: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 300
    system_prompt: str
    few_shot: list[FewShotExample] = Field(default_factory=list)
    output_schema: dict = Field(default_factory=dict)

    def cache_fingerprint(self) -> dict:
        """Fields that affect model output — used as the cache-key basis."""
        return {
            "version": self.version,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "few_shot": [fs.model_dump() for fs in self.few_shot],
        }


class PredictionRecord(BaseModel):
    """One classified ticket, as persisted to a run's predictions.jsonl."""

    id: str
    true_category: Category
    true_urgency: Urgency
    true_needs_human: bool
    pred_category: Category
    pred_urgency: Urgency
    pred_needs_human: bool
    source: Source
    rule_id: str | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    schema_valid: bool = True
    is_injection: bool = False
    attack_type: str | None = None

    @property
    def category_correct(self) -> bool:
        return self.pred_category == self.true_category

    @property
    def urgency_correct(self) -> bool:
        return self.pred_urgency == self.true_urgency

    @property
    def needs_human_correct(self) -> bool:
        return self.pred_needs_human == self.true_needs_human

    @property
    def fully_correct(self) -> bool:
        return self.category_correct and self.urgency_correct and self.needs_human_correct
