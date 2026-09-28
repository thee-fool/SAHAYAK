"""Pydantic settings for kasoti."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KASOTI_", extra="ignore")

    anthropic_api_key: str | None = None

    data_dir: Path = Path("kasoti/data")
    prompts_dir: Path = Path("kasoti/prompts")
    cache_dir: Path = Path(".cache")
    runs_dir: Path = Path("runs")

    max_concurrency: int = 8
    request_timeout_s: float = 60.0
    max_retries: int = 5
    backoff_base_s: float = 1.0

    # Approximate published pricing for claude-haiku, USD per million tokens.
    # Configurable since real pricing/FX drift over time — see README limitations.
    input_price_per_mtok_usd: float = 0.80
    output_price_per_mtok_usd: float = 4.00
    usd_to_inr: float = 88.0

    golden_ticket_count: int = 150
    ambiguous_ticket_count: int = 25
    injection_ticket_count: int = 15
    data_seed: int = 42


settings = Settings()
