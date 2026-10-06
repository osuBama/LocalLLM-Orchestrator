"""Configuration loading.

Everything (paths, models, endpoints, limits) comes from config.yaml.
AI_ROOT overrides paths.root; AI_CONFIG selects a different config file.
"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR.parent / "config" / "config.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    root: str
    memory: str = "memory"
    conversations: str = "conversations"
    database: str = "database/memory.db"
    logs: str = "logs"
    backups: str = "backups"
    prompts: str = "prompts"


class OllamaEndpoint(_Strict):
    base_url: str
    model: str
    timeout_seconds: float = 600
    num_ctx: int | None = None
    keep_alive: str | None = None
    think: bool | None = None


class OllamaConfig(_Strict):
    primary: OllamaEndpoint
    memory: OllamaEndpoint


class MemoryConfig(_Strict):
    project_id: str = "default"
    max_context_tokens: int = Field(2500, gt=0)
    max_memory_file_tokens: int = 4000
    max_entry_tokens: int = 600
    enable_semantic_retrieval: bool = False
    asynchronous_updates: bool = True
    create_backups: bool = True
    history_versions_per_file: int = 50
    max_entry_chars: int = 1500
    min_confidence: float = 0.6
    trigger_mode: Literal["heuristic", "always"] = "heuristic"
    extractor_memory_tokens: int = 1500
    extractor_interaction_tokens: int = 3000
    worker_enabled: bool = True
    worker_poll_seconds: float = 2.0
    max_attempts: int = 5
    retry_base_seconds: float = 30.0


class ConversationConfig(_Strict):
    retain_raw_history: bool = True
    format: Literal["jsonl"] = "jsonl"
    recent_turns: int = Field(2, ge=0)   # short-term window for POST /chat (memory covers the rest)


class ProxyConfig(_Strict):
    enabled: bool = True
    inject_memory: bool = True
    append_system_prompt: bool = True
    queue_memory_updates: bool = True
    # Stepped trimming: once history exceeds trim_trigger_user_turns, cut it back to
    # about trim_keep_user_turns. The cut point then stays fixed for several turns,
    # so Ollama's prompt cache keeps hitting. 0 disables trimming.
    trim_trigger_user_turns: int = Field(10, ge=0)
    trim_keep_user_turns: int = Field(4, ge=1)
    # Never cut turns the session summary does not cover yet.
    trim_requires_summary: bool = True

    @model_validator(mode="after")
    def _check_trim(self):
        if self.trim_trigger_user_turns and self.trim_trigger_user_turns <= self.trim_keep_user_turns:
            raise ValueError("proxy.trim_trigger_user_turns must be greater than trim_keep_user_turns (or 0)")
        return self


class FlagsConfig(_Strict):
    enabled: bool = True
    max_per_turn: int = Field(3, ge=1, le=10)
    max_chars: int = Field(300, ge=20, le=1000)


class CompressionConfig(_Strict):
    """Replace old, large tool results with helper-written digests (stepped, cache-friendly)."""
    enabled: bool = True
    keep_recent_user_turns: int = Field(2, ge=1)   # tool results this recent always stay verbatim
    min_result_tokens: int = Field(400, ge=50)     # smaller results are not worth a digest
    digest_max_tokens: int = Field(200, ge=40, le=1000)
    step_turns: int = Field(6, ge=1)               # schedule used before/without trimming
    never_compress_tools: list[str] = Field(default_factory=list)


class SessionConfig(_Strict):
    summaries_enabled: bool = True
    summary_max_tokens: int = Field(400, ge=50, le=2000)


class ApplicationConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"


class Config(_Strict):
    paths: PathsConfig
    ollama: OllamaConfig
    memory: MemoryConfig = MemoryConfig()
    conversation: ConversationConfig = ConversationConfig()
    proxy: ProxyConfig = ProxyConfig()
    flags: FlagsConfig = FlagsConfig()
    session: SessionConfig = SessionConfig()
    compression: CompressionConfig = CompressionConfig()

    @model_validator(mode="after")
    def _check_compression(self):
        if (self.compression.enabled and self.proxy.trim_trigger_user_turns
                and self.compression.keep_recent_user_turns >= self.proxy.trim_keep_user_turns):
            raise ValueError("compression.keep_recent_user_turns must be smaller than "
                             "proxy.trim_keep_user_turns, or compression would never apply")
        return self
    application: ApplicationConfig = ApplicationConfig()

    # Populated by load_config; not part of the YAML.
    source_path: str | None = None

    # ---- resolved paths -------------------------------------------------
    def _resolve(self, value: str) -> Path:
        p = _native_path(value)
        if p.is_absolute():
            return p
        return self.root_dir / p

    @property
    def root_dir(self) -> Path:
        return _native_path(self.paths.root)

    @property
    def memory_dir(self) -> Path:
        return self._resolve(self.paths.memory)

    @property
    def history_dir(self) -> Path:
        return self.memory_dir / "history"

    @property
    def conversations_dir(self) -> Path:
        return self._resolve(self.paths.conversations)

    @property
    def database_path(self) -> Path:
        return self._resolve(self.paths.database)

    @property
    def logs_dir(self) -> Path:
        return self._resolve(self.paths.logs)

    @property
    def backups_dir(self) -> Path:
        return self._resolve(self.paths.backups)

    @property
    def prompts_dir(self) -> Path:
        p = self._resolve(self.paths.prompts)
        if p.exists():
            return p
        # Fall back to the prompts shipped next to the application code.
        return APP_DIR.parent / "prompts"

    def ensure_dirs(self) -> None:
        for d in (self.memory_dir, self.history_dir, self.conversations_dir,
                  self.database_path.parent, self.logs_dir, self.backups_dir):
            d.mkdir(parents=True, exist_ok=True)

    def prompt(self, name: str) -> str:
        return (self.prompts_dir / name).read_text(encoding="utf-8").strip()


def _native_path(value: str) -> Path:
    """Accept Windows-style paths in YAML even when running elsewhere (tests, WSL)."""
    if os.name != "nt" and ("\\" in value or (len(value) > 1 and value[1] == ":")):
        win = PureWindowsPath(value)
        if win.drive:
            # G:\AI -> /mnt/g/AI when running under WSL/Linux.
            drive = win.drive.rstrip(":").lower()
            return Path("/mnt", drive, *win.parts[1:])
        return Path(*win.parts)
    return Path(value)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("AI_CONFIG") or DEFAULT_CONFIG_PATH)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if os.environ.get("AI_ROOT"):
        data.setdefault("paths", {})["root"] = os.environ["AI_ROOT"]
    cfg = Config(**data)
    cfg.source_path = str(cfg_path)
    return cfg
