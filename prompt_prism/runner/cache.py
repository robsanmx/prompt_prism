"""
Response Caching to avoid duplicate LLM calls and unnecessary token costs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .client import LLMResponse


class ResponseCache:
    """In-memory and SQLite-backed response cache."""

    def __init__(
        self, db_path: Optional[Union[str, Path]] = None, enabled: bool = True
    ):
        self.enabled = enabled
        self.db_path = Path(db_path) if db_path else None
        self._memory_cache: Dict[str, LLMResponse] = {}

        if self.db_path and self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    content TEXT,
                    raw_json TEXT,
                    latency_ms REAL,
                    token_usage_json TEXT
                )
            """)
            conn.commit()

    def _hash_key(self, prompt: Any, params: Dict[str, Any]) -> str:
        serialized = json.dumps(
            {"prompt": prompt, "params": params}, sort_keys=True, default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(
        self, prompt: Any, params: Optional[Dict[str, Any]] = None
    ) -> Optional[LLMResponse]:
        if not self.enabled:
            return None

        h = self._hash_key(prompt, params or {})
        if h in self._memory_cache:
            return self._memory_cache[h]

        if self.db_path and self.db_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT content, raw_json, latency_ms, token_usage_json FROM response_cache WHERE prompt_hash = ?",
                    (h,),
                )
                row = cur.fetchone()
                if row:
                    content, raw_json, latency_ms, token_json = row
                    raw = json.loads(raw_json) if raw_json else None
                    tokens = json.loads(token_json) if token_json else {}
                    res = LLMResponse(
                        content=content,
                        raw=raw,
                        latency_ms=latency_ms,
                        token_usage=tokens,
                    )
                    self._memory_cache[h] = res
                    return res

        return None

    def set(
        self,
        prompt: Any,
        response: LLMResponse,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or response.error:
            return

        h = self._hash_key(prompt, params or {})
        self._memory_cache[h] = response

        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO response_cache (prompt_hash, content, raw_json, latency_ms, token_usage_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        h,
                        response.content,
                        (
                            json.dumps(response.raw, default=str)
                            if response.raw is not None
                            else ""
                        ),
                        response.latency_ms,
                        json.dumps(response.token_usage),
                    ),
                )
                conn.commit()
