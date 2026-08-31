"""
Persistent and in-memory caching for LLM-as-a-Judge evaluations to eliminate redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    """Cached output from an LLM judge evaluation."""
    score: float
    reason: str = ""
    success: bool = True
    token_usage: Dict[str, int] = Field(default_factory=dict)


class JudgeCache:
    """
    Two-tier (in-memory + SQLite) cache for storing deterministic and expensive LLM judge evaluations.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._memory_cache: Dict[str, JudgeResult] = {}
        self._lock = threading.Lock()

        if self.db_path:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        if not self.db_path:
            return
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS judge_cache (
                    hash_key TEXT PRIMARY KEY,
                    metric_name TEXT,
                    judge_model TEXT,
                    score REAL,
                    reason TEXT,
                    success INTEGER,
                    token_usage TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def _hash_key(
        metric_name: str,
        metric_config: str,
        judge_model_id: str,
        input_text: str,
        actual_output: str,
        expected_output: str,
    ) -> str:
        payload = f"{metric_name}||{metric_config}||{judge_model_id}||{input_text}||{actual_output}||{expected_output}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        metric_name: str,
        metric_config: str = "",
        judge_model_id: str = "default",
        input_text: str = "",
        actual_output: str = "",
        expected_output: str = "",
    ) -> Optional[JudgeResult]:
        """Look up judge score from cache."""
        key = self._hash_key(metric_name, metric_config, judge_model_id, input_text, actual_output, expected_output)

        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key]

        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT score, reason, success, token_usage FROM judge_cache WHERE hash_key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    if row:
                        usage = json.loads(row[3]) if row[3] else {}
                        result = JudgeResult(
                            score=float(row[0]),
                            reason=str(row[1] or ""),
                            success=bool(row[2]),
                            token_usage=usage,
                        )
                        with self._lock:
                            self._memory_cache[key] = result
                        return result
            except Exception:
                pass

        return None

    def set(
        self,
        metric_name: str,
        metric_config: str = "",
        judge_model_id: str = "default",
        input_text: str = "",
        actual_output: str = "",
        expected_output: str = "",
        score: float = 0.0,
        reason: str = "",
        success: bool = True,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """Store judge result in memory and SQLite."""
        key = self._hash_key(metric_name, metric_config, judge_model_id, input_text, actual_output, expected_output)
        res = JudgeResult(
            score=score,
            reason=reason,
            success=success,
            token_usage=token_usage or {},
        )

        with self._lock:
            self._memory_cache[key] = res

        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO judge_cache
                        (hash_key, metric_name, judge_model, score, reason, success, token_usage)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            metric_name,
                            judge_model_id,
                            score,
                            reason,
                            1 if success else 0,
                            json.dumps(token_usage or {}),
                        ),
                    )
                    conn.commit()
            except Exception:
                pass
