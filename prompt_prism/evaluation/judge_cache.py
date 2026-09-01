"""
Persistent and in-memory caching for LLM-as-a-Judge evaluations to eliminate redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

from pydantic import BaseModel, Field

SCHEMA_VERSION: int = 2


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
        self._warned_sqlite_error = False

        if self.db_path:
            self._init_sqlite()

    def _claim_sqlite_warning(self) -> bool:
        """Return True for the first SQLite failure only.

        Guarded by the same lock as the memory cache: without it, concurrent trials in the
        runner's thread pool can each read False and emit a duplicate warning.
        """
        with self._lock:
            if self._warned_sqlite_error:
                return False
            self._warned_sqlite_error = True
            return True

    def _init_sqlite(self) -> None:
        if not self.db_path:
            return
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.db_path) as conn:
                # No schema_version column: _hash_key already prefixes every key with
                # v{SCHEMA_VERSION}, so a row written under an older layout simply can
                # never be looked up. Storing the version a second time needed a
                # migration and bought nothing. Naming fewer columns than an existing
                # table has is valid, so pre-existing databases keep working untouched.
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
        except sqlite3.Error as e:
            if self._claim_sqlite_warning():
                warnings.warn(
                    f"JudgeCache SQLite initialization failed: {e}",
                    UserWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _hash_key(
        metric_name: str,
        metric_config: str,
        judge_model_id: str,
        input_text: str,
        actual_output: str,
        expected_output: str,
        context: Optional[Sequence[str]] = None,
        retrieval_context: Optional[Sequence[str]] = None,
    ) -> str:
        if not isinstance(judge_model_id, str):
            raise TypeError(
                f"judge_model_id must be a string identifier, got {type(judge_model_id).__name__}"
            )

        ctx_str = json.dumps(list(context)) if context is not None else ""
        ret_ctx_str = (
            json.dumps(list(retrieval_context)) if retrieval_context is not None else ""
        )
        payload = (
            f"v{SCHEMA_VERSION}||{metric_name}||{metric_config}||{judge_model_id}||"
            f"{input_text}||{actual_output}||{expected_output}||{ctx_str}||{ret_ctx_str}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        metric_name: str,
        metric_config: str = "",
        judge_model_id: str = "default",
        input_text: str = "",
        actual_output: str = "",
        expected_output: str = "",
        context: Optional[Sequence[str]] = None,
        retrieval_context: Optional[Sequence[str]] = None,
    ) -> Optional[JudgeResult]:
        """Look up judge score from cache."""
        key = self._hash_key(
            metric_name=metric_name,
            metric_config=metric_config,
            judge_model_id=judge_model_id,
            input_text=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )

        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key]

        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT score, reason, success, token_usage FROM judge_cache WHERE hash_key = ?",
                        (key,),
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
            except sqlite3.Error as e:
                if self._claim_sqlite_warning():
                    warnings.warn(
                        f"JudgeCache SQLite read error: {e}",
                        UserWarning,
                        stacklevel=2,
                    )

        return None

    def set(
        self,
        metric_name: str,
        metric_config: str = "",
        judge_model_id: str = "default",
        input_text: str = "",
        actual_output: str = "",
        expected_output: str = "",
        context: Optional[Sequence[str]] = None,
        retrieval_context: Optional[Sequence[str]] = None,
        score: float = 0.0,
        reason: str = "",
        success: bool = True,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """Store judge result in memory and SQLite."""
        key = self._hash_key(
            metric_name=metric_name,
            metric_config=metric_config,
            judge_model_id=judge_model_id,
            input_text=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )
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
            except sqlite3.Error as e:
                if self._claim_sqlite_warning():
                    warnings.warn(
                        f"JudgeCache SQLite write error: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
