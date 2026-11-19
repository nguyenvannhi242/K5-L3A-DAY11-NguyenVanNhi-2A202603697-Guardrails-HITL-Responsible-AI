"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store user input and start timestamp keyed by request_id or user_id."""
        key = request_id or user_id
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "request_id": request_id,
            "input": text,
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output preview, guardrail decision, and latency into self.logs."""
        key = request_id or user_id
        start = self._open.pop(key, None)
        pending = self._pending.pop(key, {})
        latency_ms = (
            round((time.perf_counter() - start) * 1000.0, 2)
            if start is not None
            else 0.0
        )
        entry = {
            "timestamp": pending.get("timestamp") or utc_now_iso(),
            "completed_at": utc_now_iso(),
            "request_id": request_id,
            "user_id": user_id,
            "input": pending.get("input", ""),
            "output_preview": (text or "")[:300],
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None) -> Path:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        out = Path(filepath or default_audit_log_path())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
