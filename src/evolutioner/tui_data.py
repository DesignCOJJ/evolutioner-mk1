"""Telemetry source for the TUI — real data, zero heavy deps.

Reads: Ollama HTTP API (version/tags/ps), /proc (cpu/mem/load), the SQLite
usage tracker, session files and memories. Every accessor degrades
gracefully when a source is unavailable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ollama_backend import DEFAULT_HOST, PORTFOLIO_TAGS
from .usage import UsageTracker

_PROC_STAT = Path("/proc/stat")
_PROC_MEMINFO = Path("/proc/meminfo")
_PROC_LOADAVG = Path("/proc/loadavg")
_PROC_UPTIME = Path("/proc/uptime")


def _fetch_json(url: str, timeout: float = 2.5) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class Telemetry:
    def __init__(self, config, host: Optional[str] = None) -> None:
        self.config = config
        self.host = (host or config.extra.get("ollama_host") or os.environ.get(
            "OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self._last_cpu: Optional[tuple[int, int]] = None  # (idle, total)
        self.cpu_percent: float = 0.0
        self.tracker: Optional[UsageTracker] = None
        self._tracker_failed = False

    # ---------------------------- OLLAMA ---------------------------- #
    @property
    def ollama_up(self) -> bool:
        return _fetch_json(f"{self.host}/api/version", timeout=1.5) is not None

    def ollama_version(self) -> str:
        data = _fetch_json(f"{self.host}/api/version", timeout=1.5)
        return (data or {}).get("version", "—")

    def installed_models(self) -> List[str]:
        data = _fetch_json(f"{self.host}/api/tags", timeout=3.0)
        return [m.get("name", "?") for m in (data or {}).get("models", [])]

    def portfolio_status(self) -> Dict[str, bool]:
        installed = set(self.installed_models())
        return {key: tag in installed for key, tag in PORTFOLIO_TAGS.items()}

    def loaded_models(self) -> List[Dict[str, Any]]:
        data = _fetch_json(f"{self.host}/api/ps", timeout=2.5)
        return [
            {
                "name": m.get("name", "?"),
                "size_vram": m.get("size_vram", 0),
                "size": m.get("size", 0),
                "expires": m.get("expires_at", ""),
            }
            for m in (data or {}).get("models", [])
        ]

    # ---------------------------- SYSTEM ---------------------------- #
    def cpu_sample(self) -> float:
        try:
            fields = _PROC_STAT.read_text().splitlines()[0].split()[1:]
            vals = [int(v) for v in fields]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            if self._last_cpu:
                d_idle, d_total = idle - self._last_cpu[0], total - self._last_cpu[1]
                if d_total > 0:
                    self.cpu_percent = max(0.0, min(1.0, 1 - d_idle / d_total))
            self._last_cpu = (idle, total)
        except (OSError, ValueError, IndexError):
            self.cpu_percent = 0.0
        return self.cpu_percent

    def memory(self) -> Dict[str, float]:
        out: Dict[str, float] = {"total": 0.0, "available": 0.0, "used": 0.0, "frac": 0.0}
        try:
            info: Dict[str, float] = {}
            for line in _PROC_MEMINFO.read_text().splitlines():
                key, val = line.split(":", 1)
                info[key.strip()] = float(val.strip().split()[0])  # kB
            total = info.get("MemTotal", 0.0)
            avail = info.get("MemAvailable", 0.0)
            used = max(0.0, total - avail)
            out = {
                "total": total / 1e6,  # GB
                "available": avail / 1e6,
                "used": used / 1e6,
                "frac": used / total if total else 0.0,
            }
        except (OSError, ValueError, IndexError):
            pass
        return out

    def loadavg(self) -> tuple[float, float, float]:
        try:
            parts = _PROC_LOADAVG.read_text().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
        except (OSError, ValueError, IndexError):
            return (0.0, 0.0, 0.0)

    def uptime(self) -> str:
        try:
            secs = float(_PROC_UPTIME.read_text().split()[0])
            days, rem = divmod(int(secs), 86400)
            h, rem = divmod(rem, 3600)
            m = rem // 60
            return f"{days}d {h:02d}:{m:02d}" if days else f"{h:02d}:{m:02d}"
        except (OSError, ValueError, IndexError):
            return "—"

    # --------------------------- INTERNALS --------------------------- #
    def usage_summary(self) -> Dict[str, Any]:
        if self.tracker is None and not self._tracker_failed:
            try:
                self.tracker = UsageTracker(self.config.usage_db)
            except Exception:
                self._tracker_failed = True
        return self.tracker.summary() if self.tracker else {
            "total_calls": 0, "prompt_tokens": 0, "gen_tokens": 0,
            "total_tokens": 0, "duration_s": 0.0, "avg_tok_per_s": 0.0, "by_kind": {},
        }

    def usage_recent(self, limit: int = 12) -> List[Dict[str, Any]]:
        if self.tracker is None and not self._tracker_failed:
            try:
                self.tracker = UsageTracker(self.config.usage_db)
            except Exception:
                self._tracker_failed = True
        return self.tracker.recent(limit) if self.tracker else []

    def session_count(self) -> int:
        try:
            return len(list(Path(self.config.session_dir).glob("*.json")))
        except OSError:
            return 0

    def memory_count(self) -> int:
        try:
            return len(list(Path(self.config.memories_dir).glob("*.json")))
        except OSError:
            return 0

    def status_badges(self) -> Dict[str, str]:
        up = self.ollama_up
        cpu = self.cpu_sample()
        mem = self.memory()
        badges = {
            "SYSTEM": ("ONLINE", "bright_green") if up else ("OFFLINE", "bright_red"),
            "CORE": ("OPTIMAL", "bright_green") if cpu < 0.85 else ("STRAINED", "bright_yellow"),
            "MEMORY": ("NOMINAL", "bright_green") if mem["frac"] < 0.9 else ("PRESSURE", "bright_yellow"),
            "PROFILE": ("ACTIVE", "bright_cyan") if self.memory_count() or self.session_count() else ("COLD", "grey35"),
        }
        return badges

    def close(self) -> None:
        if self.tracker:
            self.tracker.close()
            self.tracker = None
