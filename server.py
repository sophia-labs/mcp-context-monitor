#!/usr/bin/env python3
"""Context Window Monitor — MCP server for context estimation.

Supports multiple backends (Claude Code, Codex CLI) via config file.
Estimates current context window usage by parsing the active session's
JSONL transcript.

Configuration (loaded in order, later overrides earlier):
  1. Built-in defaults per backend
  2. Config file: ~/.config/context-monitor/config.toml
  3. Environment variables: CONTEXT_MONITOR_*

Transcript detection:
  The server detects the active transcript ONCE at startup (most recently
  modified .jsonl in the project/session directory). Since each MCP server
  lives exactly as long as its host session, this avoids race conditions.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from glob import glob
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Backend defaults
# ---------------------------------------------------------------------------

BACKEND_DEFAULTS = {
    "claude-code": {
        "context_window": 200_000,
        "autocompact_buffer": 33_000,
        "static_overhead": 43_500,
        "bytes_per_token": 3.2,
        "transcript_dir": "~/.claude/projects",
        "state_dir": "~/.claude",
        "compaction_marker": "continued from a previous conversation that ran out of context",
        "compaction_check": "content_starts_with",  # how to verify compaction events
        "compaction_content_prefix": "This session is being",
        "use_native_token_counts": False,
        "transcript_pattern": "*.jsonl",
        "skip_prefix": "agent-",
        # JSONL structure: {message: {role, content}} with content blocks
        "format": "claude",
    },
    "codex-cli": {
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "autocompact_buffer": 0,  # Codex uses ratio-based threshold, not fixed buffer
        "autocompact_ratio": 0.95,  # compaction at 95% of effective window
        "static_overhead": 30_000,  # needs calibration
        "bytes_per_token": 3.2,
        "transcript_dir": "~/.codex/sessions",
        "state_dir": "~/.codex",
        "use_native_token_counts": True,  # Codex embeds token counts in turn events
        "transcript_pattern": "**/*.jsonl",  # Codex nests in YYYY/MM/DD subdirs
        "skip_prefix": "",
        "format": "codex",
    },
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("~/.config/context-monitor/config.toml").expanduser()


def _load_config() -> dict:
    """Load configuration from TOML file, with env var overrides."""
    config = {}

    # Load TOML if it exists
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                config = tomllib.load(f)
            print(f"[context-monitor] Config loaded: {CONFIG_PATH}", file=sys.stderr)
        except Exception as e:
            print(f"[context-monitor] Config error: {e}", file=sys.stderr)

    # Determine backend
    backend = (
        os.environ.get("CONTEXT_MONITOR_BACKEND")
        or config.get("backend", {}).get("type", "auto")
    )

    if backend == "auto":
        backend = _detect_backend()

    # Merge: defaults -> config file -> env vars
    defaults = BACKEND_DEFAULTS.get(backend, BACKEND_DEFAULTS["claude-code"])
    backend_config = config.get(backend.replace("-", "_"), config.get(backend, {}))

    merged = {**defaults, **backend_config}
    merged["backend"] = backend

    # Env var overrides (same names as before for backwards compat)
    env_overrides = {
        "CONTEXT_MONITOR_OVERHEAD": ("static_overhead", int),
        "CONTEXT_MONITOR_BUFFER": ("autocompact_buffer", int),
        "CONTEXT_MONITOR_WINDOW": ("context_window", int),
        "CONTEXT_MONITOR_BPT": ("bytes_per_token", float),
        "CONTEXT_MONITOR_PROJECT_DIR": ("transcript_dir", str),
        "CONTEXT_MONITOR_TRANSCRIPT": ("transcript_path", str),
        "CONTEXT_MONITOR_STATE_DIR": ("state_dir", str),
    }
    for env_key, (config_key, cast) in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            merged[config_key] = cast(val)

    # Compute effective ceiling
    if merged.get("autocompact_buffer", 0) > 0:
        merged["effective_ceiling"] = merged["context_window"] - merged["autocompact_buffer"]
    elif merged.get("autocompact_ratio"):
        effective_window = merged["context_window"] - merged.get("max_output_tokens", 0)
        merged["effective_ceiling"] = int(effective_window * merged["autocompact_ratio"])
    else:
        merged["effective_ceiling"] = merged["context_window"]

    return merged


def _detect_backend() -> str:
    """Auto-detect which CLI tool we're running under."""
    # Check for Codex CLI markers
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    if codex_home.is_dir() and (codex_home / "sessions").is_dir():
        # Also check if Claude is present — prefer whichever has more recent transcripts
        claude_projects = Path("~/.claude/projects").expanduser()
        if claude_projects.is_dir():
            codex_latest = _latest_mtime(codex_home / "sessions", "*.jsonl")
            claude_latest = _latest_mtime_recursive(claude_projects, "*.jsonl")
            if codex_latest > claude_latest:
                return "codex-cli"
            return "claude-code"
        return "codex-cli"

    return "claude-code"


def _latest_mtime(directory: Path, pattern: str) -> float:
    """Get the most recent mtime of files matching pattern in directory."""
    best = 0.0
    for f in directory.glob(pattern):
        try:
            mt = f.stat().st_mtime
            if mt > best:
                best = mt
        except OSError:
            pass
    return best


def _latest_mtime_recursive(directory: Path, pattern: str) -> float:
    """Get the most recent mtime of files matching pattern recursively."""
    best = 0.0
    for f in directory.rglob(pattern):
        if f.name.startswith("agent-"):
            continue
        try:
            mt = f.stat().st_mtime
            if mt > best:
                best = mt
        except OSError:
            pass
    return best


# ---------------------------------------------------------------------------
# Load config at startup
# ---------------------------------------------------------------------------

CFG = _load_config()

# Status thresholds
THRESHOLD_SOON = 75
THRESHOLD_IMMINENT = 90

# Sidecar state
SIDECAR_DIR = Path(CFG.get("state_dir", "~/.claude")).expanduser()
SIDECAR_FILE = SIDECAR_DIR / "context-monitor-state.json"


# ---------------------------------------------------------------------------
# Transcript detection (runs once at startup)
# ---------------------------------------------------------------------------

def _find_project_dir_claude() -> str | None:
    """Find Claude Code project transcript directory."""
    transcript_dir = Path(CFG["transcript_dir"]).expanduser()
    if not transcript_dir.is_dir():
        return None

    # Match CWD-encoded directory name
    cwd = os.getcwd()
    encoded_cwd = cwd.replace("/", "-")
    for d in transcript_dir.iterdir():
        if d.is_dir() and (d.name == encoded_cwd or d.name == encoded_cwd.lstrip("-")):
            return str(d)

    # Fallback: most recent transcript across all project dirs
    best_dir = None
    best_mtime = 0.0
    for d in transcript_dir.iterdir():
        if not d.is_dir():
            continue
        for f in d.glob(CFG["transcript_pattern"]):
            if CFG["skip_prefix"] and f.name.startswith(CFG["skip_prefix"]):
                continue
            mt = f.stat().st_mtime
            if mt > best_mtime:
                best_mtime = mt
                best_dir = str(d)
    return best_dir


def _find_project_dir_codex() -> str | None:
    """Find Codex CLI session directory."""
    session_dir = Path(CFG["transcript_dir"]).expanduser()
    if session_dir.is_dir():
        return str(session_dir)
    return None


def _find_transcript(project_dir: str | None) -> str | None:
    """Find the most recently modified transcript."""
    # Explicit override
    explicit = CFG.get("transcript_path") or os.environ.get("CONTEXT_MONITOR_TRANSCRIPT")
    if explicit:
        return explicit

    if not project_dir or not os.path.isdir(project_dir):
        return None

    candidates = []
    pattern = os.path.join(project_dir, CFG["transcript_pattern"])
    for f in glob(pattern, recursive=True):
        basename = os.path.basename(f)
        if CFG["skip_prefix"] and basename.startswith(CFG["skip_prefix"]):
            continue
        candidates.append(f)

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


# Detect project dir at startup (stable — based on CWD)
if CFG["backend"] == "codex-cli":
    _PROJECT_DIR = _find_project_dir_codex()
else:
    _PROJECT_DIR = _find_project_dir_claude()

_TRANSCRIPT_PATH: str | None = None  # detected on first call, then locked in

print(f"[context-monitor] Backend: {CFG['backend']}", file=sys.stderr)
if _PROJECT_DIR:
    print(f"[context-monitor] Project dir: {_PROJECT_DIR}", file=sys.stderr)
else:
    print("[context-monitor] WARNING: No project directory found.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sidecar state (incremental scanning)
# ---------------------------------------------------------------------------

def _read_sidecar() -> dict:
    if SIDECAR_FILE.exists():
        try:
            return json.loads(SIDECAR_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_sidecar(state: dict) -> None:
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SIDECAR_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(SIDECAR_FILE)


# ---------------------------------------------------------------------------
# Claude Code parser
# ---------------------------------------------------------------------------

CLAUDE_COMPACTION_MARKER = CFG.get(
    "compaction_marker",
    "continued from a previous conversation that ran out of context"
).encode("utf-8")

CLAUDE_CONTENT_PREFIX = CFG.get("compaction_content_prefix", "This session is being")


def _is_real_compaction_claude(line_bytes: bytes) -> bool:
    """Verify a line is a real Claude Code compaction event."""
    try:
        obj = json.loads(line_bytes)
    except json.JSONDecodeError:
        return False

    if not isinstance(obj, dict):
        return False

    msg = obj.get("message", {})
    if not isinstance(msg, dict):
        return False

    if msg.get("role") != "user":
        return False

    content = msg.get("content", "")
    if isinstance(content, str) and CLAUDE_CONTENT_PREFIX in content[:50]:
        return True

    return False


def _find_last_compaction_claude(filepath: str, sidecar: dict) -> tuple[int, dict]:
    """Find byte offset of last compaction in a Claude Code transcript."""
    file_size = os.path.getsize(filepath)

    cached_path = sidecar.get("transcript_path")
    cached_offset = sidecar.get("last_compaction_offset", 0)
    cached_file_size = sidecar.get("file_size_at_compaction_scan", 0)

    start_scan = 0
    last_compaction_offset = 0
    if cached_path == filepath and cached_offset >= 0 and cached_file_size <= file_size:
        start_scan = cached_offset
        last_compaction_offset = cached_offset

    current_offset = start_scan

    with open(filepath, "rb") as f:
        if start_scan > 0:
            f.seek(start_scan)

        for line in f:
            if CLAUDE_COMPACTION_MARKER in line:
                if _is_real_compaction_claude(line):
                    last_compaction_offset = current_offset
            current_offset += len(line)

    sidecar["transcript_path"] = filepath
    sidecar["last_compaction_offset"] = last_compaction_offset
    sidecar["file_size_at_compaction_scan"] = file_size

    return last_compaction_offset, sidecar


def _estimate_usage_claude(filepath: str, compaction_offset: int, sidecar: dict) -> tuple[dict, dict]:
    """Parse Claude Code transcript from compaction_offset and estimate token usage."""
    file_size = os.path.getsize(filepath)

    cached_path = sidecar.get("transcript_path")
    cached_compaction = sidecar.get("last_compaction_offset", 0)
    cached_scan_end = sidecar.get("last_stats_scan_offset", 0)
    cached_stats = sidecar.get("running_stats")

    if (cached_path == filepath
            and cached_compaction == compaction_offset
            and cached_stats is not None
            and cached_scan_end > compaction_offset
            and cached_scan_end <= file_size):
        scan_from = cached_scan_end
        stats = dict(cached_stats)
    else:
        scan_from = compaction_offset
        stats = _empty_stats()

    with open(filepath, "rb") as f:
        f.seek(scan_from)

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["entry_count"] += 1

            if not isinstance(obj, dict) or "message" not in obj:
                stats["metadata_bytes"] += len(line)
                continue

            msg = obj["message"]
            if not isinstance(msg, dict):
                stats["metadata_bytes"] += len(line)
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            content_json = json.dumps(content).encode("utf-8")
            wrapper_overhead = len(line) - len(content_json)
            stats["metadata_bytes"] += max(0, wrapper_overhead)

            if isinstance(content, str):
                content_bytes = len(content.encode("utf-8"))
                if role == "user":
                    stats["user_message_count"] += 1
                    if (CLAUDE_COMPACTION_MARKER.decode() in content
                            and CLAUDE_CONTENT_PREFIX in content[:50]):
                        stats["compaction_summary_bytes"] += content_bytes
                    else:
                        stats["text_bytes"] += content_bytes
                elif role == "system":
                    stats["system_bytes"] += content_bytes
                else:
                    stats["assistant_message_count"] += 1
                    stats["text_bytes"] += content_bytes

            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    block_bytes = len(json.dumps(block).encode("utf-8"))

                    if btype == "thinking":
                        stats["thinking_bytes"] += block_bytes
                    elif btype == "tool_use":
                        stats["tool_use_bytes"] += block_bytes
                        stats["tool_call_count"] += 1
                    elif btype == "tool_result":
                        stats["tool_result_bytes"] += block_bytes
                    elif btype == "text":
                        stats["text_bytes"] += block_bytes
                    else:
                        stats["text_bytes"] += block_bytes

                if role == "user":
                    stats["user_message_count"] += 1
                elif role == "assistant":
                    stats["assistant_message_count"] += 1

    sidecar["last_stats_scan_offset"] = file_size
    sidecar["running_stats"] = stats

    return stats, sidecar


# ---------------------------------------------------------------------------
# Codex CLI parser
# ---------------------------------------------------------------------------

def _find_last_compaction_codex(filepath: str, sidecar: dict) -> tuple[int, dict]:
    """Find byte offset of last compaction in a Codex CLI transcript.

    Codex compaction events have top-level type: "compacted".
    However, since we use native token counts from token_count events
    (which already reflect post-compaction state), compaction detection
    is only used for activity counting, not for token estimation.
    """
    file_size = os.path.getsize(filepath)

    cached_path = sidecar.get("transcript_path")
    cached_offset = sidecar.get("last_compaction_offset", 0)
    cached_file_size = sidecar.get("file_size_at_compaction_scan", 0)

    start_scan = 0
    last_compaction_offset = 0
    if cached_path == filepath and cached_offset >= 0 and cached_file_size <= file_size:
        start_scan = cached_offset
        last_compaction_offset = cached_offset

    current_offset = start_scan

    with open(filepath, "rb") as f:
        if start_scan > 0:
            f.seek(start_scan)

        for line in f:
            if b'"compacted"' in line:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("type") == "compacted":
                        last_compaction_offset = current_offset
                except json.JSONDecodeError:
                    pass

            current_offset += len(line)

    sidecar["transcript_path"] = filepath
    sidecar["last_compaction_offset"] = last_compaction_offset
    sidecar["file_size_at_compaction_scan"] = file_size

    return last_compaction_offset, sidecar


def _estimate_usage_codex(filepath: str, compaction_offset: int, sidecar: dict) -> tuple[dict, dict]:
    """Parse Codex CLI transcript and estimate token usage.

    Codex transcripts include native token counts in token_count events
    (type: "event_msg", payload.type: "token_count"). The key fields:
      - payload.info.last_token_usage.input_tokens = current turn's context size
      - payload.info.model_context_window = effective ceiling (258400)

    Since last_token_usage already reflects post-compaction state, we don't
    need compaction boundary detection for token estimation — we just need
    the most recent token_count event with info set.
    """
    file_size = os.path.getsize(filepath)

    # For Codex, we always scan the whole file (or from cache) to find
    # the latest token_count event. Compaction offset isn't used for
    # token estimation since native counts handle it.
    cached_path = sidecar.get("transcript_path")
    cached_scan_end = sidecar.get("last_stats_scan_offset", 0)
    cached_stats = sidecar.get("running_stats")

    if (cached_path == filepath
            and cached_stats is not None
            and cached_scan_end <= file_size):
        scan_from = cached_scan_end
        stats = dict(cached_stats)
    else:
        scan_from = 0
        stats = _empty_stats()
        stats["native_input_tokens"] = 0
        stats["native_output_tokens"] = 0
        stats["native_cached_tokens"] = 0
        stats["native_reasoning_tokens"] = 0
        stats["model_context_window"] = 0
        stats["has_native_counts"] = False
        stats["compaction_count"] = 0

    with open(filepath, "rb") as f:
        f.seek(scan_from)

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            stats["entry_count"] += 1
            record_type = obj.get("type", "")

            # Native token counts from token_count events
            if record_type == "event_msg":
                payload = obj.get("payload", {})
                if isinstance(payload, dict) and payload.get("type") == "token_count":
                    info = payload.get("info")
                    if isinstance(info, dict):
                        last_usage = info.get("last_token_usage", {})
                        if last_usage:
                            stats["has_native_counts"] = True
                            stats["native_input_tokens"] = last_usage.get("input_tokens", 0)
                            stats["native_cached_tokens"] = last_usage.get("cached_input_tokens", 0)
                            stats["native_output_tokens"] = last_usage.get("output_tokens", 0)
                            stats["native_reasoning_tokens"] = last_usage.get("reasoning_output_tokens", 0)
                        mcw = info.get("model_context_window")
                        if mcw:
                            stats["model_context_window"] = mcw
                elif isinstance(payload, dict) and payload.get("type") == "task_complete":
                    stats["assistant_message_count"] += 1
                elif isinstance(payload, dict) and payload.get("type") == "task_started":
                    stats["user_message_count"] += 1

            # Compaction events
            elif record_type == "compacted":
                stats["compaction_count"] = stats.get("compaction_count", 0) + 1

            # Response items for activity counting
            elif record_type == "response_item":
                payload = obj.get("payload", {})
                if isinstance(payload, dict):
                    item_type = payload.get("type", "")
                    if item_type == "function_call":
                        stats["tool_call_count"] += 1

    sidecar["last_stats_scan_offset"] = file_size
    sidecar["running_stats"] = stats

    return stats, sidecar


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _empty_stats() -> dict:
    return {
        "text_bytes": 0,
        "tool_use_bytes": 0,
        "tool_result_bytes": 0,
        "system_bytes": 0,
        "thinking_bytes": 0,
        "compaction_summary_bytes": 0,
        "metadata_bytes": 0,
        "entry_count": 0,
        "tool_call_count": 0,
        "user_message_count": 0,
        "assistant_message_count": 0,
    }


def _build_report(stats: dict, filepath: str) -> dict:
    """Build the context status report from raw stats."""
    bpt = CFG["bytes_per_token"]

    # If Codex with native token counts, use those directly
    if stats.get("has_native_counts"):
        # native_input_tokens = last_token_usage.input_tokens = current context size
        total_tokens = stats["native_input_tokens"]
        # Use model_context_window from the event if available (more accurate
        # than config since it comes directly from the server)
        effective_ceiling = stats.get("model_context_window") or CFG["effective_ceiling"]
    else:
        # Estimate from byte counts (Claude Code path)
        in_context_bytes = (
            stats["text_bytes"]
            + stats["tool_use_bytes"]
            + stats["tool_result_bytes"]
            + stats["system_bytes"]
            + stats["compaction_summary_bytes"]
        )
        dynamic_tokens = int(in_context_bytes / bpt)
        total_tokens = dynamic_tokens + CFG["static_overhead"]
        effective_ceiling = CFG["effective_ceiling"]

    remaining = max(0, effective_ceiling - total_tokens)
    if stats.get("has_native_counts"):
        # For Codex, model_context_window IS the effective ceiling
        usage_pct = round((total_tokens / effective_ceiling) * 100, 1) if effective_ceiling > 0 else 0
    else:
        usage_pct = round((total_tokens / CFG["context_window"]) * 100, 1)
    compact_pct = round((total_tokens / effective_ceiling) * 100, 1) if effective_ceiling > 0 else 0

    if compact_pct >= THRESHOLD_IMMINENT:
        status = "CRITICAL"
    elif compact_pct >= THRESHOLD_SOON:
        status = "HIGH"
    elif compact_pct >= 50:
        status = "MODERATE"
    else:
        status = "OK"

    report = {
        "status": status,
        "usage_percent": usage_pct,
        "compaction_percent": compact_pct,
        "estimated_tokens_used": total_tokens,
        "estimated_tokens_remaining": remaining,
        "effective_ceiling": effective_ceiling,
        "backend": CFG["backend"],
        "activity": {
            "entries": stats["entry_count"],
            "user_messages": stats["user_message_count"],
            "assistant_messages": stats["assistant_message_count"],
            "tool_calls": stats["tool_call_count"],
        },
        "transcript": os.path.basename(filepath),
    }

    if stats.get("has_native_counts"):
        report["breakdown"] = {
            "input_tokens": stats["native_input_tokens"],
            "cached_tokens": stats["native_cached_tokens"],
            "output_tokens": stats["native_output_tokens"],
            "reasoning_tokens": stats["native_reasoning_tokens"],
            "compactions": stats.get("compaction_count", 0),
        }
    else:
        report["breakdown"] = {
            "dynamic_content_tokens": int((
                stats["text_bytes"]
                + stats["tool_use_bytes"]
                + stats["tool_result_bytes"]
                + stats["system_bytes"]
                + stats["compaction_summary_bytes"]
            ) / bpt),
            "static_overhead_tokens": CFG["static_overhead"],
            "text_tokens": int(stats["text_bytes"] / bpt),
            "tool_use_tokens": int(stats["tool_use_bytes"] / bpt),
            "tool_result_tokens": int(stats["tool_result_bytes"] / bpt),
            "system_tokens": int(stats["system_bytes"] / bpt),
            "compaction_summary_tokens": int(stats["compaction_summary_bytes"] / bpt),
            "thinking_tokens_excluded": int(stats["thinking_bytes"] / bpt),
        }

    return report


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def _find_compaction(filepath: str, sidecar: dict) -> tuple[int, dict]:
    if CFG["backend"] == "codex-cli":
        return _find_last_compaction_codex(filepath, sidecar)
    return _find_last_compaction_claude(filepath, sidecar)


def _estimate(filepath: str, compaction_offset: int, sidecar: dict) -> tuple[dict, dict]:
    if CFG["backend"] == "codex-cli":
        return _estimate_usage_codex(filepath, compaction_offset, sidecar)
    return _estimate_usage_claude(filepath, compaction_offset, sidecar)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "context-monitor",
    instructions=(
        "Context window usage monitor. Call context_status() to check how "
        "close you are to context compaction. Use this to self-manage: write "
        "key insights to persistent storage (memory queue, documents) before "
        "compaction erases them. Status levels: OK (<50%), MODERATE (50-75%), "
        "HIGH (75-90%, start saving state), CRITICAL (>90%, save NOW). "
        "Optionally pass transcript_path to target a specific session file."
    ),
)


@mcp.tool(
    name="context_status",
    description=(
        "Estimate current context window usage and distance to compaction. "
        "Returns status (OK/MODERATE/HIGH/CRITICAL), usage percentages, "
        "token estimates. Use this to self-manage: "
        "OK (<50%), MODERATE (50-75%), HIGH (75-90%, start saving state), "
        "CRITICAL (>90%, save NOW). "
        "Optionally pass transcript_path to target a specific session file."
    ),
)
def context_status_tool(transcript_path: Optional[str] = None) -> dict:
    """Estimate context window usage from the session transcript."""
    # Detect transcript on first call, then lock in. Can't detect at startup
    # because the session JSONL doesn't exist yet when MCP servers start.
    # By first call, the current session's file exists and is most recent.
    global _TRANSCRIPT_PATH
    if _TRANSCRIPT_PATH is None:
        _TRANSCRIPT_PATH = _find_transcript(_PROJECT_DIR)
        if _TRANSCRIPT_PATH:
            print(f"[context-monitor] Tracking: {os.path.basename(_TRANSCRIPT_PATH)}", file=sys.stderr)
    filepath = transcript_path or _TRANSCRIPT_PATH

    if not filepath or not os.path.exists(filepath):
        return {
            "error": "No transcript found",
            "hint": (
                "Set CONTEXT_MONITOR_TRANSCRIPT=/path/to/file.jsonl or "
                "CONTEXT_MONITOR_PROJECT_DIR, or create a config file at "
                f"{CONFIG_PATH}"
            ),
        }

    sidecar = _read_sidecar()

    compaction_offset, sidecar = _find_compaction(filepath, sidecar)
    stats, sidecar = _estimate(filepath, compaction_offset, sidecar)

    _write_sidecar(sidecar)

    report = _build_report(stats, filepath)

    return {
        "status": report["status"],
        "usage_percent": report["usage_percent"],
        "compaction_percent": report["compaction_percent"],
        "estimated_tokens_used": report["estimated_tokens_used"],
        "estimated_tokens_remaining": report["estimated_tokens_remaining"],
    }


if __name__ == "__main__":
    mcp.run()
