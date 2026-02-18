# mcp-context-monitor

Context window usage estimation for AI coding agents via [MCP](https://modelcontextprotocol.io/).

Parses your session transcript to estimate how close you are to context compaction, so agents can proactively save important state before it's lost. Supports Claude Code and Codex CLI backends with auto-detection.

## Why

AI coding agents accumulate context through conversation, tool calls, and file reads. When the context window fills up, compaction discards older content. Agents that know compaction is coming can write key insights to persistent storage first — memory queues, documents, knowledge graphs — instead of losing them silently.

## Features

- **Live estimation** — Single MCP tool returns usage percentage, distance to compaction, and status level
- **Multi-backend** — Supports Claude Code (JSONL transcripts) and Codex CLI (native token counts) with auto-detection
- **Incremental scanning** — Sidecar cache tracks scan position; subsequent calls only process new bytes
- **Compaction-aware** — Finds the last compaction boundary and measures only post-compaction content
- **Configurable** — TOML config for thresholds, token ratios, and backend-specific settings
- **Zero infrastructure** — Reads the transcript file directly, no daemon or network calls

## Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sophia-labs/mcp-context-monitor.git
cd mcp-context-monitor
uv sync
```

### Claude Code

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "context-monitor": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-context-monitor", "python", "server.py"]
    }
  }
}
```

### Codex CLI

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.context-monitor]
command = "uv"
args = ["run", "--directory", "/path/to/mcp-context-monitor", "python", "server.py"]
```

The backend is auto-detected based on which CLI has the most recent transcript.

## Usage

Call `context_status()` from your agent:

```json
{
  "status": "HIGH",
  "usage_percent": 73.9,
  "compaction_percent": 88.5,
  "estimated_tokens_used": 147780,
  "estimated_tokens_remaining": 19220
}
```

### Status Levels

| Status | Compaction % | Recommended Action |
|--------|-------------|-------------------|
| OK | < 50% | Normal operation |
| MODERATE | 50–75% | Be aware, no action needed |
| HIGH | 75–90% | Start saving important state to persistent storage |
| CRITICAL | 90%+ | Save everything immediately — compaction is imminent |

### How Agents Should Use This

- Call `context_status()` periodically during long sessions
- At **HIGH**: write key insights to memory queue, sing if at a phase transition
- At **CRITICAL**: write everything important to persistent storage immediately
- The `compaction_percent` measures distance to the compaction trigger, not the total window

## Configuration

Create `~/.config/context-monitor/config.toml`:

```toml
# Backend selection: "auto", "claude-code", or "codex-cli"
[backend]
type = "auto"

# Claude Code settings
[claude-code]
context_window = 200000
autocompact_buffer = 33000
static_overhead = 43500
bytes_per_token = 3.2
# transcript_dir = "~/.claude/projects"

# Codex CLI settings
[codex-cli]
context_window = 400000
max_output_tokens = 128000
autocompact_ratio = 0.95
static_overhead = 30000
bytes_per_token = 3.2
# transcript_dir = "~/.codex/sessions"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CONTEXT_MONITOR_BACKEND` | Force backend: `claude-code` or `codex-cli` |
| `CONTEXT_MONITOR_WINDOW` | Context window size (tokens) |
| `CONTEXT_MONITOR_BUFFER` | Autocompact buffer (tokens) |
| `CONTEXT_MONITOR_OVERHEAD` | Static overhead estimate (tokens) |
| `CONTEXT_MONITOR_BPT` | Bytes-per-token ratio |
| `CONTEXT_MONITOR_TRANSCRIPT` | Explicit transcript file path |
| `CONTEXT_MONITOR_PROJECT_DIR` | Transcript directory |

## How It Works

1. **Startup**: Auto-detects backend (Claude Code or Codex CLI) and finds the active session transcript
2. **Compaction detection**: Scans for compaction markers to find the boundary of current context
3. **Content estimation**: Parses post-compaction content, categorizing by type (text, tool calls, tool results, thinking, system)
4. **Token estimation**:
   - *Claude Code*: Estimates tokens from byte counts using a calibrated bytes-per-token ratio
   - *Codex CLI*: Uses native token counts from `turn_complete` events when available
5. **Caching**: Stores scan position in a sidecar file so subsequent calls only process new bytes

### What's Counted

- User messages, assistant messages, system prompts
- Tool use (function calls) and tool results
- Compaction summaries (from prior compactions)

### What's Excluded

- Thinking/reasoning blocks (not retained in context after generation)
- JSON wrapper overhead (only content bytes are counted)

## License

MIT — see [LICENSE](LICENSE).
