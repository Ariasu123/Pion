# Pion

**A minimal, hackable coding agent harness in Python** — a from-scratch reimplementation of the core architecture of [pi](https://github.com/earendil-works/pi) (TypeScript, by Mario Zechner), rebuilt for the Python ecosystem.

[中文文档](README.zh-CN.md)

> primitives, not features — the kernel stays tiny; everything else is an extension.

## Why pion

pi proved that a coding agent needs no bloat: a <1000-token system prompt, four tools (`read`/`write`/`edit`/`bash`), and a powerful extension system. pion ports that philosophy to Python (~3.5k lines of core):

- **Two API shapes, every model** — OpenAI-compatible endpoints (DeepSeek, Kimi/Moonshot, Qwen, Zhipu, OpenAI, vLLM/Ollama…) and Anthropic Messages (Claude), unified behind one streaming event protocol.
- **A real agent loop** — streamed turns, parallel/sequential tool execution, steering & follow-up queues, truncated-tool-call protection, graceful abort.
- **Tree-structured sessions** — append-only JSONL with branches, plus auto-compaction (structured summarization when the context fills up).
- **Hooks, not features** — drop a `.py` file into `~/.pion/extensions/` to intercept tool calls, rewrite context, register tools/commands. Hot-reloadable.

## Install

Requires Python ≥ 3.11.

```bash
# from GitHub
pip install git+https://github.com/your-name/pion.git

# or with uv
uv tool install git+https://github.com/your-name/pion.git

# from source (development)
git clone https://github.com/your-name/pion.git
cd pion && uv sync --extra dev
```

## Quickstart

```bash
# 1. set a provider key (DeepSeek shown; see table below for others)
export DEEPSEEK_API_KEY=sk-...

# 2. start the agent
pion                      # interactive REPL
pion -p "create hello.txt with hello world, then cat it"   # single-shot
pion -m kimi-k2-0905-preview      # pick another model
pion --session my.jsonl           # resume a session
```

REPL slash commands: `/help` `/model <id>` `/compact` `/stats` `/exit`. Ctrl-C aborts the current run gracefully.

No key? Run the fully offline end-to-end demo (scripted provider, real agent + tools):

```bash
uv run python demos/mock_e2e.py
```

## Built-in models

| Model id | Provider | API | Env var |
|---|---|---|---|
| `deepseek-chat` / `deepseek-reasoner` | DeepSeek | openai-completions | `DEEPSEEK_API_KEY` |
| `kimi-k2-0905-preview` | Moonshot | openai-completions | `MOONSHOT_API_KEY` |
| `glm-4.6` | Zhipu | openai-completions | `ZHIPU_API_KEY` |
| `qwen3-max` | Alibaba | openai-completions | `DASHSCOPE_API_KEY` |
| `claude-sonnet-4-5` / `claude-opus-4-1` | Anthropic | anthropic-messages | `ANTHROPIC_API_KEY` |

Any OpenAI-compatible/self-hosted endpoint works via `--base-url` + `--api-key`, or `register_model()` in code. `<PROVIDER>_BASE_URL` env vars override endpoints.

## Extensions

Drop a file at `~/.pion/extensions/my_ext.py` (or `.pion/extensions/` in a project):

```python
def setup(api):
    # block dangerous commands
    @api.on("tool_call")
    def guard(event):
        if event.tool_name == "bash" and "rm -rf /" in str(event.args):
            return {"block": True, "reason": "dangerous command blocked"}

    # inject context before every LLM call (RAG, memory, …)
    @api.on("context")
    def inject(messages):
        return messages  # or a transformed copy
```

Events: `before_agent_start`, `context`, `tool_call` (blockable), `tool_result` (overridable), `agent_start`, `agent_end`, `session_before_compact`. Extensions can also `api.register_tool(...)` and `api.register_command(...)`.

## Architecture

| pion module | pi counterpart | what it does |
|---|---|---|
| `pion/llm/` | `packages/ai` | unified streaming LLM API, 2 provider shapes, usage/cost |
| `pion/agent/` | `packages/agent` | agent loop, event stream, tool orchestration |
| `pion/tools/` | `packages/coding-agent` tools | read / write / edit / bash |
| `pion/session/` | session-manager + compaction | JSONL session tree, auto-compaction |
| `pion/hooks.py` | extension system | lifecycle hooks, dynamic tools/commands, hot reload |
| `pion/cli.py` | `packages/coding-agent` CLI | REPL, slash commands, sessions |

Intentionally **not** ported: pi's custom TUI (differential rendering), themes, keybindings — pion keeps a simple rich-based REPL.

## Development

```bash
uv sync --extra dev
uv run pytest -q                 # 95+ tests, no network needed
uv run python demos/mock_e2e.py  # offline e2e
uv run python demos/real_e2e.py  # live DeepSeek e2e (needs DEEPSEEK_API_KEY; exits 2 = honest skip without it)
```

## Roadmap

- [ ] Long-term memory / RAG extension (built on the `context` + `session_before_compact` hooks)
- [ ] MCP client extension + multi-agent orchestration
- [ ] Agent trajectory export & LLM-as-a-Judge evaluation

## Credit & License

Architecture and design philosophy credit goes to [pi](https://github.com/earendil-works/pi) by Mario Zechner and Earendil Inc. pion is an independent Python reimplementation. MIT License.
