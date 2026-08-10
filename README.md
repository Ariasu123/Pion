<p align="center">
  <img src="./docs/assets/pion-cover.png" alt="Pion cover" width="480">
</p>

<p align="center">
  <strong>English</strong> | <a href="./README.zh-CN.md">中文</a>
</p>

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Sandbox-2496ED?logo=docker&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-69B34C)](./LICENSE)

Pion is a lightweight, extensible Python coding agent inspired by the open-source pi agent. It provides foundational support for agent loops, LLM providers, tools, sessions, and hooks, and will continue to evolve as a platform for further experimentation and extension.

> **Alpha:** Pion is under active development. Interfaces and configuration may change before a stable release.

- **Small, readable core** — a streamed agent loop with parallel tool execution and hooks.
- **Terminal-native workflow** — an inline TUI that keeps the conversation in your scrollback.
- **Open extension surface** — add Python tools, lifecycle hooks, slash commands, or stdio MCP servers.
- **Optional isolation** — run file and shell tools through the standalone [sandbox-docker-mcp](https://github.com/Ariasu123/Agent-Toolkit/tree/main/Personal/MCP-Hub/sandbox-docker-mcp) package.

## Quick start

The one-line installer supports macOS and Linux. It installs [uv](https://docs.astral.sh/uv/) when needed; uv then provides a compatible Python runtime. Docker is optional and only required for sandboxed execution.

```bash
curl -LsSf https://raw.githubusercontent.com/Ariasu123/Pion/main/install.sh | sh
```

Restart your terminal after installation, then configure and start Pion:

```bash
pion --configure
pion
```

`--configure` saves a model profile in `~/.pion/config.json`. You can also provide a built-in provider key through its environment variable, such as `DEEPSEEK_API_KEY` or `ANTHROPIC_API_KEY`.

Run `pion` from any project directory; that directory becomes the agent workspace. Rerun the installer to upgrade to the latest stable release, or remove Pion with `uv tool uninstall pion`. To install a specific release, pipe the installer into `PION_VERSION=v0.1.0 sh`.

> **Security:** By default, Pion's `bash`, `read`, `write`, and `edit` tools run directly on the host. Use `--sandbox mcp` when you want project-scoped Docker isolation.

<details>
<summary><strong>Development install from source</strong></summary>

```bash
git clone https://github.com/Ariasu123/Pion.git
cd Pion
uv sync --group dev
uv tool install --editable . --force
```

The editable install makes the global `pion` command use this checkout directly, so source changes take effect without reinstalling. Moving or deleting the checkout breaks that command until it is reinstalled.

</details>

## Architecture

```text
CLI / TUI
    ↓
Controller → Session tree / compaction
    ↓
Agent loop → LLM provider
    ↓
Tools → Host runtime or MCP → Docker sandbox
```

The main extension points are:

- **Providers:** OpenAI-compatible Chat Completions and Anthropic Messages adapters.
- **Tools:** typed Python tools with validated arguments and streamed updates.
- **Hooks:** lifecycle middleware for context, tool calls, tool results, and custom commands.
- **Sessions:** append-only JSONL history with branching, labels, summaries, and compaction.

<details>
<summary><strong>Terminal UI shortcuts</strong></summary>

| Key | Action |
| --- | --- |
| `Enter` | Send, or queue the next message while running |
| `Alt+Enter` / `Alt+Up` | Queue a follow-up / restore the last queued message |
| `Ctrl+J` or `Shift+Enter` | Insert a newline |
| `Esc` | Abort the active turn or close a selector |
| `Ctrl+O` / `Ctrl+T` | Toggle tool output / thinking content |
| `Ctrl+L` / `Ctrl+P` / `Ctrl+B` | Open model, command, or session-tree selectors |
| `Ctrl+Q` | Exit |

Type `/` for slash-command completion and `@` for file completion. Built-in commands include `/help`, `/model`, `/compact`, `/stats`, `/tree`, `/theme`, and `/exit`.

</details>

<details>
<summary><strong>MCP servers</strong></summary>

Add trusted stdio servers to `~/.pion/config.json`; discovered tools are exposed as `<server>__<tool>`.

```json
{
  "version": 1,
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/absolute/project/path"],
      "env": {},
      "enabled": true,
      "timeout_seconds": 30
    }
  }
}
```

Pion currently supports MCP tools over stdio, not resources, prompts, or Streamable HTTP. External MCP servers run as trusted host processes and are not isolated by Pion's Docker sandbox.

</details>

<details>
<summary><strong>Docker sandbox</strong></summary>

`uv run pion --sandbox mcp` starts the independently maintained [sandbox-docker-mcp](https://github.com/Ariasu123/Agent-Toolkit/tree/main/Personal/MCP-Hub/sandbox-docker-mcp) server through Pion's compatibility entry point. It runs in a disposable, non-root container. Only the current project is bind-mounted; Git metadata is read-only by default, protected files such as `.env` are masked, and host environment variables and the Docker socket are not injected.

Useful options: `--sandbox-image IMAGE`, `--sandbox-network bridge|none`, `--sandbox-git-write`, and `--allow-project-extensions`. The default bridge network permits outbound access; use `--sandbox-network none` for untrusted repositories. Project extensions execute on the host and are disabled in sandbox mode unless explicitly allowed.

The sandbox server can also be mounted by another MCP client:

```json
{"mcpServers": {"pion-sandbox": {"command": "uv", "args": ["run", "pion", "mcp"]}}}
```

</details>

Pion is licensed under the [MIT License](./LICENSE). Its architecture and interaction model are inspired by [pi](https://github.com/earendil-works/pi).
