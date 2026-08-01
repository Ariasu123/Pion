<p align="center">
  <img src="./docs/assets/pion-cover.png" alt="Pion cover" width="480">
</p>

---

<p align="center">
  <strong>English</strong> | <a href="./README.zh-CN.md">中文</a>
</p>

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Sandbox-2496ED?logo=docker&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-69B34C)](./LICENSE)

Pion is a lightweight, extensible Python coding-agent project inspired by the open-source pi agent; it currently provides a foundation for agent loops, LLM providers, tools, sessions, and hooks, and will evolve through further experimentation and extensions.

## Terminal UI

Running `pion` in an interactive terminal now opens the full-screen Textual
interface. The existing line-oriented REPL remains available for low-capability
terminals and troubleshooting:

```text
pion                         # full-screen TUI (default)
pion --ui tui                # explicitly select the TUI
pion --ui plain              # legacy line-oriented REPL
pion --print "your prompt"   # one prompt, then exit; unchanged
pion --session path.jsonl    # resume a session in the TUI
```

An interactive launch requires a TTY. In a pipe, CI job, or other non-TTY
environment, use `--print`; Pion will not attempt to draw a full-screen UI.
Terminals reporting `TERM=dumb` fall back to the plain interface.

The conversation pane renders streaming Markdown and keeps tool calls in
collapsed cards. A card shows its status, elapsed time, argument summary,
output or error, and the originating MCP server when applicable. Scrolling up
pauses automatic follow; returning to the bottom resumes it.

Keyboard shortcuts:

| Key | Action |
| --- | --- |
| `Enter` | Send the editor contents |
| `Ctrl+J` | Insert a newline |
| `Esc` | Abort the current turn or branch summary |
| `Ctrl+P` | Open the command menu |
| `Ctrl+B` | Toggle or focus the session tree |
| `Ctrl+Q` | Exit Pion |
| `Ctrl+O` | Cycle tree filters while the tree is focused |
| `Shift+L` | Set or clear a label on the selected tree entry |

The session tree is visible on terminals at least 100 columns wide and becomes
an overlay on narrower terminals. Its filters are `default`, `no-tools`,
`user-only`, `labeled-only`, and `all`. Selecting a user entry moves to its
parent and restores the old prompt in the editor for revision; selecting an
assistant, tool result, compaction, or branch-summary entry moves directly to
that point. When this abandons the active suffix, Pion offers no summary, a
default summary, or a summary with custom focus. Summary generation is
transactional: a failure or abort leaves the active branch and JSONL file
unchanged.

The TUI supports `/help`, `/model`, `/compact`, `/stats`, `/tree`, `/exit`, and
commands registered by extensions. Model selection is available in v1;
sandbox, MCP, and profile configuration remain display-only and are edited
through the CLI or `~/.pion/config.json`.

## MCP servers

Pion includes a built-in [Model Context Protocol](https://modelcontextprotocol.io/)
client for stdio servers. Enabled servers are started with Pion, their tools
are discovered automatically, and each tool is exposed to the model as
`<server-name>__<tool-name>`. A server that fails to start is reported and
disabled for that run without preventing Pion or other servers from starting.

Add servers to `~/.pion/config.json`:

```json
{
  "version": 1,
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/absolute/path/to/project"
      ],
      "env": {},
      "enabled": true,
      "timeout_seconds": 30
    }
  },
  "active_profile": null,
  "profiles": {}
}
```

`env` overrides variables only for that child process; other host environment
variables are inherited. Environment values are redacted from Pion's MCP
startup and shutdown errors. The timeout applies to connection setup,
discovery, and tool calls.

MCP stdio servers are trusted host processes. They are **not** run inside
Pion's Docker sandbox and may access resources available to the Pion process.
Only configure servers and commands you trust. This first release supports MCP
tools over stdio; MCP resources, prompts, and Streamable HTTP are not yet
exposed.

## Docker sandbox

Pion runs its default shell and file tools with a Docker sandbox. Docker is
fail-closed: if the CLI, daemon, image build, or container startup is
unavailable, Pion exits before sending a request to the model. Use
`--sandbox off` only when unrestricted host execution is intentional.

The sandbox keeps Pion itself, model credentials, configuration, and sessions
on the host. It creates one disposable, non-root container per Pion process and
bind-mounts only the current project at the same absolute path. Project changes
are therefore visible immediately on the host. Git metadata is read-only by
default, `.env` files are denied to file tools and masked in the container, and
host environment variables and the Docker socket are not injected.

Useful options:

```text
--sandbox docker|off
--sandbox-image IMAGE
--sandbox-network bridge|none
--sandbox-git-write
--allow-project-extensions
```

The default `bridge` network is convenient but does not prevent source
exfiltration. Use `--sandbox-network none` for untrusted repositories. Project
extensions are disabled while sandboxed unless explicitly enabled because
their Python code executes in the host Pion process and can bypass the sandbox.
User-level extensions under `~/.pion/extensions` are treated as trusted host
code.

Sandbox defaults can be stored in the version 1 configuration:

```json
{
  "version": 1,
  "sandbox": {
    "backend": "docker",
    "image": null,
    "network": "bridge",
    "memory_mb": 4096,
    "cpus": 2.0,
    "pids_limit": 256,
    "git_write": false,
    "protect_paths": [".env", ".env.*"]
  },
  "active_profile": null,
  "profiles": {}
}
```

A custom image must already be available to Docker and provide `sleep` plus the
tools needed by the agent. The built-in image contains Python 3.12, uv, Bash,
Git, ripgrep, curl, certificates, and basic compilation tools.

This v1 sandbox is intended for a single-user local coding agent. It protects
host resources outside the project, but it is not a multi-tenant isolation
boundary; cloud deployments should use a VM-, gVisor-, Kata-, or
Firecracker-backed `SandboxRuntime`.
