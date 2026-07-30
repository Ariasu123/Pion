<p align="right">
  <strong>English</strong> · <a href="./README.zh-CN.md">简体中文</a>
</p>

# Pion

Pion is a lightweight, extensible Python coding-agent project inspired by the open-source pi agent; it currently provides a foundation for agent loops, LLM providers, tools, sessions, and hooks, and will evolve through further experimentation and extensions.

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
