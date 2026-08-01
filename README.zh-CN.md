<p align="center">
  <img src="./docs/assets/pion-cover.png" alt="Pion 封面" width="480">
</p>

---

<p align="center">
  <a href="./README.md">English</a> | <strong>中文</strong>
</p>

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Sandbox-2496ED?logo=docker&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-69B34C)](./LICENSE)

Pion 是一个受开源项目 pi agent 启发的轻量、可扩展 Python 编码智能体项目；当前为 agent 循环、LLM 提供商、工具、会话和钩子提供基础能力，后续将在此基础上持续进行实验与扩展。

## MCP 服务

Pion 内置了基于 stdio 的 [Model Context Protocol](https://modelcontextprotocol.io/)
客户端。启用的服务会随 Pion 启动，工具会被自动发现，并以
`<服务名>__<工具名>` 的名称提供给模型。某个服务启动失败时，Pion 会报告并在本次
运行中禁用它，但不会阻止 Pion 或其他 MCP 服务继续启动。

在 `~/.pion/config.json` 中添加服务：

```json
{
  "version": 1,
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/项目的绝对路径"
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

`env` 只覆盖该子进程中的同名变量，其余宿主环境变量会被继承。Pion 在 MCP
启动和关闭错误中会隐藏已配置的环境变量值。超时时间同时用于连接、工具发现和工具
调用。

stdio MCP 服务是受信任的宿主进程，**不会**在 Pion 的 Docker 沙盒中运行，能够访问
Pion 进程可访问的宿主资源。请只配置你信任的服务和命令。首版仅支持通过 stdio
使用 MCP tools，暂不提供 resources、prompts 和 Streamable HTTP。

## Docker 沙盒

Pion 默认使用 Docker 沙盒运行 shell 和文件工具。Docker 采用失败关闭策略：
如果 Docker CLI、daemon、镜像构建或容器启动不可用，Pion 会在向模型发送请求前退出。
只有明确需要不受限制的宿主执行时，才应使用 `--sandbox off`。

Pion 进程本身、模型凭据、配置和会话保留在宿主机上。每个 Pion 进程会创建一个
一次性的非 root 容器，并仅将当前项目以相同的绝对路径绑定挂载到容器中。因此，
项目修改会实时反映到宿主机。Git 元数据默认只读；文件工具无法访问 `.env` 文件，
容器内对应路径也会被遮蔽；宿主环境变量和 Docker socket 不会注入容器。

常用选项：

```text
--sandbox docker|off
--sandbox-image IMAGE
--sandbox-network bridge|none
--sandbox-git-write
--allow-project-extensions
```

默认的 `bridge` 网络便于日常开发，但不能阻止源码外泄。处理不可信仓库时，请使用
`--sandbox-network none`。启用沙盒后，项目级 extension 默认不会加载；只有显式允许
后才会加载，因为其中的 Python 代码在宿主 Pion 进程中执行，可以绕过沙盒。
`~/.pion/extensions` 下的用户级 extension 被视为可信宿主代码。

可以在 version 1 配置中保存沙盒默认值：

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

自定义镜像必须已经存在于 Docker 本地，并提供 `sleep` 以及 Agent 所需的工具。
内置镜像包含 Python 3.12、uv、Bash、Git、ripgrep、curl、CA 证书和基础编译工具。

v1 沙盒面向单用户本地编码 Agent。它能够保护项目之外的宿主资源，但不是多租户
隔离边界；云端部署应通过 `SandboxRuntime` 接入基于 VM、gVisor、Kata 或
Firecracker 的运行后端。
