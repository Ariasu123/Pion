<p align="center">
  <img src="./docs/assets/pion-cover.png" alt="Pion 封面" width="480">
</p>

<p align="center">
  <a href="./README.md">English</a> | <strong>中文</strong>
</p>

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Sandbox-2496ED?logo=docker&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-69B34C)](./LICENSE)

Pion 是一个受开源项目 pi agent 启发的轻量、可扩展 Python 编码智能体项目；当前为 agent 循环、LLM 提供商、工具、会话和钩子提供基础能力，后续将在此基础上持续进行实验与扩展

> **Alpha 阶段：** Pion 正在积极开发中，稳定版本发布前，接口和配置仍可能变化。

- **小而清晰的核心** —— 流式 Agent 循环、并行工具执行和生命周期钩子。
- **原生终端工作流** —— 内联 TUI 将完整对话保留在终端 scrollback 中。
- **开放的扩展能力** —— 添加 Python 工具、钩子、斜杠命令或 stdio MCP 服务。
- **可选隔离执行** —— 通过 MCP 将文件和 shell 工具放入一次性 Docker 沙盒。

## 快速开始

一键安装器支持 macOS 和 Linux。缺少 [uv](https://docs.astral.sh/uv/) 时会自动安装，uv 随后会提供兼容的 Python 运行时。Docker 是可选依赖，仅在沙盒执行时需要。

```bash
curl -LsSf https://raw.githubusercontent.com/Ariasu123/Pion/main/install.sh | sh
```

安装完成后重启终端，然后配置并启动 Pion：

```bash
pion --configure
pion
```

`--configure` 会将模型配置保存到 `~/.pion/config.json`。你也可以通过环境变量提供内置模型服务的密钥，例如 `DEEPSEEK_API_KEY` 或 `ANTHROPIC_API_KEY`。

在任意项目目录运行 `pion`，该目录就会成为 Agent 工作区。重新执行安装命令可升级到最新稳定版；使用 `uv tool uninstall pion` 可卸载。需要固定版本时，将安装器传给 `PION_VERSION=v0.1.0 sh`。

> **安全提示：** 默认情况下，Pion 的 `bash`、`read`、`write` 和 `edit` 工具直接在宿主机运行。需要项目级 Docker 隔离时，请使用 `--sandbox mcp`。

<details>
<summary><strong>从源码进行开发安装</strong></summary>

```bash
git clone https://github.com/Ariasu123/Pion.git
cd Pion
uv sync --group dev
uv tool install --editable . --force
```

Editable 安装会让全局 `pion` 命令直接引用这个源码目录，因此修改代码后无需重新安装即可生效。移动或删除该目录后，需要重新安装才能继续使用命令。

</details>

## 架构

```text
CLI / TUI
    ↓
Controller → 会话树 / 上下文压缩
    ↓
Agent Loop → LLM Provider
    ↓
Tools → Host Runtime 或 MCP → Docker Sandbox
```

主要扩展点包括：

- **Providers：** OpenAI-compatible Chat Completions 和 Anthropic Messages 适配器。
- **Tools：** 参数经过类型校验、支持流式更新的 Python 工具。
- **Hooks：** 可修改上下文、工具调用、工具结果和自定义命令的生命周期中间件。
- **Sessions：** 支持分支、标签、摘要和压缩的 append-only JSONL 历史。

<details>
<summary><strong>终端界面快捷键</strong></summary>

| 按键 | 操作 |
| --- | --- |
| `Enter` | 发送；运行中则排队为下一条消息 |
| `Alt+Enter` / `Alt+Up` | 排队 follow-up / 取回最近的排队消息 |
| `Ctrl+J` 或 `Shift+Enter` | 插入换行 |
| `Esc` | 中断当前轮次或关闭选择器 |
| `Ctrl+O` / `Ctrl+T` | 切换工具输出 / thinking 内容 |
| `Ctrl+L` / `Ctrl+P` / `Ctrl+B` | 打开模型、命令或会话树选择器 |
| `Ctrl+Q` | 退出 |

输入 `/` 可补全斜杠命令，输入 `@` 可补全文件。内置命令包括 `/help`、`/model`、`/compact`、`/stats`、`/tree`、`/theme` 和 `/exit`。

</details>

<details>
<summary><strong>MCP 服务</strong></summary>

在 `~/.pion/config.json` 中添加受信任的 stdio 服务；发现的工具会以 `<服务名>__<工具名>` 暴露。

```json
{
  "version": 1,
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/项目的绝对路径"],
      "env": {},
      "enabled": true,
      "timeout_seconds": 30
    }
  }
}
```

Pion 目前只支持基于 stdio 的 MCP tools，尚不支持 resources、prompts 和 Streamable HTTP。外部 MCP 服务是受信任的宿主进程，不受 Pion Docker 沙盒隔离。

</details>

<details>
<summary><strong>Docker 沙盒</strong></summary>

`uv run pion --sandbox mcp` 会在一次性非 root 容器中启动 Pion 内置 MCP 服务。容器只绑定挂载当前项目；Git 元数据默认只读，`.env` 等受保护文件会被遮蔽，宿主环境变量和 Docker socket 不会注入。

常用选项包括 `--sandbox-image IMAGE`、`--sandbox-network bridge|none`、`--sandbox-git-write` 和 `--allow-project-extensions`。默认 bridge 网络允许出站访问；处理不可信仓库时请使用 `--sandbox-network none`。项目 extension 在宿主机执行，因此沙盒模式下默认禁用，除非显式允许。

其他 MCP Client 也可以挂载该沙盒服务：

```json
{"mcpServers": {"pion-sandbox": {"command": "uv", "args": ["run", "pion", "mcp"]}}}
```

</details>

Pion 使用 [MIT License](./LICENSE)，架构和交互方式受到 [pi](https://github.com/earendil-works/pi) 启发。
