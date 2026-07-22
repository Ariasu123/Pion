# Pion

**极简、可扩展的 Python coding agent harness**——从零重写了 TypeScript 项目 [pi](https://github.com/earendil-works/pi)（作者 Mario Zechner）的核心架构，面向 Python 生态。

[English README](README.md)

> primitives, not features——内核极小，一切能力靠扩展。

## 为什么是 pion

pi 证明了 coding agent 不需要臃肿：系统提示词不到 1000 token、默认只有四个工具（`read`/`write`/`edit`/`bash`）、强大的扩展系统。pion 将这套哲学移植到 Python（核心约 3.5k 行）：

- **两种 API 形状接入所有模型**——OpenAI 兼容接口（DeepSeek、Kimi/月之暗面、通义、智谱、OpenAI、vLLM/Ollama…）与 Anthropic Messages（Claude），统一为一套流式事件协议。
- **真正的 agent loop**——流式回合、串/并行工具执行、steering 与 follow-up 队列、截断工具调用保护、优雅中断。
- **树状会话**——append-only JSONL、支持分支，外加 auto-compaction（上下文将满时结构化摘要压缩）。
- **钩子而非功能**——往 `~/.pion/extensions/` 丢一个 `.py` 文件即可拦截工具调用、改写上下文、注册工具/命令，支持热加载。

## 安装

需要 Python ≥ 3.11。

```bash
# 从 GitHub 安装
pip install git+https://github.com/your-name/pion.git

# 或用 uv
uv tool install git+https://github.com/your-name/pion.git

# 源码安装（开发）
git clone https://github.com/your-name/pion.git
cd pion && uv sync --extra dev
```

## 快速上手

```bash
# 1. 配置 provider key（以 DeepSeek 为例，其他见下表）
export DEEPSEEK_API_KEY=sk-...

# 2. 启动 agent
pion                      # 交互式 REPL
pion -p "创建 hello.txt 写入 hello world，然后 cat 出来"   # 单发模式
pion -m kimi-k2-0905-preview      # 切换模型
pion --session my.jsonl           # 恢复会话
```

REPL 斜杠命令：`/help` `/model <id>` `/compact` `/stats` `/exit`。运行中 Ctrl-C 优雅中断当前任务。

没有 key？可以先跑完全离线的端到端 demo（脚本化 provider + 真实 agent 与工具）：

```bash
uv run python demos/mock_e2e.py
```

## 内置模型

| 模型 id | 厂商 | API | 环境变量 |
|---|---|---|---|
| `deepseek-chat` / `deepseek-reasoner` | DeepSeek | openai-completions | `DEEPSEEK_API_KEY` |
| `kimi-k2-0905-preview` | 月之暗面 | openai-completions | `MOONSHOT_API_KEY` |
| `glm-4.6` | 智谱 | openai-completions | `ZHIPU_API_KEY` |
| `qwen3-max` | 阿里 | openai-completions | `DASHSCOPE_API_KEY` |
| `claude-sonnet-4-5` / `claude-opus-4-1` | Anthropic | anthropic-messages | `ANTHROPIC_API_KEY` |

任意 OpenAI 兼容/自托管端点都可用 `--base-url` + `--api-key` 接入，或在代码里 `register_model()`。`<PROVIDER>_BASE_URL` 环境变量可覆盖端点。

## 扩展

在 `~/.pion/extensions/my_ext.py`（或项目内 `.pion/extensions/`）新建文件：

```python
def setup(api):
    # 拦截危险命令
    @api.on("tool_call")
    def guard(event):
        if event.tool_name == "bash" and "rm -rf /" in str(event.args):
            return {"block": True, "reason": "dangerous command blocked"}

    # 每次 LLM 调用前改写上下文（RAG、记忆……）
    @api.on("context")
    def inject(messages):
        return messages  # 或返回改写后的列表
```

事件：`before_agent_start`、`context`、`tool_call`（可阻断）、`tool_result`（可改写）、`agent_start`、`agent_end`、`session_before_compact`。扩展还可以 `api.register_tool(...)` 和 `api.register_command(...)`。

## 架构对照

| pion 模块 | pi 对应物 | 职责 |
|---|---|---|
| `pion/llm/` | `packages/ai` | 统一流式 LLM API，两种 provider 形状，usage/成本 |
| `pion/agent/` | `packages/agent` | agent loop、事件流、工具编排 |
| `pion/tools/` | `packages/coding-agent` 工具 | read / write / edit / bash |
| `pion/session/` | session-manager + compaction | JSONL 会话树、auto-compaction |
| `pion/hooks.py` | 扩展系统 | 生命周期钩子、动态工具/命令、热加载 |
| `pion/cli.py` | `packages/coding-agent` CLI | REPL、斜杠命令、会话 |

刻意**未移植**：pi 的自研 TUI（差分渲染）、主题、键位绑定——pion 只保留基于 rich 的简洁 REPL。

## 开发

```bash
uv sync --extra dev
uv run pytest -q                 # 95+ 测试，无需联网
uv run python demos/mock_e2e.py  # 离线 e2e
uv run python demos/real_e2e.py  # 真实 DeepSeek e2e（需 DEEPSEEK_API_KEY；无 key 时如实跳过，退出码 2）
```

## 路线图

- [ ] 长期记忆 / RAG 扩展（基于 `context` 与 `session_before_compact` 钩子）
- [ ] MCP client 扩展 + 多 agent 编排
- [ ] agent 轨迹导出 + LLM-as-a-Judge 评测

## 致谢与协议

架构与设计哲学致谢 [pi](https://github.com/earendil-works/pi)（Mario Zechner 与 Earendil Inc.）。pion 是独立的 Python 重实现。MIT License。
