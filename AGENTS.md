# Pion 项目说明

## 网络与代理

本机访问 GitHub 可能超时。执行 `git push` / `git fetch` 等网络命令若超时，先在**同一条命令内**开启终端代理再重试（Bash 工具每次调用都是新 shell，alias 不会生效，必须显式 export）：

```bash
export http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897 && git push origin main
```

代理定义见用户 `~/.zshrc` 的 `proxy` / `unproxy` / `proxy_status` alias（端口 7897）。

## 测试

```bash
uv run pytest -q
```

TUI 为自研内联差分渲染架构（移植自 pi），位于 `pion/tui/`（core/components/app/theme），测试基于 `FakeTerminal`，无需 TTY。
