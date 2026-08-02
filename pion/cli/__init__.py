"""pion command line interface.

`pion` starts the inline TUI (terminal-native scrollback, differential
rendering), `pion --ui plain` starts the legacy REPL, and `pion --print "..."`
runs a single prompt and exits. Sessions persist as JSONL under
~/.pion/sessions (or a file passed via --session), with automatic context
compaction when the conversation approaches the model's context window.
"""

from .. import __version__
from ..config import default_config_path
from ..mcp import MCPClientManager
from ..sandbox import build_runtime, check_docker_available
from ._shared import console, err_console
from .app import app, main, mcp_command
from .bootstrap import (
    _async_main,
    build_system_prompt,
    default_session_path,
    extension_dirs,
    find_first_kept_entry_id,
    resolve_sandbox_settings,
)
from .plain import (
    Repl,
    StreamRenderer,
    api_key_env_name,
    parse_slash_command,
    summarize_tool_args,
    summarize_tool_result,
)
from .profiles import (
    DEFAULT_MODEL_ID,
    _load_user_config,
    _version_callback,
    configure_profile,
    is_interactive,
    resolve_startup,
    resolve_ui_mode,
    select_profile,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "MCPClientManager",
    "Repl",
    "StreamRenderer",
    "__version__",
    "_async_main",
    "_load_user_config",
    "_version_callback",
    "api_key_env_name",
    "app",
    "build_runtime",
    "build_system_prompt",
    "check_docker_available",
    "configure_profile",
    "console",
    "default_config_path",
    "default_session_path",
    "err_console",
    "extension_dirs",
    "find_first_kept_entry_id",
    "is_interactive",
    "main",
    "mcp_command",
    "parse_slash_command",
    "resolve_sandbox_settings",
    "resolve_startup",
    "resolve_ui_mode",
    "select_profile",
    "summarize_tool_args",
    "summarize_tool_result",
]
