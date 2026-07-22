"""pion: a minimal, hackable coding agent harness in Python.

Architecture inspired by pi (https://github.com/earendil-works/pi):
  llm     — unified multi-provider LLM streaming API (OpenAI-compatible + Anthropic Messages)
  agent   — agent loop with tool calling and a full event stream
  tools   — the four default tools: read / write / edit / bash
  session — JSONL tree-structured sessions + auto-compaction
  hooks   — extension hook system (the soul of pi: primitives, not features)
"""

__version__ = "0.1.0"
