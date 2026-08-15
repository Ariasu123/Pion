"""Tests for the four default tools (read, write, edit, bash)."""

from __future__ import annotations

import sys

import pytest

from pion.tools import BASH_TOOL, DEFAULT_TOOLS, EDIT_TOOL, READ_TOOL, WRITE_TOOL
from pion.tools.base import ToolCallError, validate_arguments
from pion.tools.bash import MAX_OUTPUT_BYTES, BashArgs
from pion.tools.edit import EditArgs
from pion.tools.read import MAX_LINES, ReadArgs
from pion.tools.write import WriteArgs


def text_of(result) -> str:
    return "".join(c.text for c in result.content if c.type == "text")


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


async def test_read_basic(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\ntwo\nthree\n")
    result = await READ_TOOL.execute("t1", ReadArgs(path=str(f)))
    assert text_of(result) == "1\tone\n2\ttwo\n3\tthree\n4\t"
    assert result.details == {"truncated": False, "linesReturned": 4, "totalLines": 4}


async def test_read_offset_and_limit(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("".join(f"line{i}\n" for i in range(1, 11)))
    result = await READ_TOOL.execute("t2", ReadArgs(path=str(f), offset=5, limit=2))
    text = text_of(result)
    assert text.startswith("5\tline5\n6\tline6")
    assert result.details["linesReturned"] == 2
    assert result.details["totalLines"] == 11  # trailing newline -> empty last line
    # limit stopped early while the file has more content: continuation note, not truncation
    assert result.details["truncated"] is False
    assert "offset=7" in text


async def test_read_negative_offset(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("a\nb\nc\nd")
    result = await READ_TOOL.execute("t3", ReadArgs(path=str(f), offset=-2))
    assert text_of(result) == "3\tc\n4\td"


async def test_read_line_cap_truncation(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("".join(f"line{i}\n" for i in range(1, MAX_LINES + 501)))
    result = await READ_TOOL.execute("t4", ReadArgs(path=str(f)))
    assert result.details["truncated"] is True
    assert result.details["linesReturned"] == MAX_LINES
    assert "Use offset=1001" in text_of(result)


async def test_read_offset_beyond_eof(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("x\ny\n")
    result = await READ_TOOL.execute("t5", ReadArgs(path=str(f), offset=100))
    assert "beyond end of file" in text_of(result)
    assert result.details["linesReturned"] == 0


async def test_read_offset_zero_is_an_error(tmp_path):
    f = tmp_path / "zero.txt"
    f.write_text("a\nb\nc\nd")
    result = await READ_TOOL.execute("t7", ReadArgs(path=str(f), offset=0))
    text = text_of(result)
    assert "1-indexed" in text
    assert result.details["linesReturned"] == 0


async def test_read_limit_zero_has_no_bogus_note(tmp_path):
    f = tmp_path / "lim.txt"
    f.write_text("a\nb\nc\nd")
    result = await READ_TOOL.execute("t8", ReadArgs(path=str(f), limit=0))
    text = text_of(result)
    assert result.details["linesReturned"] == 0
    assert "1-0" not in text


async def test_read_not_found(tmp_path):
    result = await READ_TOOL.execute("t6", ReadArgs(path=str(tmp_path / "nope.txt")))
    assert "not found" in text_of(result)


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


async def test_write_creates_parents_and_overwrites(tmp_path):
    f = tmp_path / "nested" / "deep" / "w.txt"
    result = await WRITE_TOOL.execute("t7", WriteArgs(path=str(f), content="hello"))
    assert f.read_text() == "hello"
    assert result.details == {"bytes": 5}

    result2 = await WRITE_TOOL.execute("t8", WriteArgs(path=str(f), content="héllo"))
    assert f.read_text() == "héllo"
    assert result2.details == {"bytes": len("héllo".encode("utf-8"))}


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


async def test_edit_success(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("foo bar baz")
    result = await EDIT_TOOL.execute("t9", EditArgs(path=str(f), old_string="bar", new_string="qux"))
    assert f.read_text() == "foo qux baz"
    assert result.details == {"replacements": 1}


async def test_edit_file_not_found(tmp_path):
    result = await EDIT_TOOL.execute(
        "t10", EditArgs(path=str(tmp_path / "nope.txt"), old_string="a", new_string="b")
    )
    assert "not found" in text_of(result)


async def test_edit_old_string_not_found(tmp_path):
    f = tmp_path / "e2.txt"
    f.write_text("hello world")
    result = await EDIT_TOOL.execute("t11", EditArgs(path=str(f), old_string="xyz", new_string="b"))
    assert "not found" in text_of(result)
    assert f.read_text() == "hello world"


async def test_edit_ambiguous_reports_count(tmp_path):
    f = tmp_path / "e3.txt"
    f.write_text("dup dup dup")
    result = await EDIT_TOOL.execute("t12", EditArgs(path=str(f), old_string="dup", new_string="x"))
    text = text_of(result)
    assert "3 times" in text
    assert f.read_text() == "dup dup dup"  # unchanged


async def test_edit_replace_all(tmp_path):
    f = tmp_path / "e4.txt"
    f.write_text("dup dup dup")
    result = await EDIT_TOOL.execute(
        "t13", EditArgs(path=str(f), old_string="dup", new_string="x", replace_all=True)
    )
    assert f.read_text() == "x x x"
    assert result.details == {"replacements": 3}


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


async def test_bash_echo(tmp_path):
    result = await BASH_TOOL.execute("t14", BashArgs(command="echo hello"))
    assert "hello" in text_of(result)
    assert result.details == {"exitCode": 0, "truncated": False}


async def test_bash_exit_code_propagates():
    result = await BASH_TOOL.execute("t15", BashArgs(command="echo oops; exit 3"))
    assert result.details["exitCode"] == 3
    text = text_of(result)
    assert "oops" in text
    assert "exit code 3" in text


async def test_bash_stderr_is_captured():
    result = await BASH_TOOL.execute("t16", BashArgs(command="echo err >&2"))
    assert "err" in text_of(result)
    assert result.details["exitCode"] == 0


async def test_bash_timeout_kills_process():
    result = await BASH_TOOL.execute(
        "t17", BashArgs(command="echo earlybird; sleep 30; echo latebird", timeout_s=1)
    )
    assert result.details["exitCode"] is None
    text = text_of(result)
    assert "timed out" in text
    assert "earlybird" in text  # partial output preserved
    assert "latebird" not in text


async def test_bash_output_cap_keeps_tail():
    # Print well over the 100KB cap.
    n = MAX_OUTPUT_BYTES * 2
    cmd = f'"{sys.executable}" -c "import sys; sys.stdout.write(\'x\' * {n}); print(\'TAILMARKER\')"'
    result = await BASH_TOOL.execute("t18", BashArgs(command=cmd, timeout_s=60))
    assert result.details["truncated"] is True
    assert result.details["exitCode"] == 0
    text = text_of(result)
    assert "output truncated" in text
    assert "TAILMARKER" in text
    # Captured payload (minus the truncation note) stays near the cap.
    assert len(text) < MAX_OUTPUT_BYTES + 4096


async def test_bash_on_update_streams():
    updates = []

    def on_update(partial):
        updates.append("".join(c.text for c in partial.content if c.type == "text"))

    result = await BASH_TOOL.execute(
        "t19", BashArgs(command="echo stream-test", timeout_s=30), on_update=on_update
    )
    assert result.details["exitCode"] == 0
    assert updates, "expected at least one partial update"
    assert any("stream-test" in u for u in updates)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------


def test_default_tools_and_schema():
    assert [t.name for t in DEFAULT_TOOLS] == ["read", "write", "edit", "bash"]
    for tool in DEFAULT_TOOLS:
        schema = tool.parameters
        assert schema["type"] == "object"
        assert "path" in schema["properties"] or "command" in schema["properties"]
    assert READ_TOOL.execution_mode == "parallel"
    assert WRITE_TOOL.execution_mode == "sequential"
    assert EDIT_TOOL.execution_mode == "sequential"
    assert BASH_TOOL.execution_mode == "sequential"


def test_validate_arguments_raises_tool_call_error():
    with pytest.raises(ToolCallError):
        validate_arguments(READ_TOOL, {"path": 123, "offset": "nope"})
    args = validate_arguments(READ_TOOL, {"path": "x"})
    assert args.offset == 1 and args.limit is None
