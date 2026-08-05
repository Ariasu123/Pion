from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_uv_script() -> str:
    return """#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$PION_TEST_LOG"
if [ "$1" = "tool" ] && [ "$2" = "install" ]; then
    if [ "${PION_TEST_INSTALL_FAIL:-0}" = "1" ]; then
        exit 23
    fi
    mkdir -p "$PION_TEST_TOOL_BIN"
    printf '#!/bin/sh\\nprintf "pion 0.1.0\\n"\\n' > "$PION_TEST_TOOL_BIN/pion"
    chmod +x "$PION_TEST_TOOL_BIN/pion"
elif [ "$1" = "tool" ] && [ "$2" = "dir" ]; then
    printf '%s\\n' "$PION_TEST_TOOL_BIN"
fi
"""


@pytest.fixture
def installer_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    tool_bin = tmp_path / "tool-bin"
    log = tmp_path / "uv.log"

    _write_executable(
        fake_bin / "uname",
        '#!/bin/sh\nprintf "%s\\n" "${PION_TEST_UNAME:-Linux}"\n',
    )
    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
set -eu
case "$*" in
    *releases/latest*) printf '%s' 'https://github.com/Ariasu123/Pion/releases/tag/v0.1.0' ;;
    *astral.sh/uv/install.sh*)
        output=''
        previous=''
        for argument in "$@"; do
            if [ "$previous" = '-o' ]; then output="$argument"; break; fi
            previous="$argument"
        done
        [ -n "$output" ]
        cp "$PION_TEST_UV_INSTALLER" "$output"
        ;;
esac
""",
    )
    fake_uv = tmp_path / "fake-uv"
    _write_executable(fake_uv, _fake_uv_script())
    _write_executable(fake_bin / "uv", _fake_uv_script())

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:/bin:/usr/bin",
            "PION_TEST_FAKE_UV": str(fake_uv),
            "PION_TEST_LOG": str(log),
            "PION_TEST_TOOL_BIN": str(tool_bin),
            "SHELL": "/bin/sh",
        }
    )
    return env, fake_bin


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", str(INSTALLER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_installs_pinned_release(installer_env: tuple[dict[str, str], Path]) -> None:
    env, _ = installer_env
    env["PION_VERSION"] = "v9.8.7"

    result = _run(env)

    assert result.returncode == 0, result.stderr
    log = Path(env["PION_TEST_LOG"]).read_text(encoding="utf-8")
    assert "tool install --force" in log
    assert "/archive/refs/tags/v9.8.7.tar.gz" in log
    assert "tool update-shell" in log
    assert "installed pion 0.1.0" in result.stdout


def test_resolves_latest_release(installer_env: tuple[dict[str, str], Path]) -> None:
    env, _ = installer_env

    result = _run(env)

    assert result.returncode == 0, result.stderr
    log = Path(env["PION_TEST_LOG"]).read_text(encoding="utf-8")
    assert "/archive/refs/tags/v0.1.0.tar.gz" in log
    assert "resolving the latest stable Pion release" in result.stdout


def test_bootstraps_uv_when_missing(installer_env: tuple[dict[str, str], Path]) -> None:
    env, fake_bin = installer_env
    (fake_bin / "uv").unlink()
    uv_installer = Path(env["HOME"]) / "fake-uv-installer.sh"
    uv_installer.parent.mkdir(parents=True)
    _write_executable(
        uv_installer,
        """#!/bin/sh
set -eu
mkdir -p "$HOME/.local/bin"
cp "$PION_TEST_FAKE_UV" "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
""",
    )
    env["PION_TEST_UV_INSTALLER"] = str(uv_installer)
    env["PION_VERSION"] = "v0.1.0"

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert "uv was not found" in result.stdout
    assert (Path(env["HOME"]) / ".local/bin/uv").is_file()


def test_rejects_unsupported_platform(installer_env: tuple[dict[str, str], Path]) -> None:
    env, _ = installer_env
    env["PION_TEST_UNAME"] = "Windows_NT"

    result = _run(env)

    assert result.returncode != 0
    assert "only macOS and Linux are supported" in result.stderr


@pytest.mark.parametrize("version", ["main", "v0.1.0/unsafe"])
def test_rejects_invalid_version(
    installer_env: tuple[dict[str, str], Path], version: str
) -> None:
    env, _ = installer_env
    env["PION_VERSION"] = version

    result = _run(env)

    assert result.returncode != 0
    assert "PION_VERSION" in result.stderr


def test_reports_install_failure(installer_env: tuple[dict[str, str], Path]) -> None:
    env, _ = installer_env
    env["PION_VERSION"] = "v0.1.0"
    env["PION_TEST_INSTALL_FAIL"] = "1"

    result = _run(env)

    assert result.returncode != 0
    assert "failed to install Pion v0.1.0" in result.stderr
