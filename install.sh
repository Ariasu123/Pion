#!/bin/sh

set -eu

REPOSITORY="Ariasu123/Pion"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"

info() {
    printf '%s\n' "pion-installer: $*"
}

fail() {
    printf '%s\n' "pion-installer: error: $*" >&2
    exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required to install Pion"

case "$(uname -s)" in
    Darwin|Linux) ;;
    *) fail "only macOS and Linux are supported by this installer" ;;
esac

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

uv_bin="$(find_uv || true)"
if [ -z "$uv_bin" ]; then
    info "uv was not found; installing it with the official Astral installer"
    installer_file="$(mktemp "${TMPDIR:-/tmp}/pion-uv-installer.XXXXXX")"
    trap 'rm -f "$installer_file"' EXIT HUP INT TERM
    curl -LsSf "$UV_INSTALLER_URL" -o "$installer_file" \
        || fail "failed to download the uv installer"
    UV_INSTALLER_NO_MODIFY_PATH=1 sh "$installer_file" \
        || fail "the uv installer failed"
    rm -f "$installer_file"
    trap - EXIT HUP INT TERM
    uv_bin="$(find_uv || true)"
    [ -n "$uv_bin" ] || fail "uv was installed but its executable could not be found"
fi

version="${PION_VERSION:-}"
if [ -z "$version" ]; then
    info "resolving the latest stable Pion release"
    latest_url="$(
        curl -LsSf -o /dev/null -w '%{url_effective}' \
            "https://github.com/$REPOSITORY/releases/latest"
    )" || fail "failed to resolve the latest Pion release"
    case "$latest_url" in
        */releases/tag/*) version="${latest_url##*/}" ;;
        *) fail "no stable Pion release was found" ;;
    esac
fi

case "$version" in
    v[0-9]*.[0-9]*.[0-9]*) ;;
    *) fail "PION_VERSION must be a release tag such as v0.1.0" ;;
esac
case "$version" in
    *[!A-Za-z0-9._-]*) fail "PION_VERSION contains unsupported characters" ;;
esac

archive_url="https://github.com/$REPOSITORY/archive/refs/tags/$version.tar.gz"
info "installing Pion $version"
"$uv_bin" tool install --force "$archive_url" \
    || fail "failed to install Pion $version"

if ! "$uv_bin" tool update-shell >/dev/null 2>&1; then
    info "could not update your shell PATH automatically; run: $uv_bin tool update-shell"
fi

bin_dir="$("$uv_bin" tool dir --bin)" \
    || fail "could not locate the uv tool executable directory"
pion_bin="$bin_dir/pion"
[ -x "$pion_bin" ] || fail "installation finished but $pion_bin was not created"
installed_version="$("$pion_bin" --version)" \
    || fail "Pion was installed but failed its version check"

info "installed $installed_version at $pion_bin"
info "restart your terminal, then run 'pion --configure' followed by 'pion'"
