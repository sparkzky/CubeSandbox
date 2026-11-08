#!/usr/bin/env bash
# Copyright (c) 2024 Tencent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Install cube-gpu-daemon and its systemd unit on a GPU node.
#
# Usage: sudo ./install-daemon.sh [-n|--dry-run] [path-to-binary]
#
#   -n, --dry-run     print what would be done, change nothing
#   binary            pre-built cube-gpu-daemon; defaults to
#                     ../host/target/release/cube-gpu-daemon relative to this
#                     script (build it with: cd gpu/host && cargo build --release)

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly GPU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly UNIT_SRC="${GPU_DIR}/systemd/cube-gpu-daemon.service"
readonly UNIT_DST="/etc/systemd/system/cube-gpu-daemon.service"
readonly BIN_DST="/usr/local/bin/cube-gpu-daemon"
readonly SHM_DIR="/var/run/cube-gpu/shm"

DRY_RUN=0
BIN_SRC="${GPU_DIR}/host/target/release/cube-gpu-daemon"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) BIN_SRC="$1"; shift ;;
    esac
done

log() { printf '[install-daemon] %s\n' "$*"; }
warn() { printf '[install-daemon] WARNING: %s\n' "$*" >&2; }
run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[install-daemon] (dry-run) %s\n' "$*"
    else
        "$@"
    fi
}

require_root() {
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "[install-daemon] ERROR: must run as root (sudo ./install-daemon.sh)" >&2
        exit 1
    fi
}

require_root

if [[ ! -f "${BIN_SRC}" ]]; then
    echo "[install-daemon] ERROR: daemon binary not found: ${BIN_SRC}" >&2
    echo "[install-daemon] build it first: (cd ${GPU_DIR}/host && cargo build --release)" >&2
    exit 1
fi
if [[ ! -x "${BIN_SRC}" ]]; then
    echo "[install-daemon] ERROR: ${BIN_SRC} is not executable" >&2
    exit 1
fi
if [[ ! -f "${UNIT_SRC}" ]]; then
    echo "[install-daemon] ERROR: systemd unit not found: ${UNIT_SRC}" >&2
    exit 1
fi
if [[ "${DRY_RUN}" -eq 0 ]] && ! command -v systemctl >/dev/null 2>&1; then
    echo "[install-daemon] ERROR: systemctl not found; this installer targets systemd hosts" >&2
    exit 1
fi

log "installing ${BIN_SRC} -> ${BIN_DST}"
run install -m 0755 -o root -g root "${BIN_SRC}" "${BIN_DST}"

log "installing unit ${UNIT_SRC} -> ${UNIT_DST}"
run install -m 0644 -o root -g root "${UNIT_SRC}" "${UNIT_DST}"

log "creating ${SHM_DIR} (0750)"
run install -d -m 0750 -o root -g root "${SHM_DIR}"

# systemd's RuntimeDirectory would be preferable, but the shim (not the daemon)
# creates the per-sandbox files, so the directory must survive a daemon restart.
run systemctl daemon-reload
log "enabling and starting cube-gpu-daemon"
run systemctl enable --now cube-gpu-daemon.service

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "dry-run complete; nothing was changed"
else
    log "installed. Check status with: systemctl status cube-gpu-daemon"
    if ! systemctl is-active --quiet cube-gpu-daemon.service; then
        warn "service is not active; inspect with: journalctl -u cube-gpu-daemon -e"
    fi
fi
