#!/usr/bin/env bash
# Copyright (c) 2024 Tencent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Start a single-node NVIDIA MPS server (one GPU, one MPS daemon) that
# cube-gpu-daemon shares between sandboxes. Consumer GPUs cannot use MIG, so
# MPS is the coexistence mechanism; it is not a security boundary.
#
# Usage: sudo ./start-mps.sh [active_thread_percentage]
#   active_thread_percentage  optional, 1-100, the default per-client SM share
#                             (sets CUDA_MPS_ACTIVE_THREAD_PERCENTAGE).
#
# Idempotent: re-running it leaves an already-running MPS server alone and
# only re-applies the requested default thread percentage.

set -euo pipefail

readonly MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
readonly MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

ACTIVE_THREAD_PERCENTAGE="${1:-}"

log() { printf '[start-mps] %s\n' "$*"; }
warn() { printf '[start-mps] WARNING: %s\n' "$*" >&2; }

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[start-mps] ERROR: nvidia-cuda-mps-control not found (install the NVIDIA driver + CUDA toolkit)" >&2
    exit 1
fi

if [[ -n "${ACTIVE_THREAD_PERCENTAGE}" ]]; then
    if ! [[ "${ACTIVE_THREAD_PERCENTAGE}" =~ ^[0-9]+$ ]] ||
        (( ACTIVE_THREAD_PERCENTAGE < 1 || ACTIVE_THREAD_PERCENTAGE > 100 )); then
        echo "[start-mps] ERROR: active thread percentage must be an integer in [1,100]" >&2
        exit 1
    fi
fi

# MPS needs both directories, and the pipe directory must be writable by every
# client (the daemon/guest share it).
mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
chmod 0777 "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}" 2>/dev/null || true

# ── Optional: exclusive-process mode ─────────────────────────────────────────
# Some consumer GPUs require EXCLUSIVE_PROCESS before MPS clients can connect.
# This changes a system-wide GPU setting and needs root, so it is best-effort:
# a failure only produces a hint and we keep going.
if command -v nvidia-smi >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
        if nvidia-smi -c EXCLUSIVE_PROCESS >/dev/null 2>&1; then
            log "compute mode set to EXCLUSIVE_PROCESS"
        else
            warn "could not set EXCLUSIVE_PROCESS compute mode; MPS may still work, otherwise run: sudo nvidia-smi -c EXCLUSIVE_PROCESS"
        fi
    else
        warn "not running as root; skipping 'nvidia-smi -c EXCLUSIVE_PROCESS' (run as root if MPS clients cannot connect)"
    fi
fi

mps_running() {
    pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1 ||
        pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1
}

# ── Start the MPS server (once) ──────────────────────────────────────────────
if mps_running; then
    log "MPS control daemon already running; not starting another"
else
    export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
    export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
    if [[ -n "${ACTIVE_THREAD_PERCENTAGE}" ]]; then
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTIVE_THREAD_PERCENTAGE}"
    fi
    log "starting MPS control daemon (pipe=${MPS_PIPE_DIR})"
    nvidia-cuda-mps-control -d
    sleep 1
    if ! mps_running; then
        echo "[start-mps] ERROR: MPS control daemon failed to start; see ${MPS_LOG_DIR}" >&2
        exit 1
    fi
fi

# ── Apply the default SM share ───────────────────────────────────────────────
# The environment variable only affects processes spawned after it is set; once
# the server is up the persistent setting is the only reliable knob.
if [[ -n "${ACTIVE_THREAD_PERCENTAGE}" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
    if printf 'set_default_active_thread_percentage %s\n' "${ACTIVE_THREAD_PERCENTAGE}" |
        nvidia-cuda-mps-control >/dev/null 2>&1; then
        log "default active thread percentage set to ${ACTIVE_THREAD_PERCENTAGE}"
    else
        warn "could not set default active thread percentage to ${ACTIVE_THREAD_PERCENTAGE}"
    fi
fi

log "MPS ready (pipe=${MPS_PIPE_DIR}, log=${MPS_LOG_DIR})"
