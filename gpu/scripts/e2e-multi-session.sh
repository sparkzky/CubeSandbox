#!/usr/bin/env bash
#
# Copyright (c) 2024 Tencent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-session isolation test for CubeSandbox GPU remoting.
#
# Phase 1: two e2e_main instances (sandbox 2001 / 2002, 256 MiB each) run in
#          parallel; both must succeed.
# Phase 2: two long-lived session_hold processes keep sessions open.  One is
#          killed with SIGKILL; after 2 seconds the other must still complete a
#          small allocation/round-trip, and the daemon must log the destroyed
#          session without crashing.
#
# Usage: gpu/scripts/e2e-multi-session.sh
#
# Environment overrides:
#   E2E_MULTI_WORK_DIR    default /tmp/cube-gpu-e2e-multi
#   E2E_MULTI_VSOCK_PORT  default 20566
#   E2E_MULTI_QUOTA       default 268435456 (256 MiB)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

WORK=${E2E_MULTI_WORK_DIR:-/tmp/cube-gpu-e2e-multi}
PORT=${E2E_MULTI_VSOCK_PORT:-20566}
CID=${E2E_VSOCK_CID:-1}
QUOTA=${E2E_MULTI_QUOTA:-268435456}
IDS=(2001 2002)

SHM_DIR=$WORK/shm
LOG_DIR=$WORK/logs

GUEST_SO=$ROOT/gpu/guest/src/libcube-gpu.so
DAEMON=$ROOT/gpu/host/target/debug/cube-gpu-daemon
E2E_MAIN=$ROOT/gpu/tests/e2e_main
HOLD=$ROOT/gpu/tests/session_hold
PROBE=$ROOT/gpu/tests/vsock_probe
CUBIN=$ROOT/gpu/tests/kernel.cubin
LOG=$LOG_DIR/daemon.log

step() { printf '\n=== %s ===\n' "$*"; }
info() { printf '[info] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }

CLEANUP_PIDS=()
cleanup() {
    local pid
    for pid in "${CLEANUP_PIDS[@]:-}"; do
        [[ -n $pid ]] || continue
        kill "$pid" 2>/dev/null || true
    done
    if [[ -n "${DAEMON_PID:-}" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

session_env() {
    local id=$1
    shift
    env CUBE_GPU_VSOCK_CID="$CID" \
        CUBE_GPU_VSOCK_PORT="$PORT" \
        CUBE_GPU_SANDBOX_ID="$id" \
        CUBE_GPU_MEMORY_QUOTA="$QUOTA" \
        CUBE_GPU_SHM_PATH="$SHM_DIR/sandbox-$id.shm" \
        CUBE_GPU_DEBUG=1 \
        LD_PRELOAD="$GUEST_SO" \
        "$@"
}

# ── Build ──────────────────────────────────────────────────────────
step "build guest library, daemon and test programs"
# Force a rebuild: gpu/guest/src/Makefile has no header dependencies, so a
# .so compiled against an older rpc_protocol.h would otherwise be reused and
# speak a stale wire version (e.g. v1 against a v2 daemon).
make -B -C gpu/guest/src
cargo build --manifest-path gpu/host/Cargo.toml
make -C gpu/tests kernels programs

[[ -x $GUEST_SO ]] || fail "missing $GUEST_SO"
[[ -x $DAEMON ]]   || fail "missing $DAEMON"
[[ -x $HOLD ]]     || fail "missing $HOLD"

# ── Workspace ──────────────────────────────────────────────────────
step "prepare $WORK (quota ${QUOTA} bytes per session)"
rm -rf "$WORK"
mkdir -p "$SHM_DIR" "$LOG_DIR"
for id in "${IDS[@]}"; do
    truncate -s 128M "$SHM_DIR/sandbox-$id.shm"
    info "sparse SHM: $SHM_DIR/sandbox-$id.shm"
done

# ── Daemon ─────────────────────────────────────────────────────────
step "start cube-gpu-daemon on vsock port $PORT"
CUBE_GPU_SHM_DIR="$SHM_DIR" CUBE_GPU_VSOCK_PORT="$PORT" RUST_LOG=debug \
    "$DAEMON" >"$LOG" 2>&1 &
DAEMON_PID=$!
CLEANUP_PIDS+=("$DAEMON_PID")

READY=0
for _ in $(seq 1 300); do
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        cat "$LOG" >&2
        fail "daemon exited before listening"
    fi
    if "$PROBE" --cid "$CID" --port "$PORT" --timeout 100 >/dev/null 2>&1 ||
       "$PROBE" --cid 2 --port "$PORT" --timeout 100 >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.1
done
[[ $READY -eq 1 ]] || { cat "$LOG" >&2; fail "daemon never became reachable"; }

# ── Phase 1: parallel full suites ─────────────────────────────────
step "phase 1: two e2e_main instances in parallel"
session_env ${IDS[0]} "$E2E_MAIN" --cubin "$CUBIN" >"$LOG_DIR/e2e_${IDS[0]}.log" 2>&1 &
P1=$!
session_env ${IDS[1]} "$E2E_MAIN" --cubin "$CUBIN" >"$LOG_DIR/e2e_${IDS[1]}.log" 2>&1 &
P2=$!
CLEANUP_PIDS+=("$P1" "$P2")

set +e
wait "$P1"; RC1=$?
wait "$P2"; RC2=$?
set -e

printf '[check] e2e_main sandbox %s exit=%d\n' "${IDS[0]}" "$RC1"
printf '[check] e2e_main sandbox %s exit=%d\n' "${IDS[1]}" "$RC2"
if [[ $RC1 -ne 0 ]]; then
    tail -n 30 "$LOG_DIR/e2e_${IDS[0]}.log" >&2
    fail "sandbox ${IDS[0]} full suite failed"
fi
if [[ $RC2 -ne 0 ]]; then
    tail -n 30 "$LOG_DIR/e2e_${IDS[1]}.log" >&2
    fail "sandbox ${IDS[1]} full suite failed"
fi

# Wait for the phase-1 sessions to be torn down before taking a baseline.
sleep 0.5
DESTROY_BEFORE=$(grep -Eic 'session .*destroyed' "$LOG" || true)
info "destroyed sessions before kill test: ${DESTROY_BEFORE:-0}"

# ── Phase 2: kill one of two live sessions ────────────────────────
step "phase 2: two live sessions, SIGKILL one, verify the other"
READY_A=$WORK/session-a.ready
READY_B=$WORK/session-b.ready
TRIGGER_B=$WORK/session-b.trigger
RESULT_B=$WORK/session-b.result

session_env ${IDS[0]} "$HOLD" --ready-file "$READY_A" \
    --trigger-file "$WORK/session-a.trigger" --result-file "$WORK/session-a.result" \
    >"$LOG_DIR/hold-a.log" 2>&1 &
PID_A=$!
session_env ${IDS[1]} "$HOLD" --ready-file "$READY_B" \
    --trigger-file "$TRIGGER_B" --result-file "$RESULT_B" \
    >"$LOG_DIR/hold-b.log" 2>&1 &
PID_B=$!
CLEANUP_PIDS+=("$PID_A" "$PID_B")

for _ in $(seq 1 150); do
    if [[ -f $READY_A && -f $READY_B ]]; then
        break
    fi
    if ! kill -0 "$PID_A" 2>/dev/null; then
        cat "$LOG_DIR/hold-a.log" >&2
        fail "session A exited before becoming ready"
    fi
    if ! kill -0 "$PID_B" 2>/dev/null; then
        cat "$LOG_DIR/hold-b.log" >&2
        fail "session B exited before becoming ready"
    fi
    sleep 0.1
done
[[ -f $READY_A && -f $READY_B ]] || fail "live sessions did not become ready"
info "session A pid=$PID_A ($(cat "$READY_A"))"
info "session B pid=$PID_B ($(cat "$READY_B"))"

info "kill -9 session A (pid $PID_A)"
kill -9 "$PID_A" 2>/dev/null || true
set +e
wait "$PID_A" 2>/dev/null
set -e

sleep 2

if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    cat "$LOG" >&2
    fail "daemon crashed after SIGKILL of a session"
fi
echo "[PASS] daemon alive after SIGKILL"

touch "$TRIGGER_B"
RESULT_OK=0
for _ in $(seq 1 100); do
    if [[ -f $RESULT_B ]]; then
        if grep -q '^ok$' "$RESULT_B"; then
            RESULT_OK=1
            break
        fi
        cat "$RESULT_B" >&2
        break
    fi
    if ! kill -0 "$PID_B" 2>/dev/null; then
        cat "$LOG_DIR/hold-b.log" >&2
        fail "session B died while performing post-kill operation"
    fi
    sleep 0.1
done
[[ $RESULT_OK -eq 1 ]] || fail "session B failed its post-kill operation"
echo "[PASS] session B still usable after sibling SIGKILL"

DESTROYED=0
for _ in $(seq 1 30); do
    DESTROYED=$(grep -Eic 'session .*destroyed' "$LOG" || true)
    if [[ ${DESTROYED:-0} -gt ${DESTROY_BEFORE:-0} ]]; then
        break
    fi
    sleep 0.1
done
printf '[check] destroyed sessions: before=%s after=%s\n' "${DESTROY_BEFORE:-0}" "${DESTROYED:-0}"
if [[ ${DESTROYED:-0} -le ${DESTROY_BEFORE:-0} ]]; then
    fail "daemon did not log destruction of the killed session"
fi
echo "[PASS] daemon logged killed session destruction"

INITS=$(grep -Eic 'session[_ ]init' "$LOG" || true)
printf '[check] total session init records: %s\n' "${INITS:-0}"
[[ ${INITS:-0} -ge 2 ]] || fail "expected at least 2 session init records"

# ── Teardown ───────────────────────────────────────────────────────
kill "$PID_B" 2>/dev/null || true
set +e
wait "$PID_B" 2>/dev/null
set -e

printf '\n=== summary ===\n'
printf 'parallel e2e_main : PASS (2001, 2002)\n'
printf 'survivor session  : PASS\n'
printf 'daemon liveness   : PASS\n'
printf 'logs              : %s\n' "$LOG_DIR"
echo "MULTI-SESSION PASS"
