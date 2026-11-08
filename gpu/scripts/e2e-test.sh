#!/usr/bin/env bash
#
# Copyright (c) 2024 Tencent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Host-loopback end-to-end test for CubeSandbox GPU remoting.
#
# Starts cube-gpu-daemon on the local vsock loopback, then runs the CUDA test
# programs with libcube-gpu.so preloaded:
#
#   1. e2e_main  -- full API sequence (memory, kernels, mixed/32B params,
#                   memset, events, async alloc)
#   2. e2e_link  -- build-time linked -lcuda binary (LD_PRELOAD interposition)
#   3. e2e_main --quota-oom  -- quota enforcement
#
# Finally it inspects the daemon log for session init/destroy records and
# checks the daemon is still alive.
#
# Usage: gpu/scripts/e2e-test.sh [--require-bigarg] [extra e2e_main args...]
#
# Environment overrides:
#   E2E_WORK_DIR     default /tmp/cube-gpu-e2e
#   E2E_VSOCK_PORT   default 20565
#   E2E_VSOCK_CID    default 1 (VMADDR_CID_LOCAL)
#   E2E_QUOTA        default 536870912 (512 MiB)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

WORK=${E2E_WORK_DIR:-/tmp/cube-gpu-e2e}
PORT=${E2E_VSOCK_PORT:-20565}
CID=${E2E_VSOCK_CID:-1}
QUOTA=${E2E_QUOTA:-536870912}
SANDBOX=${E2E_SANDBOX_ID:-1001}

SHM_DIR=$WORK/shm
LOG_DIR=$WORK/logs
SHM_PATH=$SHM_DIR/sandbox-$SANDBOX.shm

GUEST_SO=$ROOT/gpu/guest/src/libcube-gpu.so
DAEMON=$ROOT/gpu/host/target/debug/cube-gpu-daemon
E2E_MAIN=$ROOT/gpu/tests/e2e_main
E2E_LINK=$ROOT/gpu/tests/e2e_link
PROBE=$ROOT/gpu/tests/vsock_probe
CUBIN=$ROOT/gpu/tests/kernel.cubin
PTX=$ROOT/gpu/tests/kernel.ptx

MAIN_ARGS=()
REQUIRE_BIGARG=0
for arg in "$@"; do
    case "$arg" in
        --require-bigarg) REQUIRE_BIGARG=1 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) MAIN_ARGS+=("$arg") ;;
    esac
done

step()  { printf '\n=== %s ===\n' "$*"; }
info()  { printf '[info] %s\n' "$*"; }
fail()  { printf '[FAIL] %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [[ -n "${DAEMON_PID:-}" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── 1. Build ────────────────────────────────────────────────────────
step "1/7 build guest library"
# Force a rebuild: gpu/guest/src/Makefile has no header dependencies, so a
# .so compiled against an older rpc_protocol.h would otherwise be reused and
# speak a stale wire version (e.g. v1 against a v2 daemon).
make -B -C gpu/guest/src

step "1/7 build host daemon"
cargo build --manifest-path gpu/host/Cargo.toml

step "1/7 build test programs and kernel fixtures"
make -C gpu/tests kernels programs

[[ -x $GUEST_SO ]] || fail "missing $GUEST_SO"
[[ -x $DAEMON ]]   || fail "missing $DAEMON"
[[ -f $CUBIN ]]    || fail "missing $CUBIN (make -C gpu/tests kernels)"

# ── 2. Workspace + sparse SHM files ────────────────────────────────
step "2/7 prepare $WORK"
rm -rf "$WORK"
mkdir -p "$SHM_DIR" "$LOG_DIR"
# 128 MiB = 2 * CUBE_GPU_SHM_RING_SIZE, matching what the guest maps.
# The global header is deliberately NOT pre-written: the daemon initializes
# the file in test mode and accepts an all-zero token.
truncate -s 128M "$SHM_PATH"
info "sparse SHM file: $SHM_PATH ($(stat -c %s "$SHM_PATH") bytes apparent)"

# ── 3. Start daemon ────────────────────────────────────────────────
step "3/7 start cube-gpu-daemon on vsock port $PORT"
CUBE_GPU_SHM_DIR="$SHM_DIR" CUBE_GPU_VSOCK_PORT="$PORT" RUST_LOG=debug \
    "$DAEMON" >"$LOG_DIR/daemon.log" 2>&1 &
DAEMON_PID=$!
info "daemon pid=$DAEMON_PID log=$LOG_DIR/daemon.log"

READY=0
for _ in $(seq 1 300); do
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo "---- daemon log ----" >&2
        cat "$LOG_DIR/daemon.log" >&2
        fail "daemon exited before listening"
    fi
    if "$PROBE" --cid "$CID" --port "$PORT" --timeout 100 >/dev/null 2>&1; then
        READY=1
        info "vsock CID $CID port $PORT reachable"
        break
    fi
    # Loopback also accepts VMADDR_CID_HOST on this kernel; try it as well.
    if "$PROBE" --cid 2 --port "$PORT" --timeout 100 >/dev/null 2>&1; then
        READY=1
        info "vsock CID 2 (host) port $PORT reachable"
        break
    fi
    sleep 0.1
done
[[ $READY -eq 1 ]] || { cat "$LOG_DIR/daemon.log" >&2; fail "daemon never became reachable"; }

COMMON_ENV=(
    CUBE_GPU_VSOCK_CID="$CID"
    CUBE_GPU_VSOCK_PORT="$PORT"
    CUBE_GPU_SANDBOX_ID="$SANDBOX"
    CUBE_GPU_MEMORY_QUOTA="$QUOTA"
    CUBE_GPU_SHM_PATH="$SHM_PATH"
    CUBE_GPU_DEBUG=1
    LD_PRELOAD="$GUEST_SO"
)
if [[ $REQUIRE_BIGARG -eq 1 ]]; then
    COMMON_ENV+=(E2E_REQUIRE_BIGARG=1)
fi

RC_MAIN=0
RC_PTX=0
RC_LINK=0
RC_OOM=0

# ── 4. Full API sequence ───────────────────────────────────────────
step "4/8 e2e_main (full API sequence, cubin)"
set +e
env "${COMMON_ENV[@]}" "$E2E_MAIN" --cubin "$CUBIN" "${MAIN_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/e2e_main.log"
RC_MAIN=${PIPESTATUS[0]}
set -e
info "e2e_main exit=$RC_MAIN"

# ── 5. Same sequence with the PTX (JIT) kernel path ────────────────
step "5/8 e2e_main (PTX kernel path)"
set +e
env "${COMMON_ENV[@]}" "$E2E_MAIN" --ptx "$PTX" \
    2>&1 | tee "$LOG_DIR/e2e_ptx.log"
RC_PTX=${PIPESTATUS[0]}
set -e
info "e2e_ptx exit=$RC_PTX"

# ── 6. Direct-linked binary ────────────────────────────────────────
step "6/8 e2e_link (direct -lcuda + LD_PRELOAD)"
set +e
env "${COMMON_ENV[@]}" E2E_REQUIRE_LINK_HOOK=1 "$E2E_LINK" \
    2>&1 | tee "$LOG_DIR/e2e_link.log"
RC_LINK=${PIPESTATUS[0]}
set -e
info "e2e_link exit=$RC_LINK"

# ── 7. Quota enforcement ───────────────────────────────────────────
step "7/8 e2e_main --quota-oom"
set +e
env "${COMMON_ENV[@]}" "$E2E_MAIN" --quota-oom \
    2>&1 | tee "$LOG_DIR/e2e_quota_oom.log"
RC_OOM=${PIPESTATUS[0]}
set -e
info "quota-oom exit=$RC_OOM"

# ── 8. Daemon log + liveness checks ────────────────────────────────
step "8/8 check daemon log and liveness"
CHECK_FAIL=0

if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    printf '[FAIL] daemon crashed during test runs\n'
    CHECK_FAIL=1
else
    printf '[PASS] daemon still alive after all test processes exited\n'
fi

INITS=$(grep -Eic 'session[_ ]init' "$LOG_DIR/daemon.log" || true)
printf '[check] daemon log session-init lines: %s\n' "$INITS"
if [[ ${INITS:-0} -lt 1 ]]; then
    printf '[FAIL] daemon log contains no session init records\n'
    CHECK_FAIL=1
fi

# Session teardown happens asynchronously on connection EOF; give it a moment.
DESTROYS=0
for _ in $(seq 1 30); do
    DESTROYS=$(grep -Eic 'session .*destroyed' "$LOG_DIR/daemon.log" || true)
    if [[ ${DESTROYS:-0} -ge ${INITS:-0} ]]; then
        break
    fi
    sleep 0.1
done
printf '[check] daemon log session-destroyed lines: %s (init: %s)\n' "$DESTROYS" "$INITS"
if [[ ${DESTROYS:-0} -lt 1 ]]; then
    printf '[FAIL] daemon log contains no session destroy records\n'
    CHECK_FAIL=1
elif [[ ${DESTROYS:-0} -lt ${INITS:-0} ]]; then
    printf '[WARN] fewer destroy records than init records (connection cleanup still pending)\n'
fi

printf '\n=== summary ===\n'
printf 'e2e_main      : %s\n' "$([[ $RC_MAIN -eq 0 ]] && echo PASS || echo "FAIL ($RC_MAIN)")"
printf 'e2e_ptx       : %s\n' "$([[ $RC_PTX -eq 0 ]] && echo PASS || echo "FAIL ($RC_PTX)")"
printf 'e2e_link      : %s\n' "$([[ $RC_LINK -eq 0 ]] && echo PASS || echo "FAIL ($RC_LINK)")"
printf 'quota-oom     : %s\n' "$([[ $RC_OOM -eq 0 ]] && echo PASS || echo "FAIL ($RC_OOM)")"
printf 'daemon checks : %s\n' "$([[ $CHECK_FAIL -eq 0 ]] && echo PASS || echo FAIL)"
printf 'logs          : %s\n' "$LOG_DIR"

if [[ $RC_MAIN -ne 0 || $RC_PTX -ne 0 || $RC_LINK -ne 0 || $RC_OOM -ne 0 || $CHECK_FAIL -ne 0 ]]; then
    exit 1
fi
echo "E2E PASS"
