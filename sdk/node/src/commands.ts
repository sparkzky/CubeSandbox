// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * commands — run shell commands inside a sandbox via envd's Connect-RPC
 * `process.Process/Start` endpoint.
 *
 * Wire format (grounded in `sdk/python/cubesandbox/_commands.py:119-166` and
 * `sdk/go/envd.go:89-129`, `sdk/go/commands.go`):
 *
 *  • POST to data-plane port {@link ENVD_PORT} (49983), path
 *    `/process.Process/Start`.
 *  • envd executes `/bin/bash -l -c <cmd>`.
 *  • Auth headers: `X-Access-Token: <envdAccessToken>` (per-sandbox envd token)
 *    and `Authorization: Basic base64("<user>:")` (user defaults to `root`).
 *    Restricted sandboxes additionally send `e2b-traffic-access-token`.
 *
 * Request framing note [INFERENCE — verify against a live envd in integration]:
 * the Connect protocol frames streaming-RPC request bodies as envelope-prefixed
 * messages, and the Python fallback (`_commands.py:158`) envelope-encodes the
 * request via `_encode_connect_envelope`. The Go SDK (`envd.go:100-109`)
 * instead sends raw JSON bytes with the same `application/connect+json`
 * content type. envd accepts both in practice; we follow the spec-compliant
 * envelope form (matches the assignment's "用 envd-codec 编码请求 envelope"
 * directive and reuses Foundation's {@link encodeConnectUnary}).
 *
 *  • Response is a Connect-JSON stream of envelopes. Each frame's JSON body:
 *      `{ "event": { "data": { "stdout": <base64>, "stderr": <base64> } } }`
 *      `{ "event": { "end":   { "exitCode": <number> } } }`
 *    stdout/stderr are base64-encoded; `end.exitCode` is camelCase (snake_case
 *    `exit_code` also accepted, plus a `status` regex fallback).
 */

import type { DataClient } from "./transport.js";
import { ENVD_PORT } from "./transport.js";
import {
  CONNECT_CONTENT_TYPE,
  CONNECT_PROTOCOL_VERSION,
  decodeConnectStream,
  encodeConnectUnary,
} from "./envd-codec.js";
import { ApiError, classifyHttpError } from "./errors.js";
import type { CommandResult } from "./types.js";
import {
  DEFAULT_ENVD_USER,
  isRecord,
  readErrorMessage,
  TRAFFIC_TOKEN_HEADER,
} from "./internal.js";

/** Options for {@link Commands.run}. */
export interface CommandOptions {
  /** Working directory (envd `process.cwd`). Omitted from the payload when unset. */
  readonly cwd?: string;
  /** Process environment variables (envd `process.envs`). */
  readonly envs?: Record<string, string>;
  /** Alias for {@link CommandOptions.envs} (E2B SDK parity). */
  readonly env?: Record<string, string>;
  /** envd process user; defaults to `root`. */
  readonly user?: string;
  /** Absolute command timeout (ms); sent as `Connect-Timeout-Ms` and enforced locally. */
  readonly timeoutMs?: number;
}

/** Per-sandbox credentials the commands namespace forwards to envd. */
export interface CommandsContext {
  readonly dataClient: DataClient;
  readonly sandboxID: string;
  readonly envdAccessToken?: string;
  readonly trafficAccessToken?: string;
}

/** Run shell commands inside a sandbox through envd's process API. */
export class Commands {
  constructor(private readonly ctx: CommandsContext) {}

  /** Run `cmd` (`/bin/bash -l -c`) and collect stdout/stderr/exitCode. */
  async run(cmd: string, opts: CommandOptions = {}): Promise<CommandResult> {
    const envs = opts.envs ?? opts.env ?? {};
    const user = opts.user ?? DEFAULT_ENVD_USER;
    const cwd = opts.cwd ?? "";

    // Wire payload (Python _commands.py:131-140, Go commands.go:29-37).
    const process: Record<string, unknown> = {
      cmd: "/bin/bash",
      args: ["-l", "-c", cmd],
      envs,
    };
    if (cwd) process["cwd"] = cwd;
    const payload = { process, stdin: false };

    const headers: Record<string, string> = {
      "Content-Type": CONNECT_CONTENT_TYPE,
      "Connect-Protocol-Version": CONNECT_PROTOCOL_VERSION,
      "Connect-Content-Encoding": "identity",
      Authorization: basicAuth(user),
    };
    if (this.ctx.envdAccessToken) headers["X-Access-Token"] = this.ctx.envdAccessToken;
    if (this.ctx.trafficAccessToken) headers[TRAFFIC_TOKEN_HEADER] = this.ctx.trafficAccessToken;
    if (opts.timeoutMs !== undefined) headers["Connect-Timeout-Ms"] = String(opts.timeoutMs);

    const body = encodeConnectUnary(payload);
    const signal = opts.timeoutMs !== undefined ? AbortSignal.timeout(opts.timeoutMs) : undefined;

    const res = await this.ctx.dataClient.request(
      ENVD_PORT,
      this.ctx.sandboxID,
      "/process.Process/Start",
      { method: "POST", headers, body, signal },
    );

    if (res.status >= 400) {
      const message = await readErrorMessage(res);
      throw classifyHttpError(
        res.status,
        message || `command failed: HTTP ${res.status}`,
      );
    }

    return collectProcessStream(res.body);
  }
}

/** `Authorization: Basic base64("<user>:")` — envd process user auth. */
function basicAuth(user: string): string {
  const token = Buffer.from(`${user}:`, "utf-8").toString("base64");
  return `Basic ${token}`;
}

/** Base64 → utf-8 string (envd ships stdout/stderr as base64). */
function decodeBase64(value: string): string {
  return Buffer.from(value, "base64").toString("utf-8");
}

/**
 * Aggregate a `process.Process/Start` Connect stream into a {@link CommandResult}.
 *
 * Mirrors Python `_parse_process_start_stream` (`_commands.py:211-262`) and Go
 * `parseProcessStartStream` (`envd.go:259-328`).
 */
async function collectProcessStream(
  body: ReadableStream<Uint8Array> | null,
): Promise<CommandResult> {
  if (body === null) {
    throw new ApiError("process stream ended without EndEvent");
  }

  const stdout: string[] = [];
  const stderr: string[] = [];
  let exitCode: number | null = null;

  for await (const frame of decodeConnectStream(body)) {
    if (frame.endStream) {
      raiseIfConnectError(frame.data);
      continue;
    }

    const data = frame.data;
    if (!isRecord(data)) continue;

    const event = data["event"];
    if (!isRecord(event)) continue;

    const dataEvent = event["data"];
    if (isRecord(dataEvent)) {
      const out = dataEvent["stdout"];
      if (typeof out === "string" && out.length > 0) stdout.push(decodeBase64(out));
      const err = dataEvent["stderr"];
      if (typeof err === "string" && err.length > 0) stderr.push(decodeBase64(err));
    }

    const end = event["end"];
    if (isRecord(end)) {
      exitCode = exitCodeFromEnd(end);
    }
  }

  if (exitCode === null) {
    throw new ApiError("process stream ended without EndEvent");
  }
  return { stdout: stdout.join(""), stderr: stderr.join(""), exitCode };
}

/** Throw when an end-stream trailer frame carries a Connect error object. */
function raiseIfConnectError(data: unknown): void {
  if (!isRecord(data)) return;
  const err = data["error"];
  if (!isRecord(err)) return;
  const rawMessage = err["message"];
  const message = typeof rawMessage === "string" && rawMessage.length > 0
    ? rawMessage
    : "Connect stream error";
  const rawCode = err["code"];
  const code = typeof rawCode === "string" ? rawCode : "";
  throw new ApiError(code ? `${code}: ${message}` : message);
}

/** Resolve the exit code from a process `end` event (camelCase > snake > status). */
function exitCodeFromEnd(end: Record<string, unknown>): number {
  const camel = end["exitCode"];
  if (typeof camel === "number") return camel;
  const snake = end["exit_code"];
  if (typeof snake === "number") return snake;
  const status = end["status"];
  if (typeof status === "string") {
    const fromStatus = exitCodeFromStatus(status);
    if (fromStatus !== null) return fromStatus;
  }
  const error = end["error"];
  if (typeof error === "string" && error.length > 0) {
    throw new ApiError(`process failed: ${error}`);
  }
  throw new ApiError("process EndEvent missing exit code");
}

/** Parse envd status strings like `"exit status 1"` / `"terminated by signal 9"`. */
function exitCodeFromStatus(status: string): number | null {
  let m = /(?:exit status|exited with code)\s+(-?\d+)/.exec(status);
  if (m) return Number.parseInt(m[1], 10);
  m = /(?:signal|terminated by signal)\s+(\d+)/.exec(status);
  if (m) return 128 + Number.parseInt(m[1], 10);
  if (status === "exited") return 0;
  return null;
}
