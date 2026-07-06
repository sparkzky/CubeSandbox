// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * runCode — execute code in a sandbox via the Jupyter `/execute` endpoint.
 *
 * Wire format (grounded in `sdk/python/cubesandbox/sandbox.py:363-388` and
 * `sdk/python/cubesandbox/_stream.py`, cross-checked with `sdk/go/stream.go`):
 *
 *  • POST to data-plane port {@link JUPYTER_PORT} (49999), path `/execute`.
 *  • Request body is a single JSON object:
 *      `{ "code": string, "language": string|null, "env_vars": Record|null }`
 *    Python sends `language`/`env_vars` as JSON `null` when unset; we match.
 *  • Response is newline-delimited JSON (ndjson). Each line carries a `type`:
 *      `result`               → a rich {@link Result} (wire keys are snake_case:
 *                               `json_data`, `is_main_result`; rest are flat).
 *      `stdout` / `stderr`    → `{ text, timestamp }` appended to logs.
 *      `error`                → `{ name, value, traceback }`.
 *      `number_of_executions` → `{ execution_count }`.
 *
 * Restricted sandboxes (those minted with `allow_public_traffic=false`) require
 * the per-sandbox `e2b-traffic-access-token` header on every data-plane call;
 * the Foundation data client does not inject it, so we add it here
 * (matches Python `sandbox.py:752-765` `_build_data_client`).
 */

import type { DataClient } from "./transport.js";
import { JUPYTER_PORT } from "./transport.js";
import { classifyHttpError } from "./errors.js";
import type {
  Execution,
  ExecutionError,
  OutputMessage,
  Result,
  RunCodeOptions,
} from "./types.js";
import { isRecord, readErrorMessage, TRAFFIC_TOKEN_HEADER } from "./internal.js";

/** Per-sandbox credentials forwarded to the data plane. */
export interface RunCodeAuth {
  /** envd RPC token. `/execute` does not consume it, but callers pass it. */
  readonly envdAccessToken?: string;
  /** Restricted-sandbox public-traffic token (`e2b-traffic-access-token`). */
  readonly trafficAccessToken?: string;
}

/**
 * Execute `code` in the sandbox and stream back an {@link Execution}.
 *
 * Callbacks in `opts` fire as events arrive; the returned Execution is the
 * fully-aggregated result once the stream ends.
 */
export async function runCode(
  dataClient: DataClient,
  sandboxID: string,
  code: string,
  opts: RunCodeOptions = {},
  auth: RunCodeAuth = {},
): Promise<Execution> {
  // Wire body matches Python sandbox.py:364-368 — `language` and `env_vars`
  // are sent as JSON null when the caller omits them.
  const payload = JSON.stringify({
    code,
    language: opts.language ?? null,
    env_vars: opts.envs ?? null,
  });

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (auth.trafficAccessToken) {
    headers[TRAFFIC_TOKEN_HEADER] = auth.trafficAccessToken;
  }

  // opts.timeoutMs is an absolute cap (AbortSignal.timeout). Undefined ⇒ no
  // timeout, matching Python's `read=None` default for long streams.
  const signal = opts.timeoutMs !== undefined ? AbortSignal.timeout(opts.timeoutMs) : undefined;

  const res = await dataClient.request(JUPYTER_PORT, sandboxID, "/execute", {
    method: "POST",
    headers,
    body: payload,
    signal,
  });

  if (res.status >= 400) {
    const message = await readErrorMessage(res);
    // Route through the shared classifier so 401/403 → AuthenticationError,
    // 404 → SandboxNotFoundError, etc. — consistent with commands/files.
    throw classifyHttpError(res.status, message || `execute failed: HTTP ${res.status}`);
  }

  const execution: Execution = {
    results: [],
    logs: { stdout: [], stderr: [] },
    error: null,
    executionCount: null,
  };

  const body = res.body;
  if (body !== null) {
    for await (const line of ndjsonLines(body)) {
      parseLine(execution, line, opts);
    }
  }

  // Convenience text: the main result's `text` (Go `mainText`, models.go:176).
  // Set during parsing; default to "" when no main result (or empty body) so
  // `execution.text` is always a string — never undefined.
  if (execution.text === undefined) execution.text = "";
  return execution;
}

// ─── ndjson line reader ───────────────────────────────────────────────────

/** Yield each non-empty line from a byte stream (newline-delimited). */
async function* ndjsonLines(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of stream as AsyncIterable<Uint8Array>) {
    buffer += decoder.decode(chunk, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      if (line.length > 0) yield line;
    }
  }
  // Flush the decoder and any trailing line lacking a final newline.
  buffer += decoder.decode();
  if (buffer.length > 0) yield buffer;
}

// ─── event dispatch ───────────────────────────────────────────────────────

/** Parse one ndjson line and fold it into `execution`, firing callbacks. */
function parseLine(execution: Execution, line: string, opts: RunCodeOptions): void {
  let data: Record<string, unknown>;
  try {
    data = JSON.parse(line) as Record<string, unknown>;
  } catch {
    // Malformed JSON — skip, mirroring Python `_stream.py:27-29`.
    return;
  }

  const type = data["type"];
  if (type === "result") {
    const result = parseResult(data);
    execution.results.push(result);
    if (result.isMainResult) execution.text = result.text ?? "";
    opts.onResult?.(result);
  } else if (type === "stdout" || type === "stderr") {
    const rawText = data["text"];
    const lineText = typeof rawText === "string" ? rawText : "";
    if (type === "stdout") execution.logs.stdout.push(lineText);
    else execution.logs.stderr.push(lineText);

    const msg: OutputMessage = { text: lineText };
    const ts = data["timestamp"];
    if (typeof ts === "number" || typeof ts === "string") msg.timestamp = ts;
    if (type === "stderr") msg.error = true;

    if (type === "stdout") opts.onStdout?.(msg);
    else opts.onStderr?.(msg);
  } else if (type === "error") {
    const rawName = data["name"];
    const rawValue = data["value"];
    const rawTb = data["traceback"];
    const err: ExecutionError = {
      name: typeof rawName === "string" ? rawName : "",
      value: typeof rawValue === "string" ? rawValue : "",
      traceback:
        typeof rawTb === "string"
          ? rawTb
          : Array.isArray(rawTb)
            ? rawTb
            : [],
    };
    execution.error = err;
    opts.onError?.(err);
  } else if (type === "number_of_executions") {
    const count = data["execution_count"];
    execution.executionCount = typeof count === "number" ? count : null;
  }
  // Unknown event types are ignored (forward-compatible).
}

// Wire keys that map 1:1 to the camelCase `Result` string fields.
const RESULT_STRING_FIELDS = [
  "text",
  "html",
  "markdown",
  "svg",
  "png",
  "jpeg",
  "pdf",
  "latex",
  "javascript",
] as const;

/**
 * Map a `result` ndjson event to a {@link Result}.
 *
 * The `type` key is ignored. Wire snake_case keys `json_data` and
 * `is_main_result` become `json` and `isMainResult`; everything else is flat
 * (grounded in Go `models.go:108-121` json tags + Python `_models.py`).
 */
function parseResult(data: Record<string, unknown>): Result {
  const result: Result = {};
  for (const field of RESULT_STRING_FIELDS) {
    const value = data[field];
    if (typeof value === "string") result[field] = value;
  }

  const json = data["json_data"];
  if (isRecord(json)) result.json = json;

  const dataField = data["data"];
  if (isRecord(dataField)) result.data = dataField;

  const extra = data["extra"];
  if (isRecord(extra)) result.extra = extra;

  if (data["is_main_result"] === true) result.isMainResult = true;

  const chart = data["chart"];
  if (chart !== undefined) result.chart = chart;

  return result;
}
