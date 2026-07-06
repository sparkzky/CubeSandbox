// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Adversarial tests for the CubeSandbox Node SDK (issue #760 v1).
 *
 * These target edge cases, error paths and invariants that the
 * implementer-authored core/foundation suites do NOT exercise. Every test
 * drives a public data-plane entry point (runCode, Commands.run, Files.*)
 * through a hand-written fake {@link DataClient} so the parsing/aggregation
 * logic can be probed deterministically, without a real envd/Jupyter backend.
 *
 * The SDK source is intentionally left untouched per the assignment. Any
 * divergences from the contract are reported separately and pinned here as
 * characterization assertions so a future fix updates the test in lockstep.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { Config } from "../src/config.js";
import {
  ApiError,
  AuthenticationError,
  classifyHttpError,
  CubeSandboxError,
  SandboxNotFoundError,
  TemplateNotFoundError,
} from "../src/errors.js";
import { decodeConnectUnary, encodeConnectUnary } from "../src/envd-codec.js";
import { Commands } from "../src/commands.js";
import { Files } from "../src/filesystem.js";
import { runCode } from "../src/code-execution.js";
import type { DataClient, DataRequestInit } from "../src/transport.js";
import {
  FILE_TYPE_DIRECTORY,
  isDirectory,
  type Execution,
  type OutputMessage,
} from "../src/types.js";

// ─── fakes ────────────────────────────────────────────────────────────────

/** One captured data-plane request. */
interface RecordedCall {
  port: number;
  sandboxID: string;
  path: string;
  init: DataRequestInit;
}

/**
 * Build an in-memory {@link DataClient} that records every request and replies
 * via `respond`. Tests assert on `calls` for request shape and on the returned
 * promise for parsed behaviour.
 */
function fakeDataClient(
  respond: (call: RecordedCall) => Response | Promise<Response>,
): { client: DataClient; calls: RecordedCall[] } {
  const calls: RecordedCall[] = [];
  const client: DataClient = {
    async request(port, sandboxID, path, init = {}) {
      const call = { port, sandboxID, path, init };
      calls.push(call);
      return await respond(call);
    },
  };
  return { client, calls };
}

/** Build a Response whose body is an ndjson stream of `lines` (each + `\n`). */
function ndjsonResponse(lines: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) controller.enqueue(encoder.encode(line + "\n"));
      controller.close();
    },
  });
  return new Response(body, { status });
}

/** Build a Response whose body is a concatenation of Connect envelope frames. */
function connectResponse(frames: Uint8Array[], status = 200): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(frame);
      controller.close();
    },
  });
  return new Response(body, { status });
}

// ─── env snapshot (config tests need a clean baseline) ────────────────────

const ENV_KEYS = [
  "CUBE_API_URL",
  "E2B_API_URL",
  "CUBE_API_KEY",
  "E2B_API_KEY",
  "CUBE_TEMPLATE_ID",
  "CUBE_PROXY_NODE_IP",
  "CUBE_PROXY_PORT_HTTP",
  "CUBE_PROXY_SCHEME",
  "CUBE_SANDBOX_DOMAIN",
  "CUBE_TIMEOUT",
  "CUBE_REQUEST_TIMEOUT",
];
const envSnapshot: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const key of ENV_KEYS) envSnapshot[key] = process.env[key];
  for (const key of ENV_KEYS) delete process.env[key];
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    if (envSnapshot[key] === undefined) delete process.env[key];
    else process.env[key] = envSnapshot[key];
  }
});

// ─── runCode ──────────────────────────────────────────────────────────────

describe("runCode — adversarial edges", () => {
  it("returns an empty Execution (no throw) when the response body is null", async () => {
    // Exercises the `if (body === null) return execution` branch directly.
    // NOTE (see bug report): this branch skips the `execution.text = ""`
    // defaulting that the non-null empty-body path performs, so execution.text
    // is `undefined` here. The contract for this case only mandates an empty
    // Execution with no throw, which we assert below.
    const { client } = fakeDataClient(() => new Response(null, { status: 200 }));
    const execution: Execution = await runCode(client, "sb", "x");

    expect(execution.results).toEqual([]);
    expect(execution.logs).toEqual({ stdout: [], stderr: [] });
    expect(execution.error).toBeNull();
    expect(execution.executionCount).toBeNull();
  });

  it("skips malformed ndjson lines and keeps processing later lines", async () => {
    const { client } = fakeDataClient(() =>
      ndjsonResponse([
        "{ this is not json",
        JSON.stringify({ type: "stdout", text: "first" }),
        "another } totally broken line",
        JSON.stringify({ type: "stdout", text: "second" }),
      ]),
    );
    const execution = await runCode(client, "sb", "x");
    expect(execution.logs.stdout).toEqual(["first", "second"]);
    expect(execution.logs.stderr).toEqual([]);
  });

  it("preserves per-channel order and callback arrival order for interleaved stdout/stderr", async () => {
    const { client } = fakeDataClient(() =>
      ndjsonResponse([
        JSON.stringify({ type: "stdout", text: "a" }),
        JSON.stringify({ type: "stderr", text: "b" }),
        JSON.stringify({ type: "stdout", text: "c" }),
        JSON.stringify({ type: "stderr", text: "d" }),
      ]),
    );
    const arrivals: string[] = [];
    const execution = await runCode(client, "sb", "x", {
      onStdout: (m: OutputMessage) => arrivals.push("U" + m.text),
      onStderr: (m: OutputMessage) => arrivals.push("E" + m.text),
    });
    // Per-channel aggregation preserves relative order within each channel.
    expect(execution.logs.stdout).toEqual(["a", "c"]);
    expect(execution.logs.stderr).toEqual(["b", "d"]);
    // Callbacks fire in stream-arrival order (interleaved, not channel-batched).
    expect(arrivals).toEqual(["Ua", "Eb", "Uc", "Ed"]);
  });

  it("defaults a stdout event with missing or non-string text to the empty string", async () => {
    const { client } = fakeDataClient(() =>
      ndjsonResponse([
        JSON.stringify({ type: "stdout" }), // no text field at all
        JSON.stringify({ type: "stdout", text: 42 }), // wrong type (number)
      ]),
    );
    const received: string[] = [];
    const execution = await runCode(client, "sb", "x", {
      onStdout: (m) => received.push(m.text),
    });
    expect(execution.logs.stdout).toEqual(["", ""]);
    expect(received).toEqual(["", ""]);
  });

  it("coerces missing/null traceback to [] while preserving string and array forms", async () => {
    const { client } = fakeDataClient(() =>
      ndjsonResponse([
        JSON.stringify({ type: "error", name: "A", value: "1" }), // missing tb
        JSON.stringify({ type: "error", name: "B", value: "2", traceback: "single" }),
        JSON.stringify({ type: "error", name: "C", value: "3", traceback: ["f1", "f2"] }),
        JSON.stringify({ type: "error", name: "D", value: "4", traceback: null }),
      ]),
    );
    const seen: string[] = [];
    const execution = await runCode(client, "sb", "x", {
      onError: (e) => seen.push(JSON.stringify(e.traceback)),
    });

    // Every error event is delivered via onError with its traceback shape.
    expect(seen).toEqual([
      JSON.stringify([]),
      JSON.stringify("single"),
      JSON.stringify(["f1", "f2"]),
      JSON.stringify([]),
    ]);
    // The last error event wins on execution.error (null tb → []).
    expect(execution.error?.name).toBe("D");
    expect(execution.error?.traceback).toEqual([]);
  });
});

// ─── commands ─────────────────────────────────────────────────────────────

describe("commands.run — adversarial edges", () => {
  it("throws when the end-stream trailer carries a Connect error object (code + message)", async () => {
    // Trailer frame: end-stream flag (0x02) + body { error: { code, message } }.
    const trailer = encodeConnectUnary(
      { error: { code: "permission_denied", message: "envd denied the call" } },
      0x02, // CONNECT_END_STREAM_FLAG
    );
    const { client } = fakeDataClient(() => connectResponse([trailer]));
    await expect(
      new Commands({ dataClient: client, sandboxID: "sb" }).run("x"),
    ).rejects.toThrow("permission_denied: envd denied the call");
  });

  it("throws 'process stream ended without EndEvent' when the stream has no end event", async () => {
    // A clean stream (no partial frame) that simply ends after a data frame.
    const { client } = fakeDataClient(() =>
      connectResponse([
        encodeConnectUnary({ event: { data: { stdout: Buffer.from("hi").toString("base64") } } }),
      ]),
    );
    await expect(
      new Commands({ dataClient: client, sandboxID: "sb" }).run("x"),
    ).rejects.toThrow("process stream ended without EndEvent");
  });

  it("decodes base64 stdout including multi-byte UTF-8 (Chinese) without splitting", async () => {
    const b64 = Buffer.from("你好，世界").toString("base64");
    const { client } = fakeDataClient(() =>
      connectResponse([
        encodeConnectUnary({ event: { data: { stdout: b64 } } }),
        encodeConnectUnary({ event: { end: { exitCode: 0 } } }),
      ]),
    );
    const result = await new Commands({ dataClient: client, sandboxID: "sb" }).run("x");
    expect(result.stdout).toBe("你好，世界");
    expect(result.exitCode).toBe(0);
  });

  it("reads exitCode from the camelCase `exitCode` field on the EndEvent", async () => {
    const { client } = fakeDataClient(() =>
      connectResponse([encodeConnectUnary({ event: { end: { exitCode: 42 } } })]),
    );
    const result = await new Commands({ dataClient: client, sandboxID: "sb" }).run("x");
    expect(result.exitCode).toBe(42);
  });

  it("reads exitCode from a `status` string ('exit status 1') on the EndEvent", async () => {
    const { client } = fakeDataClient(() =>
      connectResponse([encodeConnectUnary({ event: { end: { status: "exit status 1" } } })]),
    );
    const result = await new Commands({ dataClient: client, sandboxID: "sb" }).run("x");
    expect(result.exitCode).toBe(1);
  });

  it("throws 'process failed: <error>' when the EndEvent carries an error string", async () => {
    const { client } = fakeDataClient(() =>
      connectResponse([encodeConnectUnary({ event: { end: { error: "oom killed" } } })]),
    );
    await expect(
      new Commands({ dataClient: client, sandboxID: "sb" }).run("x"),
    ).rejects.toThrow("process failed: oom killed");
  });

  it("sends X-Access-Token and a Basic auth header reflecting a non-default user", async () => {
    const { client, calls } = fakeDataClient(() =>
      connectResponse([encodeConnectUnary({ event: { end: { exitCode: 0 } } })]),
    );
    await new Commands({ dataClient: client, sandboxID: "sb", envdAccessToken: "tok-123" }).run(
      "echo hi",
      { user: "alice" },
    );
    const headers = calls[0]!.init.headers!;
    expect(headers["X-Access-Token"]).toBe("tok-123");
    const auth = headers["Authorization"];
    expect(auth.startsWith("Basic ")).toBe(true);
    const decoded = Buffer.from(auth.slice("Basic ".length), "base64").toString("utf-8");
    expect(decoded).toBe("alice:");
  });

  it("round-trips a process.Start payload through the envd Connect envelope codec", () => {
    const payload = {
      process: {
        cmd: "/bin/bash",
        args: ["-l", "-c", "echo hi"],
        envs: { FOO: "bar", ÜÑ: "中文" },
        cwd: "/work",
      },
      stdin: false,
    };
    const encoded = encodeConnectUnary(payload);
    // Envelope framing: 1 flag byte + 4 big-endian length bytes + UTF-8 JSON body.
    const bodyBytes = new TextEncoder().encode(JSON.stringify(payload));
    expect(encoded.byteLength).toBe(5 + bodyBytes.byteLength);
    expect(encoded[0]).toBe(0); // flags (uncompressed, not end-stream)
    const decoded = decodeConnectUnary(encoded);
    expect(decoded).toEqual(payload);
  });

  it("classifies a 401 data-plane response as AuthenticationError via classifyHttpError", async () => {
    const { client } = fakeDataClient(
      () => new Response(JSON.stringify({ message: "unauthorized" }), { status: 401 }),
    );
    await expect(
      new Commands({ dataClient: client, sandboxID: "sb" }).run("x"),
    ).rejects.toMatchObject({ name: "AuthenticationError", statusCode: 401 });
  });
});

// ─── files ────────────────────────────────────────────────────────────────

describe("files — adversarial edges", () => {
  it("read() URL-encodes path + username (spaces & special chars) on a GET to ENVD_PORT", async () => {
    const path = "/home/u/my file (1).txt";
    const { client, calls } = fakeDataClient(() => new Response("file-body", { status: 200 }));
    const text = await new Files({
      dataClient: client,
      sandboxID: "sb",
      envdAccessToken: "tok",
    }).read(path, { user: "alice" });

    expect(text).toBe("file-body");
    const call = calls[0]!;
    expect(call.port).toBe(49983); // ENVD_PORT
    expect(call.init.method).toBe("GET");
    expect(call.path).toBe(
      `/files?path=${encodeURIComponent(path)}&username=alice`,
    );
    expect(call.init.headers!["X-Access-Token"]).toBe("tok");
  });

  it("list() maps FILE_TYPE_DIRECTORY so isDirectory() is true (and coerces string sizes)", async () => {
    const { client } = fakeDataClient(() =>
      new Response(
        JSON.stringify({
          entries: [
            { name: "sub", type: "FILE_TYPE_DIRECTORY", path: "/d/sub", size: "0", mode: 493 },
            { name: "a.txt", type: "FILE_TYPE_UNSPECIFIED", path: "/d/a.txt", size: "128", mode: 420 },
          ],
        }),
        { status: 200 },
      ),
    );
    const entries = await new Files({ dataClient: client, sandboxID: "sb" }).list("/d");
    expect(entries).toHaveLength(2);
    expect(entries[0]!.type).toBe(FILE_TYPE_DIRECTORY);
    expect(isDirectory(entries[0]!)).toBe(true);
    expect(isDirectory(entries[1]!)).toBe(false);
    // envd serialises size as a JSON string; parser must coerce → number.
    expect(entries[1]!.size).toBe(128);
  });

  it("stat() returns the single entry and reports directories correctly", async () => {
    const { client } = fakeDataClient(() =>
      new Response(
        JSON.stringify({
          entry: { name: "d", type: "FILE_TYPE_DIRECTORY", path: "/d", size: 0, mode: 493 },
        }),
        { status: 200 },
      ),
    );
    const entry = await new Files({ dataClient: client, sandboxID: "sb" }).stat("/d");
    expect(entry.name).toBe("d");
    expect(entry.path).toBe("/d");
    expect(isDirectory(entry)).toBe(true);
  });

  it("parses a FileEntry with no fields into safe defaults (no throw)", async () => {
    const { client } = fakeDataClient(() =>
      new Response(JSON.stringify({ entries: [{}] }), { status: 200 }),
    );
    const entries = await new Files({ dataClient: client, sandboxID: "sb" }).list("/d");
    expect(entries).toHaveLength(1);
    expect(entries[0]).toEqual({
      name: "",
      type: "",
      path: "",
      size: 0,
      mode: 0,
      permissions: "",
      owner: "",
      group: "",
      modifiedTime: "",
    });
  });

  it("read() 404 is classified as SandboxNotFoundError", async () => {
    const { client } = fakeDataClient(
      () => new Response(JSON.stringify({ message: "file gone" }), { status: 404 }),
    );
    await expect(
      new Files({ dataClient: client, sandboxID: "sb" }).read("/x"),
    ).rejects.toMatchObject({ name: "SandboxNotFoundError", statusCode: 404 });
  });
});

// ─── config ───────────────────────────────────────────────────────────────

describe("Config.fromEnv — duration & normalization edges", () => {
  const DURATION_CASES: Array<[string, number]> = [
    ["1.5", 1500], // bare decimal seconds → ms
    ["5m", 300_000],
    ["2h", 7_200_000],
    ["500ms", 500],
    ["abc", 300_000], // invalid → default
    ["", 300_000], // empty → default
  ];
  for (const [input, expected] of DURATION_CASES) {
    it(`parses CUBE_TIMEOUT=${JSON.stringify(input)} → ${expected}ms`, () => {
      process.env.CUBE_TIMEOUT = input;
      expect(Config.fromEnv().timeoutMs).toBe(expected);
    });
  }

  it("strips trailing slashes from CUBE_API_URL (single and multiple)", () => {
    process.env.CUBE_API_URL = "https://api.example.com/";
    expect(Config.fromEnv().apiUrl).toBe("https://api.example.com");
    process.env.CUBE_API_URL = "https://api.example.com//";
    expect(Config.fromEnv().apiUrl).toBe("https://api.example.com");
  });

  it("falls back to the default timeout when an explicit override is <= 0", () => {
    expect(Config.fromEnv({ timeoutMs: -1 }).timeoutMs).toBe(300_000);
    expect(Config.fromEnv({ timeoutMs: 0 }).timeoutMs).toBe(300_000);
    expect(Config.fromEnv({ requestTimeoutMs: 0 }).requestTimeoutMs).toBe(30_000);
  });

  it("explicit proxyScheme=http wins over the 443→https default", () => {
    expect(Config.fromEnv({ proxyPortHttp: 443 }).proxyScheme).toBe("https");
    expect(Config.fromEnv({ proxyPortHttp: 443, proxyScheme: "http" }).proxyScheme).toBe("http");
  });
});

// ─── errors ───────────────────────────────────────────────────────────────

describe("classifyHttpError — full status matrix", () => {
  const MATRIX: Array<[number, string, typeof CubeSandboxError]> = [
    [401, "unauthorized", AuthenticationError],
    [403, "forbidden", AuthenticationError],
    [404, "template 'base' not found", TemplateNotFoundError],
    [404, "sandbox 'sb' not found", SandboxNotFoundError],
    [500, "internal", ApiError],
    [503, "unavailable", ApiError],
  ];
  for (const [status, message, cls] of MATRIX) {
    it(`HTTP ${status} (${JSON.stringify(message)}) → ${cls.name}`, () => {
      const err = classifyHttpError(status, message);
      expect(err).toBeInstanceOf(cls);
      expect(err).toBeInstanceOf(CubeSandboxError);
      expect(err).toBeInstanceOf(Error);
      expect(err.name).toBe(cls.name);
      expect(err.statusCode).toBe(status);
      expect(err.message).toBe(message);
    });
  }

  it("detects the 'template' keyword case-insensitively on 404", () => {
    expect(classifyHttpError(404, "Template missing")).toBeInstanceOf(TemplateNotFoundError);
    expect(classifyHttpError(404, "TEMPLATE deleted")).toBeInstanceOf(TemplateNotFoundError);
  });
});
