// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import http from "node:http";
import type { AddressInfo } from "node:net";

import { Config } from "../src/config.js";
import {
  decodeConnectStream,
  decodeConnectUnary,
  encodeConnectUnary,
} from "../src/envd-codec.js";
import { createDataClient } from "../src/transport.js";

// Snapshot of env so each test starts from a clean baseline.
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

describe("Config.fromEnv (Go superset)", () => {
  it("applies defaults matching sdk/go/config.go", () => {
    const cfg = Config.fromEnv();
    expect(cfg.apiUrl).toBe("http://127.0.0.1:3000");
    expect(cfg.proxyPortHttp).toBe(80);
    expect(cfg.proxyScheme).toBe("http");
    expect(cfg.sandboxDomain).toBe("cube.app");
    expect(cfg.timeoutMs).toBe(300_000);
    expect(cfg.requestTimeoutMs).toBe(30_000);
  });

  it("prefers CUBE_* over E2B_* aliases", () => {
    process.env.E2B_API_URL = "https://e2b.example.com";
    process.env.E2B_API_KEY = "e2b-key";
    process.env.CUBE_API_URL = "https://cube.example.com";
    process.env.CUBE_API_KEY = "cube-key";

    const cfg = Config.fromEnv();
    expect(cfg.apiUrl).toBe("https://cube.example.com");
    expect(cfg.apiKey).toBe("cube-key");
  });

  it("falls back to E2B_* when CUBE_* is unset", () => {
    process.env.E2B_API_URL = "https://e2b.example.com";
    process.env.E2B_API_KEY = "e2b-key";

    const cfg = Config.fromEnv();
    expect(cfg.apiUrl).toBe("https://e2b.example.com");
    expect(cfg.apiKey).toBe("e2b-key");
  });

  it("normalizes proxy scheme: https when port is 443", () => {
    process.env.CUBE_PROXY_PORT_HTTP = "443";
    expect(Config.fromEnv().proxyScheme).toBe("https");
  });

  it("parses Go-style durations (bare seconds + unit suffix)", () => {
    process.env.CUBE_TIMEOUT = "600";
    process.env.CUBE_REQUEST_TIMEOUT = "45s";
    const cfg = Config.fromEnv();
    expect(cfg.timeoutMs).toBe(600_000);
    expect(cfg.requestTimeoutMs).toBe(45_000);
  });

  it("explicit overrides win over env", () => {
    process.env.CUBE_API_KEY = "env-key";
    const cfg = Config.fromEnv({ apiKey: "override-key", proxyScheme: "https" });
    expect(cfg.apiKey).toBe("override-key");
    expect(cfg.proxyScheme).toBe("https");
  });

  it("is frozen / immutable", () => {
    const cfg = Config.fromEnv();
    expect(Object.isFrozen(cfg)).toBe(true);
  });
});

describe("Connect envelope codec", () => {
  it("round-trips a unary envelope (encode → decode)", () => {
    const payload = { process: { cwd: "/root", args: ["ls", "-la"] } };
    const encoded = encodeConnectUnary(payload);
    expect(encoded.byteLength).toBe(5 + JSON.stringify(payload).length);
    expect(encoded[0]).toBe(0); // flags
    // big-endian length in bytes 1..4
    const view = new DataView(encoded.buffer, encoded.byteOffset, encoded.byteLength);
    expect(view.getUint32(1, false)).toBe(JSON.stringify(payload).length);

    expect(decodeConnectUnary(encoded)).toEqual(payload);
  });

  it("decodes a multi-frame stream", async () => {
    const frame1 = encodeConnectUnary({ event: { data: { stdout: "aGVsbG8=" } } });
    const frame2 = encodeConnectUnary({ event: { end: { exitCode: 0 } } });
    // Concatenate the two envelopes, splitting them across chunks to exercise buffering.
    const combined = new Uint8Array(frame1.byteLength + frame2.byteLength);
    combined.set(frame1, 0);
    combined.set(frame2, frame1.byteLength);

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(combined.subarray(0, 3));
        controller.enqueue(combined.subarray(3, frame1.byteLength + 1));
        controller.enqueue(combined.subarray(frame1.byteLength + 1));
        controller.close();
      },
    });

    const frames = [];
    for await (const frame of decodeConnectStream(stream)) frames.push(frame);
    expect(frames).toHaveLength(2);
    expect(frames[0]?.data).toEqual({ event: { data: { stdout: "aGVsbG8=" } } });
    expect(frames[1]?.data).toEqual({ event: { end: { exitCode: 0 } } });
    expect(frames[1]?.endStream).toBe(false);
  });

  it("rejects compressed unary frames", () => {
    const encoded = encodeConnectUnary({ ok: true }, 0x01);
    expect(() => decodeConnectUnary(encoded)).toThrowError(/not supported/i);
  });
});

describe("data-plane transport (proxyNodeIp override)", () => {
  it("sends the virtual host as the Host header when proxyNodeIp is set", async () => {
    const server = http.createServer((req, res) => {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ host: req.headers.host, path: req.url }));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = (server.address() as AddressInfo).port;

    try {
      const cfg = Config.fromEnv({
        proxyNodeIp: "127.0.0.1",
        proxyPortHttp: port,
        sandboxDomain: "cube.app",
        proxyScheme: "http",
      });
      const client = createDataClient(cfg);

      const res = await client.request(49999, "abc123", "/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: '{"code":"1+1"}',
      });
      const body = (await res.json()) as { host: string; path: string };

      // The Host header must be the virtual host, NOT 127.0.0.1:<port>.
      expect(body.host).toBe("49999-abc123.cube.app");
      expect(body.path).toBe("/execute");
    } finally {
      server.close();
    }
  });
});
