// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import http from "node:http";
import type { AddressInfo } from "node:net";

import { Config } from "../src/config.js";
import {
  decodeConnectUnary,
  encodeConnectUnary,
} from "../src/envd-codec.js";
import { createControlClient, createDataClient } from "../src/transport.js";
import { runCode } from "../src/code-execution.js";
import { Commands } from "../src/commands.js";
import { Files } from "../src/filesystem.js";
import { Sandbox } from "../src/sandbox.js";
import type { Execution, OutputMessage, Result } from "../src/types.js";
import { ApiError, AuthenticationError, CubeSandboxError } from "../src/errors.js";
import type { DataClient } from "../src/transport.js";
/** Start an HTTP server on a random loopback port; resolves to { server, port }. */
async function startServer(
  handler: (req: http.IncomingMessage, res: http.ServerResponse) => void,
): Promise<{ server: http.Server; port: number }> {
  const server = http.createServer(handler);
  const port = await new Promise<number>((resolve, reject) => {
    server.listen(0, "127.0.0.1", () => resolve((server.address() as AddressInfo).port));
    server.on("error", reject);
  });
  return { server, port };
}

/** Read the full request body into a Buffer. */
async function readBody(req: http.IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks);
}

/**
 * Tiny deferred (executor form — `Promise.withResolvers` needs ES2024 and the
 * Foundation tsconfig is pinned to ES2022 / Node 18). Returns the same
 * `{ promise, resolve }` shape so handlers can signal one captured value.
 */
function capture<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

/** Loopback data-plane config that routes every request to `port`. */
function localDataConfig(port: number): Config {
  return Config.fromEnv({
    proxyNodeIp: "127.0.0.1",
    proxyPortHttp: port,
    proxyScheme: "http",
    sandboxDomain: "cube.test",
  });
}

// ──────────────────────────────────────────────────────────────────────────
// (a) runCode — ndjson stream parsing & event dispatch
// ──────────────────────────────────────────────────────────────────────────

describe("runCode ndjson stream", () => {
  it("dispatches every event type and maps result wire keys", async () => {
    // Build an ndjson body exercising all five event types.
    const resultEvent = JSON.stringify({
      type: "result",
      text: "42",
      json_data: { answer: 42 },
      is_main_result: true,
    });
    const stdoutEvent = JSON.stringify({ type: "stdout", text: "printing\n", timestamp: 1 });
    const stderrEvent = JSON.stringify({ type: "stderr", text: "warn\n", timestamp: 2 });
    const errorEvent = JSON.stringify({
      type: "error",
      name: "ValueError",
      value: "bad",
      traceback: ["frame1", "frame2"],
    });
    const countEvent = JSON.stringify({ type: "number_of_executions", execution_count: 3 });
    const ndjson = [resultEvent, stdoutEvent, stderrEvent, errorEvent, countEvent, ""].join("\n");

    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      res.setHeader("Content-Type", "application/json");
      res.writeHead(200);
      res.end(ndjson);
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      const stdoutMsgs: OutputMessage[] = [];
      const stderrMsgs: OutputMessage[] = [];
      const results: Result[] = [];
      let errored: boolean | undefined;

      const execution: Execution = await runCode(
        dataClient,
        "sb1",
        "1+1",
        {
          onStdout: (m) => stdoutMsgs.push(m),
          onStderr: (m) => stderrMsgs.push(m),
          onResult: (r) => results.push(r),
          onError: () => {
            errored = true;
          },
        },
        { trafficAccessToken: "traffic-tok" },
      );

      // Result: snake_case wire keys mapped to camelCase TS fields.
      expect(execution.results).toHaveLength(1);
      const result = execution.results[0]!;
      expect(result.text).toBe("42");
      expect(result.json).toEqual({ answer: 42 });
      expect(result.isMainResult).toBe(true);
      expect(results).toHaveLength(1);

      // Convenience text comes from the main result.
      expect(execution.text).toBe("42");

      // Logs.
      expect(execution.logs.stdout).toEqual(["printing\n"]);
      expect(execution.logs.stderr).toEqual(["warn\n"]);
      expect(stdoutMsgs[0]).toMatchObject({ text: "printing\n", timestamp: 1 });
      expect(stderrMsgs[0]).toMatchObject({ text: "warn\n", error: true });

      // Error event.
      expect(execution.error).not.toBeNull();
      expect(execution.error?.name).toBe("ValueError");
      expect(execution.error?.value).toBe("bad");
      expect(execution.error?.traceback).toEqual(["frame1", "frame2"]);
      expect(errored).toBe(true);

      // Execution count.
      expect(execution.executionCount).toBe(3);
    } finally {
      server.close();
    }
  });

  it("sends the e2b-traffic-access-token header for restricted sandboxes", async () => {
    const { promise, resolve } = capture<http.IncomingMessage>();
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      resolve(req);
      res.writeHead(200);
      res.end();
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      await runCode(dataClient, "sb", "x", {}, { trafficAccessToken: "tok-abc" });
      const req = await promise;
      expect(req.headers["e2b-traffic-access-token"]).toBe("tok-abc");
    } finally {
      server.close();
    }
  });

  it("maps a non-list traceback string into ExecutionError.traceback", async () => {
    const ndjson = JSON.stringify({ type: "error", name: "Err", value: "v", traceback: "single line" }) + "\n";
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      res.writeHead(200);
      res.end(ndjson);
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      const execution = await runCode(dataClient, "sb", "x");
      expect(execution.error?.traceback).toBe("single line");
    } finally {
      server.close();
    }
  });

  it("throws ApiError on HTTP 4xx with a parsed message", async () => {
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      res.setHeader("Content-Type", "application/json");
      res.writeHead(500);
      res.end(JSON.stringify({ message: "kernel panic" }));
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      await expect(runCode(dataClient, "sb", "x")).rejects.toMatchObject({
        name: "ApiError",
        statusCode: 500,
        message: expect.stringContaining("kernel panic"),
      });
    } finally {
      server.close();
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────
// (b) commands.run — Connect envelope request + stream response
// ──────────────────────────────────────────────────────────────────────────

describe("commands.run", () => {
  it("encodes the request envelope with the correct payload + headers", async () => {
    const { promise, resolve } = capture<{
      headers: http.IncomingHttpHeaders;
      body: Buffer;
    }>();
    const { server, port } = await startServer(async (req, res) => {
      const body = await readBody(req);
      resolve({ headers: req.headers, body });
      // Respond with: stdout "hello\n" then end exitCode 0.
      const stdoutB64 = Buffer.from("hello\n").toString("base64");
      const frames = Buffer.concat([
        encodeConnectUnary({ event: { data: { stdout: stdoutB64 } } }),
        encodeConnectUnary({ event: { end: { exitCode: 0 } } }),
      ]);
      res.writeHead(200, { "Content-Type": "application/connect+json" });
      res.end(frames);
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      const commands = new Commands({
        dataClient,
        sandboxID: "sb1",
        envdAccessToken: "envd-tok",
      });
      const result = await commands.run("echo hi", { cwd: "/tmp", envs: { FOO: "bar" }, user: "root" });

      // Decoded envelope payload matches the E2B/envd process.Start shape.
      const { headers, body } = await promise;
      const payload = decodeConnectUnary(body) as Record<string, unknown>;
      expect(payload["stdin"]).toBe(false);
      const process = payload["process"] as Record<string, unknown>;
      expect(process["cmd"]).toBe("/bin/bash");
      expect(process["args"]).toEqual(["-l", "-c", "echo hi"]);
      expect(process["envs"]).toEqual({ FOO: "bar" });
      expect(process["cwd"]).toBe("/tmp");

      // Headers.
      expect(headers["content-type"]).toBe("application/connect+json");
      expect(headers["connect-protocol-version"]).toBe("1");
      expect(headers["x-access-token"]).toBe("envd-tok");
      const auth = String(headers["authorization"]);
      expect(auth.startsWith("Basic ")).toBe(true);
      expect(Buffer.from(auth.slice("Basic ".length), "base64").toString("utf-8")).toBe("root:");

      // Aggregated result.
      expect(result).toEqual({ stdout: "hello\n", stderr: "", exitCode: 0 });
    } finally {
      server.close();
    }
  });

  it("aggregates interleaved stdout/stderr and honours snake_case exit_code", async () => {
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      const frames = Buffer.concat([
        encodeConnectUnary({ event: { data: { stdout: Buffer.from("a").toString("base64") } } }),
        encodeConnectUnary({ event: { data: { stderr: Buffer.from("b").toString("base64") } } }),
        encodeConnectUnary({ event: { end: { exit_code: 7 } } }),
      ]);
      res.writeHead(200);
      res.end(frames);
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      const result = await new Commands({ dataClient, sandboxID: "sb" }).run("x");
      expect(result).toEqual({ stdout: "a", stderr: "b", exitCode: 7 });
    } finally {
      server.close();
    }
  });

  it("translates a 404 data-plane response into SandboxNotFoundError", async () => {
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      res.writeHead(404);
      res.end(JSON.stringify({ message: "sandbox gone" }));
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      await expect(
        new Commands({ dataClient, sandboxID: "sb" }).run("x"),
      ).rejects.toMatchObject({ name: "SandboxNotFoundError", statusCode: 404 });
    } finally {
      server.close();
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────
// (c) Sandbox.create — control-plane request body fields
// ──────────────────────────────────────────────────────────────────────────

describe("Sandbox.create control-plane body", () => {
  it("sends the create wire body: templateID(capital), timeout(sec), envVars, allow_internet_access(snake), network", async () => {
    const { promise, resolve } = capture<{
      body: string;
    }>();
    const { server, port } = await startServer(async (req, res) => {
      const body = (await readBody(req)).toString("utf-8");
      resolve({ body });
      res.setHeader("Content-Type", "application/json");
      res.writeHead(201);
      res.end(
        JSON.stringify({
          templateID: "tpl-base",
          sandboxID: "sb-123",
          clientID: "cli-1",
          envdVersion: "0.1.0",
          envdAccessToken: "envd-tok",
          domain: "cube.test",
        }),
      );
    })

    try {
      // Flat camelCase config overrides at the top level (issue #760 shape).
      const sb = await Sandbox.create({
        apiUrl: `http://127.0.0.1:${port}`,
        apiKey: "key",
        sandboxDomain: "cube.test",
        templateId: "tpl-base",
        // 3,000 ms → ceiling 3 seconds.
        timeoutMs: 3_000,
        envVars: { A: "1" },
        allowInternetAccess: false,
        network: { allowOut: ["1.1.1.1"], denyOut: ["0.0.0.0/0"] },
      });

      const { body } = await promise;
      const payload = JSON.parse(body) as Record<string, unknown>;
      // Wire key is capital templateID (backend serde rename, mod.rs:228).
      expect(payload["templateID"]).toBe("tpl-base");
      expect(payload["timeout"]).toBe(3); // seconds, ceiling-converted
      expect(payload["envVars"]).toEqual({ A: "1" });
      // snake_case wire quirk (backend NewSandbox only deserialises this form).
      expect(payload["allow_internet_access"]).toBe(false);
      expect(payload["network"]).toMatchObject({
        allowOut: ["1.1.1.1"],
        denyOut: ["0.0.0.0/0"],
      });

      // Instance wiring (idiomatic templateId on the public handle).
      expect(sb.sandboxID).toBe("sb-123");
      expect(sb.templateId).toBe("tpl-base");
      expect(sb.envdAccessToken).toBe("envd-tok");
      expect(sb.domain).toBe("cube.test");
      expect(sb.getHost(49999)).toBe("49999-sb-123.cube.test");
    } finally {
      server.close();
    }
  });

  it("falls back to CUBE_TEMPLATE_ID when templateId is omitted", async () => {
    const saved = process.env.CUBE_TEMPLATE_ID;
    process.env.CUBE_TEMPLATE_ID = "from-env";
    const { promise, resolve } = capture<string>();
    const { server, port } = await startServer(async (req, res) => {
      resolve((await readBody(req)).toString("utf-8"));
      res.writeHead(201);
      res.end(
        JSON.stringify({
          templateID: "t",
          sandboxID: "s",
          clientID: "c",
          envdVersion: "v",
        }),
      );
    })

    try {
      const sb = await Sandbox.create({
        apiUrl: `http://127.0.0.1:${port}`,
        apiKey: "k",
        // templateId omitted → resolveConfig falls back to CUBE_TEMPLATE_ID.
      });
      const body = await promise;
      const payload = JSON.parse(body) as Record<string, unknown>;
      expect(payload["templateID"]).toBe("from-env");
      expect(payload["allow_internet_access"]).toBeUndefined();
      expect(payload["timeout"]).toBe(300); // 300_000 ms default → 300 s
      void sb;
    } finally {
      if (saved === undefined) delete process.env.CUBE_TEMPLATE_ID;
      else process.env.CUBE_TEMPLATE_ID = saved;
      server.close();
    }
  });

  it("throws when neither templateId nor CUBE_TEMPLATE_ID is set", async () => {
    const saved = process.env.CUBE_TEMPLATE_ID;
    delete process.env.CUBE_TEMPLATE_ID;
    try {
      await expect(Sandbox.create({ apiUrl: "http://127.0.0.1:1" })).rejects.toThrow(
        /template is required/,
      );
    } finally {
      if (saved !== undefined) process.env.CUBE_TEMPLATE_ID = saved;
    }
  });

  it("kill() issues DELETE /sandboxes/:id and accepts 204", async () => {
    const { promise, resolve } = capture<{ method: string; url: string }>();
    const { server, port } = await startServer(async (req, res) => {
      resolve({ method: req.method ?? "", url: req.url ?? "" });
      res.writeHead(204);
      res.end();
    })

    try {
      const config = Config.fromEnv({ apiUrl: `http://127.0.0.1:${port}`, apiKey: "k" });
      // Construct an instance without a network call via the connect path is
      // not possible (constructor is private); emulate via create-then-kill is
      // heavy, so drive the control client directly to mirror Sandbox.kill.
      const control = createControlClient(config);
      await control.request("DELETE", "/sandboxes/sb-x", { okStatus: [200, 204] });
      const { method, url } = await promise;
      expect(method).toBe("DELETE");
      expect(url).toBe("/sandboxes/sb-x");
    } finally {
      server.close();
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────
// files — read / write / list / stat round-trips
// ──────────────────────────────────────────────────────────────────────────

describe("files", () => {
  it("read() GETs /files with path+username and returns text", async () => {
    const { promise, resolve } = capture<{ url: string; token?: string }>();
    const { server, port } = await startServer(async (req, res) => {
      resolve({ url: req.url ?? "", token: req.headers["x-access-token"] as string | undefined });
      res.writeHead(200);
      res.end("file-contents");
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      const files = new Files({ dataClient, sandboxID: "sb", envdAccessToken: "tok" });
      const text = await files.read("/etc/hostname", { user: "root" });
      expect(text).toBe("file-contents");
      const { url, token } = await promise;
      expect(url).toBe("/files?path=%2Fetc%2Fhostname&username=root");
      expect(token).toBe("tok");
    } finally {
      server.close();
    }
  });

  it("write() uploads octet-stream and does not fall back on success", async () => {
    let attempts = 0;
    const { promise, resolve } = capture<{ contentType: string; body: string }>();
    const { server, port } = await startServer(async (req, res) => {
      attempts++;
      const body = (await readBody(req)).toString("utf-8");
      resolve({ contentType: req.headers["content-type"] as string, body });
      res.writeHead(200);
      res.end();
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      await new Files({ dataClient, sandboxID: "sb" }).write("/p/hello.txt", "hi");
      const { contentType, body } = await promise;
      expect(attempts).toBe(1); // no multipart fallback
      expect(contentType).toBe("application/octet-stream");
      expect(body).toBe("hi");
    } finally {
      server.close();
    }
  });

  it("write() retries as multipart when envd rejects octet-stream", async () => {
    let attempts = 0;
    const { promise, resolve } = capture<string>();
    const { server, port } = await startServer(async (req, res) => {
      attempts++;
      await readBody(req);
      if (attempts === 1) {
        res.writeHead(415); // unsupported media type → trigger fallback
        res.end();
        return;
      }
      resolve(req.headers["content-type"] as string);
      res.writeHead(200);
      res.end();
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      await new Files({ dataClient, sandboxID: "sb" }).write("/p/f.txt", "data");
      expect(attempts).toBe(2);
      const contentType = await promise;
      expect(contentType.startsWith("multipart/form-data; boundary=")).toBe(true);
    } finally {
      server.close();
    }
  });

  it("list() posts plain-JSON {path} to filesystem.Filesystem/ListDir", async () => {
    const { promise, resolve } = capture<{ url: string; body: string; ct: string }>();
    const { server, port } = await startServer(async (req, res) => {
      resolve({
        url: req.url ?? "",
        body: (await readBody(req)).toString("utf-8"),
        ct: req.headers["content-type"] as string,
      });
      res.writeHead(200);
      res.end(
        JSON.stringify({
          entries: [
            {
              name: "a.txt",
              type: "FILE_TYPE_UNSPECIFIED",
              path: "/d/a.txt",
              size: "128", // envd ships size as a JSON string
              mode: 420,
              permissions: "-rw-r--r--",
              owner: "root",
              group: "root",
              modifiedTime: "2026-01-01T00:00:00Z",
            },
            { name: "sub", type: "FILE_TYPE_DIRECTORY", path: "/d/sub", size: 0, mode: 493 },
          ],
        }),
      );
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      const entries = await new Files({ dataClient, sandboxID: "sb" }).list("/d");
      const { url, body, ct } = await promise;
      expect(url).toBe("/filesystem.Filesystem/ListDir");
      expect(body).toBe(JSON.stringify({ path: "/d" }));
      expect(ct).toBe("application/json"); // NOT envelope-framed
      expect(entries).toHaveLength(2);
      expect(entries[0]).toMatchObject({ name: "a.txt", size: 128 }); // string → number
      expect(entries[1]).toMatchObject({ name: "sub", type: "FILE_TYPE_DIRECTORY" });
    } finally {
      server.close();
    }
  });

  it("stat() returns the single entry object", async () => {
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      res.writeHead(200);
      res.end(JSON.stringify({ entry: { name: "f", type: "FILE_TYPE_UNSPECIFIED", path: "/f", size: 0, mode: 0 } }));
    })

    try {
      const dataClient = createDataClient(localDataConfig(port));
      const entry = await new Files({ dataClient, sandboxID: "sb" }).stat("/f");
      expect(entry.name).toBe("f");
      expect(entry.path).toBe("/f");
    } finally {
      server.close();
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────
// regression — maintainer review fixes (BUG-1, OBS-1, OBS-2, OBS-3)
// ──────────────────────────────────────────────────────────────────────────

describe("regression: error/consistency fixes", () => {
  // BUG-1: a null/empty response body must still yield execution.text === "".
  it("runCode normalizes execution.text to '' when the body is null", async () => {
    const stubClient: Pick<DataClient, "request"> = {
      request: async () => new Response(null, { status: 200 }),
    };
    const exec = await runCode(stubClient, "sb", "code");
    expect(typeof exec.text).toBe("string");
    expect(exec.text).toBe("");
  });

  // OBS-1: runCode ≥400 must route through classifyHttpError so 401 → AuthenticationError.
  it("runCode 401 throws AuthenticationError", async () => {
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      res.writeHead(401);
      res.end(JSON.stringify({ message: "unauthorized" }));
    });
    try {
      const dataClient = createDataClient(localDataConfig(port));
      await expect(runCode(dataClient, "sb", "x")).rejects.toBeInstanceOf(AuthenticationError);
    } finally {
      server.close();
    }
  });

  // OBS-2: files.read must treat any 2xx (e.g. 201) as success.
  it("files.read accepts HTTP 201 as success", async () => {
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      res.writeHead(201);
      res.end("created-contents");
    });
    try {
      const dataClient = createDataClient(localDataConfig(port));
      const text = await new Files({ dataClient, sandboxID: "sb" }).read("/p/f");
      expect(text).toBe("created-contents");
    } finally {
      server.close();
    }
  });

  // OBS-3: command stream errors must be CubeSandboxError subclasses (ApiError).
  it("commands stream without EndEvent throws ApiError (instanceof CubeSandboxError)", async () => {
    const { server, port } = await startServer(async (req, res) => {
      await readBody(req);
      // One data frame, no end event → exitCode stays null.
      res.writeHead(200);
      res.end(encodeConnectUnary({ event: { data: { stdout: Buffer.from("x").toString("base64") } } }));
    });
    try {
      const dataClient = createDataClient(localDataConfig(port));
      const p = new Commands({ dataClient, sandboxID: "sb" }).run("x");
      await expect(p).rejects.toBeInstanceOf(ApiError);
      await expect(p).rejects.toBeInstanceOf(CubeSandboxError);
    } finally {
      server.close();
    }
  });
});
