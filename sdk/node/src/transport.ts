// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * HTTP transport for the CubeSandbox SDK.
 *
 * Two clients:
 *  • {@link createControlClient} — talks to the control plane (`CUBE_API_URL`)
 *    over normal DNS with `Authorization: Bearer <key>`.
 *  • {@link createDataClient} — talks to a sandbox's data plane (envd :49983 /
 *    Jupyter :49999) via the virtual host `{port}-{sandboxID}.{domain}`. When
 *    `CUBE_PROXY_NODE_IP` is set, TCP is forced to `proxyNodeIp:proxyPortHttp`
 *    while the `Host` header keeps the virtual hostname (equivalent to
 *    `curl --resolve`), mirroring `sdk/python/cubesandbox/_transport.py`
 *    (`IPOverrideTransport`) and `sdk/go/transport.go`.
 *
 * Zero runtime dependencies: control plane uses the global `fetch`; data plane
 * uses Node's built-in `http`/`https` with a custom agent so the IP override
 * works for both clear-text and TLS connections (TLS SNI follows the virtual
 * host so the sandbox cert still validates).
 */

import http from "node:http";
import https from "node:https";
import net from "node:net";
import tls from "node:tls";
import { Readable } from "node:stream";
import type { Config } from "./config.js";
import { classifyHttpError, CubeSandboxError } from "./errors.js";

/** envd RPC port (serves `/files`, `/process.Process/*`, `/filesystem.*`). */
export const ENVD_PORT = 49983;
/** Jupyter `/execute` port. */
export const JUPYTER_PORT = 49999;

/** Request body shapes the SDK actually sends (subset of the fetch `BodyInit`). */
export type RequestBody = string | Uint8Array | ReadableStream<Uint8Array>;

// ─── control plane ────────────────────────────────────────────────────────

export interface ControlRequestInit {
  method?: string;
  headers?: Record<string, string>;
  body?: RequestBody | null;
  signal?: AbortSignal;
  /** Status codes treated as success (defaults to any 2xx). */
  okStatus?: number[];
}

export interface ControlClient {
  readonly baseURL: string;
  /** Issue a request and return the parsed JSON body (or `undefined` for 204). */
  request<T = unknown>(method: string, path: string, init?: ControlRequestInit): Promise<T>;
  /** Issue a request and return the raw `Response`; caller handles the body. */
  requestRaw(method: string, path: string, init?: ControlRequestInit): Promise<Response>;
}

/**
 * Build a control-plane client targeting `config.apiUrl`. Adds the bearer token
 * (when present), applies `config.requestTimeoutMs`, and classifies non-2xx
 * responses via {@link classifyHttpError}.
 */
export function createControlClient(config: Config): ControlClient {
  const baseURL = config.apiUrl;

  async function requestRaw(
    method: string,
    path: string,
    init: ControlRequestInit = {},
  ): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(
      () => controller.abort(new Error("CubeSandbox request timed out")),
      config.requestTimeoutMs,
    );
    const external = init.signal;
    if (external !== undefined) {
      if (external.aborted) {
        controller.abort(external.reason ?? new Error("aborted"));
      } else {
        external.addEventListener("abort", () => controller.abort(), { once: true });
      }
    }

    const headers = new Headers(init.headers);
    if (config.apiKey !== "" && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${config.apiKey}`);
    }

    let body: RequestBody | undefined;
    if (init.body !== undefined && init.body !== null) {
      body = init.body;
      if (!headers.has("Content-Type") && typeof init.body === "string") {
        headers.set("Content-Type", "application/json");
      }
    }

    try {
      return await fetch(baseURL + path, { method, headers, body, signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }

  async function request<T>(
    method: string,
    path: string,
    init: ControlRequestInit = {},
  ): Promise<T> {
    const res = await requestRaw(method, path, init);
    const ok =
      init.okStatus !== undefined
        ? init.okStatus.includes(res.status)
        : res.status >= 200 && res.status < 300;
    if (!ok) {
      const message = await extractErrorMessage(res);
      throw classifyHttpError(res.status, message);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  }

  return { baseURL, request, requestRaw };
}

/**
 * Best-effort extraction of a human message from an error response body.
 *
 * Reads `message`, `detail`, or `error.message` from a JSON object; every field
 * is narrowed to `unknown` before being checked, so no caller-fabricated shape
 * is trusted. Falls back to the raw body or status text.
 */
async function extractErrorMessage(res: Response): Promise<string> {
  let text: string;
  try {
    text = await res.text();
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
  if (text === "") return res.statusText || `HTTP ${res.status}`;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return text;
  }
  if (parsed === null || typeof parsed !== "object") return text;

  // `typeof === "object"` narrows to `object`; indexing requires a record view.
  // This cast is the conventional bridge (any JS object is string-indexable)
  // and yields `unknown` per key — each field is validated below before use.
  const body = parsed as Record<string, unknown>;
  const message = typeof body.message === "string" ? body.message : undefined;
  if (message !== undefined) return message;
  const detail = typeof body.detail === "string" ? body.detail : undefined;
  if (detail !== undefined) return detail;
  const err = body.error;
  if (err !== null && typeof err === "object") {
    const errObj = err as Record<string, unknown>;
    const nested = typeof errObj.message === "string" ? errObj.message : undefined;
    if (nested !== undefined) return nested;
  }
  return text;
}

// ─── data plane ──────────────────────────────────────────────────────────

export interface DataRequestInit {
  method?: string;
  /**
   * Extra request headers. Per-sandbox auth headers (`X-Access-Token`,
   * `e2b-traffic-access-token`, Connect headers) are injected by Core; the
   * `Host` header is always controlled by the transport and cannot be overridden.
   */
  headers?: Record<string, string>;
  body?: Uint8Array | string | null;
  /** Streaming upload body (mutually exclusive with {@link DataRequestInit.body}). */
  bodyStream?: ReadableStream<Uint8Array>;
  signal?: AbortSignal;
}

export interface DataClient {
  /**
   * Issue a data-plane request to `{port}-{sandboxID}.{domain}`.
   *
   * @param port sandbox port (e.g. {@link ENVD_PORT} or {@link JUPYTER_PORT}).
   * @param sandboxID target sandbox id.
   * @param path request path, including any query string.
   * @returns a fetch-shaped `Response` (`.body` is a web `ReadableStream`).
   */
  request(
    port: number,
    sandboxID: string,
    path: string,
    init?: DataRequestInit,
  ): Promise<Response>;
}

/**
 * Build a data-plane client.
 *
 * Virtual host: `{port}-{sandboxID}.{sandboxDomain}`.
 *
 * Routing:
 *  • `proxyNodeIp` unset → normal DNS on the virtual host.
 *  • `proxyNodeIp` set → TCP forced to `proxyNodeIp:proxyPortHttp`; the `Host`
 */
export function createDataClient(config: Config): DataClient {
  const override = config.proxyNodeIp !== "";
  const httpAgent = new http.Agent({ keepAlive: true });
  const httpsAgent = new https.Agent({ keepAlive: true });

  if (override) {
    const proxyIP = config.proxyNodeIp;
    const proxyPort = config.proxyPortHttp;
    // Plain HTTP: redirect the TCP dial to the proxy; no other connection
    // option matters for a clear-text request.
    httpAgent.createConnection = () => net.connect({ host: proxyIP, port: proxyPort });
    // HTTPS: redirect the dial while keeping TLS context on the virtual host.
    // https.Agent injects `servername` (== request host == virtual host) and
    // ALPN into the options before calling createConnection; net's
    // ConnectionOptions doesn't model them, so read via a narrowed view.
    httpsAgent.createConnection = (opts) => {
      const tlsContext = opts as tls.ConnectionOptions;
      return tls.connect({
        host: proxyIP,
        port: proxyPort,
        servername: tlsContext.servername,
        ALPNProtocols: tlsContext.ALPNProtocols,
        rejectUnauthorized: tlsContext.rejectUnauthorized,
      });
    };
  }

  function request(
    port: number,
    sandboxID: string,
    path: string,
    init: DataRequestInit = {},
  ): Promise<Response> {
    const virtualHost = `${port}-${sandboxID}.${config.sandboxDomain}`;
    const isHttps = config.proxyScheme === "https";
    // The Host header is the virtual host, full stop — callers may not override it.
    const headers: Record<string, string> = { ...init.headers, Host: virtualHost };
    if (init.body !== undefined && init.body !== null && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const options: http.RequestOptions = {
      method: init.method ?? "GET",
      protocol: isHttps ? "https:" : "http:",
      // `host` drives the default Host header and the TLS SNI; the custom agent
      // ignores it for the actual TCP target when `proxyNodeIp` is set. When
      // there is no override, the standard scheme port (80/443) is used so the
      // default agent reaches the virtual host on the expected public port.
      host: virtualHost,
      port: isHttps ? 443 : 80,
      path,
      headers,
      agent: isHttps ? httpsAgent : httpAgent,
    };

    return new Promise<Response>((resolve, reject) => {
      const dispatcher = isHttps ? https.request : http.request;
      const req = dispatcher(options, (res) => {
        const responseHeaders = new Headers();
        for (const [key, value] of Object.entries(res.headers)) {
          if (value === undefined) continue;
          responseHeaders.set(key, Array.isArray(value) ? value.join(", ") : String(value));
        }
        // Node types `Readable.toWeb` as an unparameterised `ReadableStream`;
        // the bytes are always `Uint8Array`.
        const body = Readable.toWeb(res) as ReadableStream<Uint8Array>;
        resolve(new Response(body, { status: res.statusCode ?? 200, headers: responseHeaders }));
      });

      req.on("error", (err: NodeJS.ErrnoException) => {
        if (req.destroyed) {
          reject(new CubeSandboxError(err.message));
        } else {
          reject(err);
        }
      });

      const signal = init.signal;
      if (signal !== undefined) {
        if (signal.aborted) {
          req.destroy();
        } else {
          signal.addEventListener("abort", () => req.destroy(), { once: true });
        }
      }

      const stream = init.bodyStream;
      if (stream !== undefined) {
        Readable.fromWeb(stream)
          .on("error", (err) => req.destroy(err))
          .pipe(req);
      } else if (init.body !== undefined && init.body !== null) {
        req.end(init.body);
      } else {
        req.end();
      }
    });
  }

  return { request };
}
