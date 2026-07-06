// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Sandbox — the central handle for a running CubeSandbox instance.
 *
 * Provides the E2B-compatible class surface (`Sandbox.create` / `.connect` /
 * `.list` / `.runCode` / `.commands` / `.files` …) on top of the Foundation
 * control-plane and data-plane clients.
 *
 * Every static factory accepts **flat, idiomatic camelCase config overrides**
 * at the top level — e.g. `Sandbox.create({ apiUrl, apiKey, proxyNodeIp })` —
 * which are forwarded to `Config.fromEnv(overrides)`. Unset fields fall back to
 * `CUBE_*` / `E2B_*` env vars and then defaults.
 *
 * Lifecycle wire format (grounded in `sdk/python/cubesandbox/sandbox.py`,
 * `sdk/go/client.go`, and the backend `CubeAPI/src/models/mod.rs`):
 *
 *  • create:  `POST /sandboxes`              → {@link SandboxCreateResponse}
 *  • connect: `POST /sandboxes/:id/connect`  → {@link SandboxCreateResponse}
 *  • list:    `GET  /sandboxes`              → {@link SandboxInfo}[]
 *  • listV2:  `GET  /v2/sandboxes`           → {@link SandboxInfo}[]
 *  • health:  `GET  /health`                 → `{ status: string, … }`
 *  • getInfo: `GET  /sandboxes/:id`          → {@link SandboxInfo}
 *  • kill:    `DELETE /sandboxes/:id`        → 204
 *
 * The create request body uses camelCase keys everywhere except the documented
 * `allow_internet_access` snake_case quirk (backend `NewSandbox`,
 * `models/mod.rs:187-189`). `timeout` is sent in **seconds** (ceiling of
 * `timeoutMs`, matching Go `durationSeconds`, `config.go:123-132`).
 */

import type {
  ControlClient,
  DataClient,
} from "./transport.js";
import {
  createControlClient,
  createDataClient,
} from "./transport.js";
import type { ProxyScheme } from "./config.js";
// `Config` is both a type (interface) and a value (const factory); a single
// value import binds both, so no `import type` for it.
import { Config } from "./config.js";
import type {
  CreateOptions,
  Execution,
  NetworkOptions,
  RunCodeOptions,
  SandboxCreateResponse,
  SandboxInfo,
} from "./types.js";
import { Commands } from "./commands.js";
import { Files } from "./filesystem.js";
import { runCode } from "./code-execution.js";

/**
 * Flat config overrides accepted by every static `Sandbox` factory.
 *
 * Each field maps 1:1 to a {@link Config} field; `undefined` means "inherit
 * from env / default". This is what makes `Sandbox.create({ apiUrl, proxyNodeIp })`
 * work without a nested `config` object.
 */
export interface SandboxConfigOverride {
  /** Control-plane base URL (`CUBE_API_URL`). */
  readonly apiUrl?: string;
  /** API key (`CUBE_API_KEY` / `E2B_API_KEY`). */
  readonly apiKey?: string;
  /** Data-plane TCP override IP (`CUBE_PROXY_NODE_IP`). */
  readonly proxyNodeIp?: string;
  /** Data-plane override port (`CUBE_PROXY_PORT_HTTP`). */
  readonly proxyPortHttp?: number;
  /** Data-plane scheme (`CUBE_PROXY_SCHEME`). */
  readonly proxyScheme?: ProxyScheme;
  /** Virtual-host domain suffix (`CUBE_SANDBOX_DOMAIN`). */
  readonly sandboxDomain?: string;
  /** Per-HTTP-request timeout in ms (`CUBE_REQUEST_TIMEOUT`). */
  readonly requestTimeoutMs?: number;
}

/**
 * {@link CreateOptions} with `templateId` made optional (falls back to
 * `Config.templateId` / `CUBE_TEMPLATE_ID`) plus flat config overrides.
 */
export type SandboxCreateOptions = Omit<CreateOptions, "templateId"> & {
  readonly templateId?: string;
} & SandboxConfigOverride;

/** Re-exported so callers can type `sandbox.commands.run(cmd, opts)`. */
export type { CommandOptions } from "./commands.js";

/**
 * A connected CubeSandbox instance. Obtain one via {@link Sandbox.create} or
 * {@link Sandbox.connect}; use {@link Sandbox.commands} / {@link Sandbox.files}
 * / {@link Sandbox.runCode} to interact with the sandbox.
 */
export class Sandbox {
  /** Sandbox identifier (wire `sandboxID`). */
  readonly sandboxID: string;
  /** Template the sandbox was created from. */
  readonly templateId: string;
  /** envd RPC token (`X-Access-Token`). Present when envd requires auth. */
  readonly envdAccessToken?: string;
  /** Restricted-sandbox public-traffic token (`e2b-traffic-access-token`). */
  readonly trafficAccessToken?: string;
  /** Virtual-host domain suffix for this sandbox. */
  readonly domain: string;

  /** Shell-command namespace (envd `process.Process/Start`). */
  readonly commands: Commands;
  /** Filesystem namespace (envd `/files` + `filesystem.Filesystem/*`). */
  readonly files: Files;

  private readonly control: ControlClient;
  private readonly dataClient: DataClient;

  private constructor(resp: SandboxCreateResponse, config: Config) {
    this.sandboxID = resp.sandboxID;
    // `resp.templateID` is the backend wire key (CubeAPI mod.rs:228); expose it
    // as the idiomatic `templateId` on the public instance.
    this.templateId = resp.templateID;
    this.envdAccessToken = resp.envdAccessToken;
    this.trafficAccessToken = resp.trafficAccessToken;
    this.domain = resp.domain ?? config.sandboxDomain;

    this.control = createControlClient(config);
    this.dataClient = createDataClient(config);

    const ctx = {
      dataClient: this.dataClient,
      sandboxID: this.sandboxID,
      envdAccessToken: this.envdAccessToken,
      trafficAccessToken: this.trafficAccessToken,
    };
    this.commands = new Commands(ctx);
    this.files = new Files(ctx);
  }

  // ─── factories ──────────────────────────────────────────────────────────

  /** Create and return a running sandbox (`POST /sandboxes`). */
  static async create(opts: SandboxCreateOptions = {}): Promise<Sandbox> {
    const config = resolveConfig(opts);

    const templateId = opts.templateId ?? config.templateId;
    if (!templateId) {
      throw new Error("template is required. Set CUBE_TEMPLATE_ID or pass templateId.");
    }

    const payload = buildCreatePayload(opts, config);
    const control = createControlClient(config);
    const resp = await control.request<SandboxCreateResponse>("POST", "/sandboxes", {
      body: JSON.stringify(payload),
      okStatus: [200, 201],
    });
    return new Sandbox(resp, config);
  }

  /** Connect to (and resume if paused) an existing sandbox (`POST /sandboxes/:id/connect`). */
  static async connect(sandboxID: string, opts: SandboxConfigOverride = {}): Promise<Sandbox> {
    const config = resolveConfig(opts);
    const control = createControlClient(config);
    const resp = await control.request<SandboxCreateResponse>(
      "POST",
      `/sandboxes/${encodeURIComponent(sandboxID)}/connect`,
      { body: JSON.stringify({ timeout: durationSeconds(config.timeoutMs) }) },
    );
    return new Sandbox(resp, config);
  }

  /** List running sandboxes, v1 shape (`GET /sandboxes`). */
  static async list(opts: SandboxConfigOverride = {}): Promise<SandboxInfo[]> {
    const config = resolveConfig(opts);
    const control = createControlClient(config);
    return control.request<SandboxInfo[]>("GET", "/sandboxes");
  }

  /** List running sandboxes, v2 shape with server-side filtering (`GET /v2/sandboxes`). */
  static async listV2(opts: SandboxConfigOverride = {}): Promise<SandboxInfo[]> {
    const config = resolveConfig(opts);
    const control = createControlClient(config);
    return control.request<SandboxInfo[]>("GET", "/v2/sandboxes");
  }

  /** Check CubeAPI service health (`GET /health`). */
  static async health(opts: SandboxConfigOverride = {}): Promise<Record<string, unknown>> {
    const config = resolveConfig(opts);
    const control = createControlClient(config);
    return control.request<Record<string, unknown>>("GET", "/health");
  }

  // ─── instance lifecycle ─────────────────────────────────────────────────

  /** Destroy the sandbox (`DELETE /sandboxes/:id`). */
  async kill(): Promise<void> {
    await this.control.request("DELETE", `/sandboxes/${this.sandboxID}`, {
      okStatus: [200, 204],
    });
  }

  /** Fetch this sandbox's detail (`GET /sandboxes/:id`). */
  async getInfo(): Promise<SandboxInfo> {
    return this.control.request<SandboxInfo>("GET", `/sandboxes/${this.sandboxID}`);
  }

  /** Execute code in the sandbox via the Jupyter `/execute` stream. */
  async runCode(code: string, opts: RunCodeOptions = {}): Promise<Execution> {
    return runCode(
      this.dataClient,
      this.sandboxID,
      code,
      opts,
      {
        envdAccessToken: this.envdAccessToken,
        trafficAccessToken: this.trafficAccessToken,
      },
    );
  }

  // ─── helpers ────────────────────────────────────────────────────────────

  /** Virtual hostname for a sandbox port, e.g. `49999-<id>.cube.app`. */
  getHost(port: number): string {
    return `${port}-${this.sandboxID}.${this.domain}`;
  }

  /** Human-readable identifier. */
  toString(): string {
    return `Sandbox(id=${this.sandboxID}, domain=${this.domain})`;
  }
}

// ─── config resolution ────────────────────────────────────────────────────

/** Build a {@link Config} from a factory's flat override fields (+ env). */
function resolveConfig(opts: SandboxConfigOverride): Config {
  return Config.fromEnv({
    apiUrl: opts.apiUrl,
    apiKey: opts.apiKey,
    proxyNodeIp: opts.proxyNodeIp,
    proxyPortHttp: opts.proxyPortHttp,
    proxyScheme: opts.proxyScheme,
    sandboxDomain: opts.sandboxDomain,
    requestTimeoutMs: opts.requestTimeoutMs,
  });
}

// ─── create payload builder ───────────────────────────────────────────────

/**
 * Build the `POST /sandboxes` JSON body from {@link SandboxCreateOptions}.
 *
 * Field mapping (backend `NewSandbox`, `CubeAPI/src/models/mod.rs:170-216`):
 *  • `templateID` — camelCase-with-capital-ID wire key (backend serde rename).
 *  • `timeout`    — integer **seconds** (ceiling of ms).
 *  • `envVars`    — camelCase wire key (`envs` alias also accepted server-side).
 *  • `metadata`   — flat object.
 *  • `allow_internet_access` — **snake_case** wire key (documented SDK quirk).
 *  • `network`    — `{ allowPublicTraffic, allowOut, denyOut, rules }`.
 */
function buildCreatePayload(opts: SandboxCreateOptions, config: Config): Record<string, unknown> {
  const timeoutMs = opts.timeoutMs ?? config.timeoutMs;
  const payload: Record<string, unknown> = {
    templateID: opts.templateId ?? config.templateId,
    timeout: durationSeconds(timeoutMs),
  };
  if (opts.envVars) payload["envVars"] = opts.envVars;
  if (opts.metadata) payload["metadata"] = opts.metadata;
  // Wire key is deliberately snake_case here — the backend only deserialises
  // `allow_internet_access` (models/mod.rs:187-189). Sent only when explicitly false,
  // matching Go `client.go:137-139`.
  if (opts.allowInternetAccess === false) payload["allow_internet_access"] = false;
  if (opts.network) {
    const network = serializeNetwork(opts.network);
    if (Object.keys(network).length > 0) payload["network"] = network;
  }
  if (opts.extra) Object.assign(payload, opts.extra);
  return payload;
}

/** Map {@link NetworkOptions} → envd/backend wire keys. */
function serializeNetwork(net: NetworkOptions): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (net.allowPublicTraffic !== undefined && net.allowPublicTraffic !== null) {
    out["allowPublicTraffic"] = net.allowPublicTraffic;
  }
  if (net.allowOut) out["allowOut"] = net.allowOut;
  if (net.denyOut) out["denyOut"] = net.denyOut;
  if (net.rules) out["rules"] = net.rules;
  return out;
}

/** Milliseconds → ceiling seconds, matching Go `durationSeconds` (`config.go:123`). */
function durationSeconds(ms: number): number {
  if (ms <= 0) return 0;
  const seconds = Math.floor(ms / 1000);
  return ms % 1000 !== 0 ? seconds + 1 : seconds;
}
