// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * CubeSandbox SDK configuration.
 *
 * Mirrors the environment-variable superset exposed by the Go SDK
 * (`sdk/go/config.go`). `CUBE_*` variables always take precedence over the
 * `E2B_*` aliases so the package is a drop-in for E2B-style deployments.
 *
 * Field names are idiomatic camelCase. The backing env-var names keep their
 * SCREAMING_SNAKE_CASE form (`CUBE_API_URL`, etc.) — only the TS field names
 * differ from the env keys.
 */

/** Data-plane proxy scheme. Derived from {@link Config.proxyPortHttp} when unset. */
export type ProxyScheme = "http" | "https";

/** Mutable, fully-optional shape accepted by {@link Config.fromEnv} as overrides. */
export type ConfigInput = Partial<Config>;

/** Immutable SDK configuration. Construct via {@link Config.fromEnv}. */
export interface Config {
  /** Control-plane base URL. Default `http://127.0.0.1:3000`. Trailing `/` stripped. */
  readonly apiUrl: string;
  /** API key sent as `Authorization: Bearer <key>`. `CUBE_API_KEY` > `E2B_API_KEY`. */
  readonly apiKey: string;
  /** Default template used when no template id is passed to create. */
  readonly templateId: string;
  /** When set, data-plane TCP is forced to this IP (bypasses `*.cube.app` DNS). */
  readonly proxyNodeIp: string;
  /** HTTP port reached through {@link Config.proxyNodeIp}. Default `80`. */
  readonly proxyPortHttp: number;
  /** Data-plane scheme. Default `http`; `https` when {@link Config.proxyPortHttp} is `443`. */
  readonly proxyScheme: ProxyScheme;
  /** Virtual hostname suffix for sandboxes. Default `cube.app`. */
  readonly sandboxDomain: string;
  /** Whole-sandbox operation timeout in milliseconds. Default 300 000 (300s). */
  readonly timeoutMs: number;
  /** Per-HTTP-request timeout in milliseconds. Default 30 000 (30s). */
  readonly requestTimeoutMs: number;
}

// Default values — copied verbatim from `sdk/go/config.go:14-19`.
const DEFAULT_API_URL = "http://127.0.0.1:3000";
const DEFAULT_PROXY_PORT_HTTP = 80;
const DEFAULT_SANDBOX_DOMAIN = "cube.app";
const DEFAULT_TIMEOUT_MS = 300_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;

/**
 * Build an immutable {@link Config} from `process.env` (Go superset), with
 * optional explicit overrides winning over environment.
 *
 * Resolution order for a field: `overrides[field]` → `CUBE_*` → `E2B_*` → default.
 */
export const Config = {
  fromEnv(overrides: ConfigInput = {}): Config {
    const apiUrl = (overrides.apiUrl ?? firstEnv("CUBE_API_URL", "E2B_API_URL") ?? "").trim();
    const apiKey = overrides.apiKey ?? firstEnv("CUBE_API_KEY", "E2B_API_KEY");
    const templateId = (overrides.templateId ?? (process.env.CUBE_TEMPLATE_ID ?? "").trim()).trim();
    const proxyNodeIp = (
      overrides.proxyNodeIp ?? (process.env.CUBE_PROXY_NODE_IP ?? "").trim()
    ).trim();
    const proxyPortHttp =
      overrides.proxyPortHttp ?? parseIntEnv("CUBE_PROXY_PORT_HTTP", DEFAULT_PROXY_PORT_HTTP);
    const proxySchemeRaw = (
      overrides.proxyScheme ?? (process.env.CUBE_PROXY_SCHEME ?? "").trim()
    ).trim();
    const sandboxDomain = (
      overrides.sandboxDomain ?? (process.env.CUBE_SANDBOX_DOMAIN ?? "").trim()
    ).trim();
    const timeoutMs = overrides.timeoutMs ?? parseDurationMs("CUBE_TIMEOUT", DEFAULT_TIMEOUT_MS);
    const requestTimeoutMs =
      overrides.requestTimeoutMs ??
      parseDurationMs("CUBE_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_MS);

    const normalizedPort = proxyPortHttp > 0 ? proxyPortHttp : DEFAULT_PROXY_PORT_HTTP;

    return Object.freeze({
      apiUrl: apiUrl.replace(/\/+$/, "") || DEFAULT_API_URL,
      apiKey,
      templateId,
      proxyNodeIp,
      proxyPortHttp: normalizedPort,
      proxyScheme: normalizeProxyScheme(proxySchemeRaw, normalizedPort),
      sandboxDomain: sandboxDomain || DEFAULT_SANDBOX_DOMAIN,
      timeoutMs: timeoutMs > 0 ? timeoutMs : DEFAULT_TIMEOUT_MS,
      requestTimeoutMs:
        requestTimeoutMs > 0 ? requestTimeoutMs : DEFAULT_REQUEST_TIMEOUT_MS,
    });
  },
} as const;

// ─── env helpers ──────────────────────────────────────────────────────────

/** First non-empty (trimmed) value among the given env names (`CUBE_*` > `E2B_*`). */
function firstEnv(...names: string[]): string {
  for (const name of names) {
    const value = (process.env[name] ?? "").trim();
    if (value !== "") return value;
  }
  return "";
}

function parseIntEnv(name: string, fallback: number): number {
  const raw = (process.env[name] ?? "").trim();
  if (raw === "") return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.trunc(parsed);
}

/**
 * Parse a Go-style duration env var into milliseconds.
 *
 * Accepts (mirroring `sdk/go/config.go:108-121`):
 *  • a bare number → seconds, e.g. `"30"` → 30 000 ms (`"1.5"` → 1 500 ms).
 *  • a Go duration unit suffix: `ms`, `s`, `m`, `h`, e.g. `"300s"`, `"5m"`.
 * Falls back when unset or unparseable.
 */
function parseDurationMs(name: string, fallbackMs: number): number {
  const raw = (process.env[name] ?? "").trim();
  if (raw === "") return fallbackMs;
  // Bare numeric → seconds (Go semantics for `strconv.ParseFloat`).
  if (/^[0-9]+(\.[0-9]+)?$/.test(raw)) {
    return Math.round(Number(raw) * 1000);
  }
  const match = /^([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)$/.exec(raw);
  if (match) {
    const value = Number(match[1]);
    const multiplier =
      match[2] === "ms" ? 1 : match[2] === "s" ? 1000 : match[2] === "m" ? 60_000 : 3_600_000;
    return Math.round(value * multiplier);
  }
  return fallbackMs;
}

/** Go config.go:75-85 — explicit http/https wins; otherwise https on port 443, else http. */
function normalizeProxyScheme(raw: string, port: number): ProxyScheme {
  const lowered = raw.trim().toLowerCase();
  if (lowered === "http" || lowered === "https") return lowered;
  return port === 443 ? "https" : "http";
}
