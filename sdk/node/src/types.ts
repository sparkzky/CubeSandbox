// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Core wire-shape models for the CubeSandbox SDK.
 *
 * Field names use camelCase (TypeScript / JSON wire convention). The backend
 * returns the same camelCase keys for these models (see `CubeAPI/src/models/mod.rs`
 * `#[serde(rename = ...)]` mappings), so no transformation is needed at the
 * Foundation layer. Snake_case ↔ camelCase bridging for legacy endpoints is a
 * Core-stage concern.
 *
 * Shapes are derived from `sdk/python/cubesandbox/_models.py` and
 * `sdk/go/models.go`.
 */

// ─── Jupyter `/execute` (port 49999) ──────────────────────────────────────

export interface Logs {
  stdout: string[];
  stderr: string[];
}

export interface ExecutionError {
  name: string;
  value: string;
  /** Traceback as a single string or stacked frames. */
  traceback: string | string[];
}

/** A single rich result from executing a cell. `isMainResult` flags the last expression. */
export interface Result {
  text?: string | null;
  html?: string | null;
  markdown?: string | null;
  svg?: string | null;
  png?: string | null;
  jpeg?: string | null;
  pdf?: string | null;
  latex?: string | null;
  /** JSON value of the result. Serialized as `json_data` on the wire. */
  json?: Record<string, unknown> | null;
  javascript?: string | null;
  /** Raw structured payload for formats not covered above. */
  data?: Record<string, unknown> | null;
  chart?: unknown;
  isMainResult?: boolean;
  extra?: Record<string, unknown> | null;
}

export interface Execution {
  results: Result[];
  logs: Logs;
  error: ExecutionError | null;
  executionCount: number | null;
  /** Convenience: text of the main result. Populated by Core when assembling. */
  text?: string;
}

/** One streamed line from a sandbox stdout/stderr stream. */
export interface OutputMessage {
  /** The line text. Canonical E2B JS SDK field name (`msg.text`). */
  text: string;
  timestamp?: number | string;
  /** True when the line came from stderr. */
  error?: boolean;
}

// ─── envd process commands (port 49983) ───────────────────────────────────

export interface CommandResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

// ─── envd filesystem (port 49983) ─────────────────────────────────────────

/**
 * A file or directory entry returned by envd filesystem RPCs.
 * A directory is identified by `type === "FILE_TYPE_DIRECTORY"`.
 */
export interface FileEntry {
  name: string;
  type: string;
  path: string;
  size: number;
  mode: number;
  permissions: string;
  owner: string;
  group: string;
  modifiedTime: string;
}

/** `FileEntry.type` value used by envd to denote a directory. */
export const FILE_TYPE_DIRECTORY = "FILE_TYPE_DIRECTORY";

/** True when the entry is a directory (`type === FILE_TYPE_DIRECTORY`). */
export function isDirectory(entry: FileEntry): boolean {
  return entry.type === FILE_TYPE_DIRECTORY;
}

// ─── control-plane models ─────────────────────────────────────────────────

export interface VolumeMount {
  name: string;
  path: string;
}

/** L3/L4 + L7 network policy applied to a sandbox. */
export interface NetworkOptions {
  allowPublicTraffic?: boolean | null;
  allowOut?: string[];
  denyOut?: string[];
  rules?: unknown[];
}

export interface CreateOptions {
  templateId: string;
  /** Sandbox lifetime in milliseconds. */
  timeoutMs?: number;
  envVars?: Record<string, string>;
  metadata?: Record<string, string>;
  allowInternetAccess?: boolean | null;
  network?: NetworkOptions;
  /** Escape hatch for forward-compatible fields. */
  extra?: Record<string, unknown>;
}

export interface RunCodeOptions {
  language?: string;
  envs?: Record<string, string>;
  timeoutMs?: number;
  onStdout?: (message: OutputMessage) => void;
  onStderr?: (message: OutputMessage) => void;
  onResult?: (result: Result) => void;
  onError?: (error: ExecutionError) => void;
}

/** Running-sandbox record returned by `GET /sandboxes` and `GET /sandboxes/{id}`. */
export interface SandboxInfo {
  templateID: string;
  alias?: string;
  sandboxID: string;
  clientID: string;
  startedAt: string;
  endAt: string;
  envdVersion: string;
  domain?: string;
  cpuCount: number;
  memoryMB: number;
  diskSizeMB?: number;
  metadata?: Record<string, string>;
  state: string;
  volumeMounts?: VolumeMount[];
}

/**
 * Body returned by `POST /sandboxes` (the "create sandbox" control-plane call).
 *
 * Mirrors `CubeAPI/src/models/mod.rs:227-251` (`Sandbox` struct). Data-plane
 * credentials (`envdAccessToken`, `trafficAccessToken`) are issued here and
 * consumed by Core when talking to envd.
 */
export interface SandboxCreateResponse {
  templateID: string;
  sandboxID: string;
  alias?: string;
  clientID: string;
  envdVersion: string;
  /** envd RPC auth (`X-Access-Token`). Present when envd requires it. */
  envdAccessToken?: string;
  /** Restricted-sandbox public-traffic auth (`e2b-traffic-access-token`). */
  trafficAccessToken?: string;
  domain?: string;
}

/** Metadata returned by snapshot-related control-plane endpoints. */
export interface SnapshotInfo {
  snapshotID: string;
  names: string[];
}
