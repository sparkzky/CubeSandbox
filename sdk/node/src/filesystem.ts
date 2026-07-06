// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * files — read, write, list and stat files inside a sandbox via envd's
 * filesystem API on data-plane port {@link ENVD_PORT} (49983).
 *
 * Wire format (grounded in `sdk/python/cubesandbox/_filesystem.py` and
 * `sdk/go/envd.go` / `sdk/go/files.go`):
 *
 *  • read:  `GET /files?path=<p>&username=<u>` → response body is the file text.
 *  • write: `POST /files?path=<p>&username=<u>` with `application/octet-stream`
 *    body. If envd rejects that (older versions), retry once as
 *    `multipart/form-data` (field `file`, filename = path) — Python
 *    `_filesystem.py:106-113`, Go `envd.go:181-188`.
 *  • list / stat: `POST /filesystem.Filesystem/{ListDir,Stat}` with a plain
 *    JSON body `{ "path": <p> }` and `Content-Type: application/json`.
 *    The response is plain JSON too: `{ "entries": [...] }` / `{ "entry": {...} }`.
 *
 *    NOTE: ListDir/Stat are Connect-JSON *unary* calls — the request and
 *    response bodies are raw JSON, NOT envelope-framed. (Both Python
 *    `_filesystem.py:40-66` and Go `envd.go:380-405` confirm this; the
 *    `application/connect+json` envelope codec is reserved for streaming RPCs
 *    such as `process.Process/Start` and `WatchDir`.) The assignment brief's
 *    "用 envd-codec 编解码 envelope" hint applies to commands, not to these
 *    unary filesystem calls.
 *
 * Auth headers: `X-Access-Token: <envdAccessToken>` on every call; restricted
 * sandboxes additionally send `e2b-traffic-access-token`.
 */

import type { DataClient } from "./transport.js";
import { ENVD_PORT } from "./transport.js";
import { CONNECT_PROTOCOL_VERSION } from "./envd-codec.js";
import { classifyHttpError } from "./errors.js";
import type { FileEntry } from "./types.js";
import {
  DEFAULT_ENVD_USER,
  isRecord,
  readErrorMessage,
  TRAFFIC_TOKEN_HEADER,
} from "./internal.js";

/** Options for {@link Files.read} / {@link Files.write}. */
export interface FileUserOptions {
  /** envd filesystem user; defaults to `root`. */
  readonly user?: string;
}

/** Per-sandbox credentials the files namespace forwards to envd. */
export interface FilesContext {
  readonly dataClient: DataClient;
  readonly sandboxID: string;
  readonly envdAccessToken?: string;
  readonly trafficAccessToken?: string;
}

/** Read, write, list and stat files in a sandbox through envd. */
export class Files {
  constructor(private readonly ctx: FilesContext) {}

  /** Read a file as utf-8 text (`GET /files`). */
  async read(path: string, opts: FileUserOptions = {}): Promise<string> {
    const query = filesQuery(path, opts.user ?? DEFAULT_ENVD_USER);
    const res = await this.ctx.dataClient.request(ENVD_PORT, this.ctx.sandboxID, `/files${query}`, {
      method: "GET",
      headers: this.baseHeaders(),
    });
    if (res.status >= 400) {
      const message = await readErrorMessage(res);
      throw classifyHttpError(
        res.status,
        message || `failed to read ${path}: HTTP ${res.status}`,
      );
    }
    return res.text();
  }

  /**
   * Write a file (`POST /files`).
   *
   * First tries an octet-stream upload; on HTTP 4xx/5xx retries once as
   * multipart/form-data (older envd versions reject raw octet-stream).
   */
  async write(path: string, data: string | Uint8Array, opts: FileUserOptions = {}): Promise<void> {
    const query = filesQuery(path, opts.user ?? DEFAULT_ENVD_USER);
    const body = typeof data === "string" ? new TextEncoder().encode(data) : data;

    let res = await this.ctx.dataClient.request(ENVD_PORT, this.ctx.sandboxID, `/files${query}`, {
      method: "POST",
      headers: { ...this.baseHeaders(), "Content-Type": "application/octet-stream" },
      body,
    });

    if (res.status >= 400) {
      // Multipart fallback — recreate the envd-accepted form shape.
      const { body: multipartBody, contentType } = multipartFileBody(path, body);
      res = await this.ctx.dataClient.request(ENVD_PORT, this.ctx.sandboxID, `/files${query}`, {
        method: "POST",
        headers: { ...this.baseHeaders(), "Content-Type": contentType },
        body: multipartBody,
      });
    }

    if (res.status >= 400) {
      const message = await readErrorMessage(res);
      throw classifyHttpError(
        res.status,
        message || `failed to write ${path}: HTTP ${res.status}`,
      );
    }
  }

  /** List entries in a directory (`filesystem.Filesystem/ListDir`). */
  async list(path: string): Promise<FileEntry[]> {
    const result = await this.filesystemRPC("ListDir", { path });
    const entries = result["entries"];
    if (!Array.isArray(entries)) return [];
    return entries.map(parseFileEntry);
  }

  /** Return metadata for a single file or directory (`filesystem.Filesystem/Stat`). */
  async stat(path: string): Promise<FileEntry> {
    const result = await this.filesystemRPC("Stat", { path });
    const entry = result["entry"];
    if (!isRecord(entry)) {
      throw new Error(`Stat returned no entry for ${path}`);
    }
    return parseFileEntry(entry);
  }

  /** Auth headers attached to every envd filesystem call. */
  private baseHeaders(): Record<string, string> {
    const headers: Record<string, string> = {};
    if (this.ctx.envdAccessToken) headers["X-Access-Token"] = this.ctx.envdAccessToken;
    if (this.ctx.trafficAccessToken) headers[TRAFFIC_TOKEN_HEADER] = this.ctx.trafficAccessToken;
    return headers;
  }

  /**
   * Invoke a Connect-JSON unary filesystem RPC.
   *
   * Body and response are plain JSON (see module note) — no envelope framing.
   */
  private async filesystemRPC(
    method: string,
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const res = await this.ctx.dataClient.request(
      ENVD_PORT,
      this.ctx.sandboxID,
      `/filesystem.Filesystem/${method}`,
      {
        method: "POST",
        headers: {
          ...this.baseHeaders(),
          "Content-Type": "application/json",
          "Connect-Protocol-Version": CONNECT_PROTOCOL_VERSION,
        },
        body: JSON.stringify(payload),
      },
    );

    if (res.status >= 400) {
      const message = await readErrorMessage(res);
      throw classifyHttpError(
        res.status,
        message || `Filesystem ${method} failed: HTTP ${res.status}`,
      );
    }

    const text = await res.text();
    if (text === "") return {};
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return {};
    }
    return isRecord(parsed) ? parsed : {};
  }
}

/** Build the `?path=&username=` query string for the `/files` endpoints. */
function filesQuery(path: string, user: string): string {
  return `?path=${encodeURIComponent(path)}&username=${encodeURIComponent(user)}`;
}

/**
 * Build a `multipart/form-data` body carrying a single `file` part.
 *
 * Mirrors Go `multipartFileBody` (`envd.go:209-223`) and Python's
 * `files={"file": (path, body)}` httpx shorthand.
 */
function multipartFileBody(
  path: string,
  data: Uint8Array,
): { body: Uint8Array; contentType: string } {
  const boundary = `cube-node-${Math.random().toString(36).slice(2)}`;
  const safeName = path.replace(/"/g, "");
  const head =
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="file"; filename="${safeName}"\r\n` +
    `Content-Type: application/octet-stream\r\n\r\n`;
  const tail = `\r\n--${boundary}--\r\n`;

  const headBytes = new TextEncoder().encode(head);
  const tailBytes = new TextEncoder().encode(tail);
  const body = new Uint8Array(headBytes.byteLength + data.byteLength + tailBytes.byteLength);
  body.set(headBytes, 0);
  body.set(data, headBytes.byteLength);
  body.set(tailBytes, headBytes.byteLength + data.byteLength);
  return { body, contentType: `multipart/form-data; boundary=${boundary}` };
}

/**
 * Map an envd filesystem entry (wire JSON) to a {@link FileEntry}.
 *
 * `size` is serialised as a JSON string by envd (Go uses `json:"size,string`);
 * we coerce both string and number forms. `modifiedTime` is the camelCase
 * wire key (Go `models.go:153`).
 */
function parseFileEntry(raw: unknown): FileEntry {
  const e = isRecord(raw) ? raw : {};
  return {
    name: stringField(e["name"]),
    type: stringField(e["type"]),
    path: stringField(e["path"]),
    size: numberField(e["size"]),
    mode: numberField(e["mode"]),
    permissions: stringField(e["permissions"]),
    owner: stringField(e["owner"]),
    group: stringField(e["group"]),
    modifiedTime: stringField(e["modifiedTime"]),
  };
}

function stringField(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numberField(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value !== "") {
    const n = Number(value);
    if (!Number.isNaN(n)) return n;
  }
  return 0;
}
