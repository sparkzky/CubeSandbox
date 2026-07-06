// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Internal helpers shared by the data-plane modules (runCode, commands,
 * filesystem). Not part of the public SDK surface.
 */

/** Default envd user (Python `_commands.py:23`, Go `envd.go:29`). */
export const DEFAULT_ENVD_USER = "root";

/** Header injected on data-plane requests for restricted sandboxes. */
export const TRAFFIC_TOKEN_HEADER = "e2b-traffic-access-token";

/** `unknown` → `Record<string, unknown>` type guard for parsed JSON shapes. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

/**
 * Best-effort human message from a data-plane error response body.
 *
 * Prefers `message`, then `detail`, then a nested `error.message`; falls back
 * to the raw body. Shared by runCode/commands/files so error classification
 * sees a consistent string regardless of which envd endpoint produced it.
 */
export async function readErrorMessage(res: Response): Promise<string> {
  let text: string;
  try {
    text = await res.text();
  } catch {
    return "";
  }
  const trimmed = text.trim();
  if (trimmed === "") return "";

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
  if (!isRecord(parsed)) return trimmed;

  const message = parsed["message"];
  if (typeof message === "string") return message;
  const detail = parsed["detail"];
  if (typeof detail === "string") return detail;
  const err = parsed["error"];
  if (isRecord(err)) {
    const nested = err["message"];
    if (typeof nested === "string") return nested;
  }
  return trimmed;
}
