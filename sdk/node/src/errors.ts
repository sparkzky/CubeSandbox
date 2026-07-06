// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * CubeSandbox error hierarchy.
 *
 * Mirrors `sdk/python/cubesandbox/_exceptions.py` and the classification logic
 * in `sdk/python/cubesandbox/sandbox.py:29-41` / `sdk/go/errors.go`.
 * There is no numeric error-code system; subclasses are distinguished by kind.
 */

/** Base class for every error raised by the SDK. */
export class CubeSandboxError extends Error {
  /** HTTP status code that produced this error, when applicable. */
  readonly statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = this.constructor.name;
    if (statusCode !== undefined) this.statusCode = statusCode;
    // Restore the prototype chain — required when extending built-ins under
    // ES5/ES2015 targets so `instanceof` keeps working.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** The referenced sandbox does not exist (HTTP 404 without "template"). */
export class SandboxNotFoundError extends CubeSandboxError {}

/** The referenced template does not exist (HTTP 404 whose message contains "template"). */
export class TemplateNotFoundError extends CubeSandboxError {}

/** Authentication or authorization failed (HTTP 401 / 403). */
export class AuthenticationError extends CubeSandboxError {}

/** Generic API error for any non-2xx response not covered above. */
export class ApiError extends CubeSandboxError {}

/**
 * Map an HTTP failure to the matching SDK error subclass.
 *
 * Classification (matches `sandbox.py:37-41`):
 *  • 401 / 403 → {@link AuthenticationError}
 *  • 404 + message contains "template" → {@link TemplateNotFoundError}
 *  • 404 otherwise → {@link SandboxNotFoundError}
 *  • everything else → {@link ApiError}
 */
export function classifyHttpError(status: number, message: string): CubeSandboxError {
  if (status === 401 || status === 403) {
    return new AuthenticationError(message, status);
  }
  if (status === 404) {
    return /template/i.test(message)
      ? new TemplateNotFoundError(message, status)
      : new SandboxNotFoundError(message, status);
  }
  return new ApiError(message, status);
}
