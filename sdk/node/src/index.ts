// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Public surface of the CubeSandbox Node.js / TypeScript SDK.
 *
 * Foundation layer (config, errors, types, transport, envd Connect codec) is
 * re-exported alongside the v1 core surface: the {@link Sandbox} class and the
 * data-plane namespaces (commands, files, runCode).
 */

export * from "./config.js";
export * from "./errors.js";
export * from "./types.js";
export * from "./transport.js";
export * from "./envd-codec.js";

// ─── v1 core surface ──────────────────────────────────────────────────────
export { Sandbox } from "./sandbox.js";
export type {
  SandboxConfigOverride,
  SandboxCreateOptions,
} from "./sandbox.js";
export { Commands } from "./commands.js";
export type { CommandOptions, CommandsContext } from "./commands.js";
export { Files } from "./filesystem.js";
export type { FilesContext, FileUserOptions } from "./filesystem.js";
export { runCode } from "./code-execution.js";
export type { RunCodeAuth } from "./code-execution.js";
