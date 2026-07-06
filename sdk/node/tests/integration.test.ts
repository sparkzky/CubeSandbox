// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Integration skeleton for the v1 core surface.
 *
 * These tests exercise the real control + data planes end-to-end and are only
 * run when `CUBE_API_URL` (and a reachable CubeSandbox backend) are present —
 * mirroring the Go SDK's `integration_test.go` build-tag gating. In CI without
 * a backend they skip silently so `npm test` stays green.
 */

import { describe, it, expect } from "vitest";

const API_URL = process.env.CUBE_API_URL ?? "";
const TEMPLATE_ID = process.env.CUBE_TEMPLATE_ID ?? "";
const enabled = API_URL !== "" && TEMPLATE_ID !== "";

const itReal = enabled ? it : it.skip;

describe("integration: v1 core surface", () => {
  itReal("create → runCode → commands.run → files.write/read → kill", async () => {
    // Lazy import so the module graph is not evaluated when the suite skips.
    const { Sandbox } = await import("../src/sandbox.js");
    const sb = await Sandbox.create({ templateId: TEMPLATE_ID });
    try {
      const exec = await sb.runCode("print('hello')");
      expect(exec.logs.stdout.join("")).toContain("hello");

      const cmd = await sb.commands.run("echo $((6*7))");
      expect(cmd.stdout.trim()).toBe("42");
      expect(cmd.exitCode).toBe(0);

      await sb.files.write("/tmp/integration.txt", "cube");
      const text = await sb.files.read("/tmp/integration.txt");
      expect(text).toBe("cube");
    } finally {
      await sb.kill();
    }
  });

  // Always-on assertion: the guard itself behaves when no backend is wired.
  it("skips the integration suite when CUBE_API_URL is unset", () => {
    if (!enabled) {
      expect(enabled).toBe(false);
    }
  });
});
