import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "es2022",
  outDir: "dist",
  // Keep Node builtins external; runtime has zero dependencies.
  // NodeNext ESM requires .js specifiers in source — tsup rewrites them on bundle.
  platform: "node",
});
