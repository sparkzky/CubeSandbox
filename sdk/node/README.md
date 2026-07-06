# @cubesandbox/sdk

Official Node.js / TypeScript SDK for **CubeSandbox** — an E2B-compatible,
production-grade code-sandbox platform. Drop-in for E2B JS SDK callers: point
it at your CubeSandbox deployment and the `Sandbox.create()` / `runCode` /
`commands` / `files` surface just works.

> **Status (v0.1.0):** the v1 core surface is implemented and verified —
> `Sandbox` lifecycle (`create` / `connect` / `kill` / `getInfo` / `list` /
> `listV2` / `health`), `runCode` streaming, `commands.run`, and `files`
> (`read` / `write` / `list` / `stat`). Zero runtime dependencies, dual ESM +
> CommonJS, full TypeScript types. v2 capabilities (snapshots, templates, PTY,
> …) are [roadmapped](#roadmap), not yet shipped.

## Install

```bash
npm i @cubesandbox/sdk
# or
pnpm add @cubesandbox/sdk
# or
yarn add @cubesandbox/sdk
```

Requires Node.js 18+. The package has **zero runtime dependencies** and ships
dual ESM + CommonJS builds with generated `.d.ts`.

## Quickstart

Configure via environment variables (simplest), then create a sandbox and go:

```bash
export CUBE_API_URL=https://api.your-cube-deployment.example.com
export CUBE_API_KEY=ck_your_key
export CUBE_TEMPLATE_ID=base
# Optional: pin the data-plane network path (curl --resolve semantics):
# export CUBE_PROXY_NODE_IP=203.0.113.10
```

```ts
import { Sandbox } from "@cubesandbox/sdk";

// 1. Create a sandbox (reads CUBE_API_URL / CUBE_API_KEY / CUBE_TEMPLATE_ID).
const sb = await Sandbox.create();

// 2. Run code — stream stdout/stderr line-by-line, then read result.text.
const result = await sb.runCode(
  "for i in range(3):\n    print(i)\nsum([1, 2, 3])",
  {
    language: "python",
    onStdout: (msg) => console.log("[stdout]", msg.text),
    onStderr: (msg) => console.error("[stderr]", msg.text),
  },
);
console.log(result.text); // "6"  (text of the main result)

// 3. Run a shell command (envd executes `/bin/bash -l -c <cmd>`).
const cmd = await sb.commands.run("echo hello cube");
console.log(cmd.stdout.trim()); // "hello cube"
console.log(cmd.exitCode);      // 0

// 4. Write & read a file.
await sb.files.write("/tmp/hello.txt", "hello");
const file = await sb.files.read("/tmp/hello.txt");
console.log(file); // "hello"

// 5. Tear down.
await sb.kill();
```

Prefer to configure programmatically? Pass an explicit `Config` to any factory:

```ts
import { Sandbox } from "@cubesandbox/sdk";

// Flat, idiomatic camelCase config overrides at the top level (issue #760 shape).
const sb = await Sandbox.create({
  apiUrl: "https://api.your-cube-deployment.example.com",
  apiKey: process.env.MY_KEY!,
  templateId: "base",
  proxyNodeIp: "203.0.113.10", // route data-plane TCP here (see Configuration)
});
```

## Configuration

All settings are read from environment variables (the Go SDK superset). `CUBE_*`
always takes precedence over the `E2B_*` alias — so existing E2B deployments keep
working until you migrate.

| Variable               | Default                  | Description                                                                              |
| ---------------------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `CUBE_API_URL`         | `http://127.0.0.1:3000`  | Control-plane base URL. Alias: `E2B_API_URL`.                                            |
| `CUBE_API_KEY`         | _(empty)_                | API key sent as `Authorization: Bearer <key>`. Alias: `E2B_API_KEY`.                     |
| `CUBE_TEMPLATE_ID`     | _(empty)_                | Default template used when `create()` is called without `templateId`.                    |
| `CUBE_PROXY_PORT_HTTP` | `80`                     | Port reached through `CUBE_PROXY_NODE_IP`.                                               |
| `CUBE_PROXY_SCHEME`    | `http` (or `https` on 443) | Data-plane scheme.                                                                       |
| `CUBE_SANDBOX_DOMAIN`  | `cube.app`               | Virtual hostname suffix for sandboxes (`{port}-{sandboxID}.{domain}`).                   |
| `CUBE_TIMEOUT`         | `300s`                   | Default sandbox TTL. Accepts bare seconds (`600`), `600s`, `10m`.                        |
| `CUBE_REQUEST_TIMEOUT` | `30s`                    | Per-HTTP-request timeout.                                                                |

> **About `CUBE_PROXY_NODE_IP`:** this is a _network-routing_ override, not
> authentication. It is the equivalent of `curl --resolve`: the TCP connection
> is forced to the given IP:port while the `Host` header (and TLS SNI) keep the
> virtual hostname `{port}-{sandboxID}.{domain}` so CubeProxy routes correctly.
> Use it when your network cannot resolve `*.cube.app` DNS. Auth still flows
> through `CUBE_API_KEY` (control plane) and the per-sandbox tokens issued at
> create time (data plane).

Load or override a config explicitly:

```ts
import { Config } from "@cubesandbox/sdk";

const cfg = Config.fromEnv();                              // reads process.env
const overridden = Config.fromEnv({ apiKey: "ck_other" }); // explicit wins over env
```

## API reference

### `Sandbox` — lifecycle

```ts
// Create a new sandbox. `templateId` is optional (falls back to CUBE_TEMPLATE_ID);
// every other top-level field is either a CreateOption or a flat Config override.
static create(opts?: {
  templateId?: string;
  timeoutMs?: number;            // sandbox TTL (ms); default CUBE_TIMEOUT
  envVars?: Record<string, string>;
  metadata?: Record<string, string>;
  allowInternetAccess?: boolean;  // default true; false restricts egress
  network?: NetworkOptions;
  extra?: Record<string, unknown>; // forward-compat escape hatch
  // flat Config overrides (undefined ⇒ inherit CUBE_* / default):
  apiUrl?: string;
  apiKey?: string;
  proxyNodeIp?: string;
  proxyPortHttp?: number;
  proxyScheme?: "http" | "https";
  sandboxDomain?: string;
  requestTimeoutMs?: number;
}): Promise<Sandbox>;

// connect / list / listV2 / health take the same flat Config overrides:
static connect(sandboxID: string, opts?: SandboxConfigOverride): Promise<Sandbox>;
static list(opts?: SandboxConfigOverride): Promise<SandboxInfo[]>;
static listV2(opts?: SandboxConfigOverride): Promise<SandboxInfo[]>;   // server-side filtering
static health(opts?: SandboxConfigOverride): Promise<Record<string, unknown>>;

kill(): Promise<void>;           // DELETE /sandboxes/:id
getInfo(): Promise<SandboxInfo>; // GET /sandboxes/:id
getHost(port: number): string;   // virtual host, e.g. 49999-<id>.cube.app
```

### `sandbox.runCode` — code execution (Jupyter `/execute`, port 49999)

```ts
const exec = await sb.runCode(code: string, opts?: {
  language?: string;                                    // default null (kernel default)
  envs?: Record<string, string>;
  timeoutMs?: number;                                   // absolute cap on the stream
  onStdout?: (msg: OutputMessage) => void;              // msg.text, msg.timestamp?
  onStderr?: (msg: OutputMessage) => void;              // msg.error === true
  onResult?: (result: Result) => void;
  onError?: (error: ExecutionError) => void;
}): Promise<Execution>;
```

`Execution` aggregates the whole stream:

```ts
interface Execution {
  results: Result[];            // rich cell results (text, html, json, png, …)
  logs: { stdout: string[]; stderr: string[] };
  error: ExecutionError | null; // { name, value, traceback }
  executionCount: number | null;
  text?: string;                // convenience: text of the main result
}
```

### `sandbox.commands` — shell commands (envd `process.Process/Start`, port 49983)

```ts
const cmd = await sb.commands.run(cmd: string, opts?: {
  cwd?: string;
  envs?: Record<string, string>;
  env?: Record<string, string>;   // alias of envs (E2B parity)
  user?: string;                   // default "root"
  timeoutMs?: number;
}): Promise<{ stdout: string; stderr: string; exitCode: number }>;
```

### `sandbox.files` — filesystem (envd `/files` + `filesystem.Filesystem/*`)

```ts
await sb.files.write(path: string, data: string | Uint8Array, opts?: { user?: string }): Promise<void>;
const text: string = await sb.files.read(path: string, opts?: { user?: string });
const entries: FileEntry[] = await sb.files.list(path: string); // directory listing
const entry: FileEntry = await sb.files.stat(path: string);     // single entry
```

`files.write` first tries an `application/octet-stream` upload and transparently
falls back to `multipart/form-data` for older envd versions. `list` / `stat` use
Connect-JSON unary calls (`{ entries: […] }` / `{ entry: {…} }`).

## Migrating from the E2B JS SDK

CubeSandbox is API-compatible with the E2B JS SDK surface (`Sandbox.create`,
`sandbox.runCode`, `sandbox.commands.run`, `sandbox.files.*`). To migrate an
existing E2B app, change two environment variables and you are done:

```bash
export CUBE_API_URL=https://api.your-cube-deployment.example.com
export CUBE_API_KEY=ck_your_key
# Your existing E2B_API_URL / E2B_API_KEY still work as aliases until you switch.
```

No code changes are required for the v1 surface; the same call shapes (including
the `msg.text` callback field on streamed output) carry over.

## Errors

Every failure the SDK raises descends from `CubeSandboxError`, so a single
`instanceof` check catches them all:

```ts
import {
  CubeSandboxError,
  AuthenticationError,    // HTTP 401 / 403
  SandboxNotFoundError,   // HTTP 404 (non-template)
  TemplateNotFoundError,  // HTTP 404 mentioning "template"
  ApiError,               // any other non-2xx, plus protocol-level failures
} from "@cubesandbox/sdk";

try {
  await sb.runCode("1/0");
} catch (err) {
  if (err instanceof CubeSandboxError) {
    console.error(err.statusCode, err.name, err.message);
  } else {
    throw err; // unexpected (network, programmer error, …)
  }
}
```

The classifier applies uniformly across the control plane (`create` / `connect` /
…) and the data plane (`runCode` / `commands` / `files`): 401/403 →
`AuthenticationError`, 404 → `SandboxNotFoundError` (or `TemplateNotFoundError`
when the message mentions a template), and everything else → `ApiError`. A
malformed envd response (e.g. a command stream that ends without an exit event)
is also raised as an `ApiError`, so it stays inside the same hierarchy.

## Roadmap

The following are **not yet implemented** in the v1 surface (tracked for v2):

- [ ] Snapshot / rollback / clone of sandbox disk + memory state
- [ ] Template build & management (`Template.build`, list, delete)
- [ ] PTY sessions (`sandbox.pty`)
- [ ] Network policy (L7 egress rules, credential injection, audit)
- [ ] `watch_dir` filesystem change streaming
- [ ] Host-mount / volume wiring beyond the create-time declaration
- [ ] Browser sandbox helpers

## License

Apache-2.0. See [LICENSE](./LICENSE).
