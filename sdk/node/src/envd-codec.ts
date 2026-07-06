// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Connect-JSON envelope codec used by envd RPCs (commands / files).
 *
 * Framing (mirrors `sdk/python/cubesandbox/_commands.py:265-266` and
 * `sdk/go/envd.go`): every envelope is
 *
 *     ┌─────────┬─────────────────────────────┬──────────────────┐
 *     │ 1 byte  │        4 bytes              │     N bytes      │
 *     │ flags   │ big-endian body length (N)  │   JSON body      │
 *     └─────────┴─────────────────────────────┴──────────────────┘
 *
 * Flag bits (Connect spec, matches Python `_commands.py:20-21`):
 *   • `0x01` — payload is compressed (unsupported by the SDK; throws on decode).
 *   • `0x02` — end-of-stream marker (streaming only).
 */

/** Maximum body length we will accept when decoding (64 MiB, matches Python). */
export const MAX_CONNECT_ENVELOPE_SIZE = 64 * 1024 * 1024;

/** Connect streaming flag: payload is compressed. */
export const CONNECT_COMPRESSED_FLAG = 0x01;
/** Connect streaming flag: this is the terminal (end-stream) frame. */
export const CONNECT_END_STREAM_FLAG = 0x02;

/** `Content-Type` for Connect-JSON requests (envd RPCs). */
export const CONNECT_CONTENT_TYPE = "application/connect+json";
/** Connect protocol version header value. */
export const CONNECT_PROTOCOL_VERSION = "1";

const HEADER_SIZE = 5;

/** One decoded Connect streaming frame. */
export interface ConnectFrame {
  /** Raw flag byte. */
  flags: number;
  /** Parsed JSON body (or `undefined` for empty bodies). */
  data: unknown;
  /** True when {@link CONNECT_END_STREAM_FLAG} is set. */
  endStream: boolean;
}

/**
 * Encode a single Connect envelope (unary request/response or one stream frame).
 *
 * @param data JSON-serializable payload.
 * @param flags flag byte (default `0`, uncompressed).
 * @returns a `Uint8Array` of `5 + body.length` bytes.
 */
export function encodeConnectUnary(data: unknown, flags = 0): Uint8Array {
  const body = new TextEncoder().encode(JSON.stringify(data));
  const out = new Uint8Array(HEADER_SIZE + body.byteLength);
  out[0] = flags & 0xff;
  new DataView(out.buffer).setUint32(1, body.byteLength, false); // big-endian
  out.set(body, HEADER_SIZE);
  return out;
}

/**
 * Decode a single Connect envelope from a buffer (unary RPC response).
 *
 * Throws on truncated frames and on compressed frames (`0x01`).
 */
export function decodeConnectUnary(buffer: Uint8Array): unknown {
  if (buffer.byteLength < HEADER_SIZE) {
    throw new Error(`Connect unary frame too short: ${buffer.byteLength} bytes`);
  }
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const flags = view.getUint8(0);
  const length = view.getUint32(1, false);
  if (buffer.byteLength < HEADER_SIZE + length) {
    throw new Error(
      `Connect unary frame truncated: declared ${length} bytes, have ${buffer.byteLength - HEADER_SIZE}`,
    );
  }
  if (length > MAX_CONNECT_ENVELOPE_SIZE) {
    throw new Error(`Connect frame too large: ${length} bytes`);
  }
  if ((flags & CONNECT_COMPRESSED_FLAG) !== 0) {
    throw new Error("Compressed Connect frames are not supported");
  }
  const body = buffer.subarray(HEADER_SIZE, HEADER_SIZE + length);
  if (body.byteLength === 0) return undefined;
  return JSON.parse(new TextDecoder().decode(body));
}

/**
 * Decode a Connect-RPC byte stream into parsed frames.
 *
 * Accepts either a web `ReadableStream<Uint8Array>` or any async iterable of
 * `Uint8Array`/`Buffer` chunks (e.g. the `body` of a `fetch` Response). Frames
 * may be split across chunks or multiple frames may share a chunk; boundaries
 * are buffered until a full envelope is available.
 *
 * Throws on compressed frames, oversized frames, or a trailing partial frame.
 */
export async function* decodeConnectStream(
  stream: ReadableStream<Uint8Array> | AsyncIterable<Uint8Array>,
): AsyncGenerator<ConnectFrame> {
  let buffer = new Uint8Array(0);

  for await (const rawChunk of stream as AsyncIterable<Uint8Array>) {
    const chunk = rawChunk instanceof Uint8Array ? rawChunk : new Uint8Array(rawChunk);
    if (chunk.byteLength === 0) continue;

    const merged = new Uint8Array(buffer.byteLength + chunk.byteLength);
    merged.set(buffer, 0);
    merged.set(chunk, buffer.byteLength);
    buffer = merged;

    while (buffer.byteLength >= HEADER_SIZE) {
      const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
      const flags = view.getUint8(0);
      const length = view.getUint32(1, false);
      if (length > MAX_CONNECT_ENVELOPE_SIZE) {
        throw new Error(`Connect stream message too large: ${length} bytes`);
      }
      if (buffer.byteLength < HEADER_SIZE + length) break;

      const body = buffer.subarray(HEADER_SIZE, HEADER_SIZE + length);
      buffer = buffer.subarray(HEADER_SIZE + length);

      if ((flags & CONNECT_COMPRESSED_FLAG) !== 0) {
        throw new Error("Compressed Connect stream messages are not supported");
      }
      const data = body.byteLength === 0 ? undefined : JSON.parse(new TextDecoder().decode(body));
      yield { flags, data, endStream: (flags & CONNECT_END_STREAM_FLAG) !== 0 };
    }
  }

  if (buffer.byteLength > 0) {
    throw new Error(`Connect stream ended with a partial frame: ${buffer.byteLength} bytes`);
  }
}
