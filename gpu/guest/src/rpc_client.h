/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * RPC client: sends CUDA API calls from guest to host via virtio-vsock.
 */

#ifndef CUBESANDBOX_GPU_RPC_CLIENT_H
#define CUBESANDBOX_GPU_RPC_CLIENT_H

#include "rpc_protocol.h"
#include <stdint.h>
#include <stddef.h>

typedef struct cube_gpu_rpc_client {
    int vsock_fd;
    uint64_t session_id;
    uint64_t next_request_id;
    int initialized;
} cube_gpu_rpc_client_t;

int  cube_gpu_rpc_client_init(cube_gpu_rpc_client_t *client, uint32_t host_cid, uint32_t port, uint64_t sandbox_id, uint64_t memory_quota);
void cube_gpu_rpc_client_destroy(cube_gpu_rpc_client_t *client);

int  cube_gpu_rpc_send(cube_gpu_rpc_client_t *client, const cube_gpu_rpc_header_t *hdr, const void *payload);
int  cube_gpu_rpc_recv(cube_gpu_rpc_client_t *client, cube_gpu_rpc_header_t *hdr_out, void *payload_out, size_t payload_buf_size);

#endif
