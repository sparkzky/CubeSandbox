/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * RPC client implementation: vsock transport for CUDA API remoting.
 */

#define _GNU_SOURCE
#include "rpc_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>

static int vsock_connect(uint32_t cid, uint32_t port) {
    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_vm addr;
    memset(&addr, 0, sizeof(addr));
    addr.svm_family = AF_VSOCK;
    addr.svm_cid = cid;
    addr.svm_port = port;

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = read(fd, p, remaining);
        if (n <= 0) {
            if (n == 0) return -1;
            if (errno == EINTR) continue;
            return -1;
        }
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

int cube_gpu_rpc_client_init(cube_gpu_rpc_client_t *client, uint32_t host_cid, uint32_t port,
                             uint64_t sandbox_id, uint64_t memory_quota) {
    memset(client, 0, sizeof(*client));

    client->vsock_fd = vsock_connect(host_cid, port);
    if (client->vsock_fd < 0) return -1;

    client->session_id = sandbox_id;
    client->next_request_id = 1;
    client->initialized = 0;

    cube_gpu_session_init_req_t req;
    memset(&req, 0, sizeof(req));
    req.memory_quota = memory_quota;
    req.sandbox_id = sandbox_id;

    cube_gpu_rpc_header_t hdr;
    cube_gpu_init_header(&hdr, CUBE_GPU_API_SESSION_INIT, client->next_request_id++,
                         client->session_id, sizeof(req), CUBE_GPU_FLAG_NONE);

    if (cube_gpu_rpc_send(client, &hdr, &req) < 0) {
        close(client->vsock_fd);
        client->vsock_fd = -1;
        return -1;
    }

    cube_gpu_rpc_header_t resp_hdr;
    cube_gpu_session_init_resp_t resp;
    memset(&resp, 0, sizeof(resp));

    if (cube_gpu_rpc_recv(client, &resp_hdr, &resp, sizeof(resp)) < 0) {
        close(client->vsock_fd);
        client->vsock_fd = -1;
        return -1;
    }

    if (resp.status != 0) {
        close(client->vsock_fd);
        client->vsock_fd = -1;
        return -1;
    }

    client->session_id = resp.session_handle;
    client->initialized = 1;
    return 0;
}

void cube_gpu_rpc_client_destroy(cube_gpu_rpc_client_t *client) {
    if (client->vsock_fd >= 0) {
        cube_gpu_rpc_header_t hdr;
        cube_gpu_init_header(&hdr, CUBE_GPU_API_SESSION_DESTROY, client->next_request_id++,
                             client->session_id, 0, CUBE_GPU_FLAG_NONE);
        cube_gpu_rpc_send(client, &hdr, NULL);
        close(client->vsock_fd);
        client->vsock_fd = -1;
    }
    client->initialized = 0;
}

int cube_gpu_rpc_send(cube_gpu_rpc_client_t *client, const cube_gpu_rpc_header_t *hdr,
                      const void *payload) {
    if (send_all(client->vsock_fd, hdr, sizeof(*hdr)) < 0) return -1;
    if (hdr->payload_len > 0 && payload) {
        if (send_all(client->vsock_fd, payload, hdr->payload_len) < 0) return -1;
    }
    return 0;
}

int cube_gpu_rpc_recv(cube_gpu_rpc_client_t *client, cube_gpu_rpc_header_t *hdr_out,
                      void *payload_out, size_t payload_buf_size) {
    if (recv_all(client->vsock_fd, hdr_out, sizeof(*hdr_out)) < 0) return -1;

    if (hdr_out->magic != CUBE_GPU_RPC_MAGIC) return -1;
    if (!(hdr_out->flags & CUBE_GPU_FLAG_RESPONSE)) return -1;

    if (hdr_out->payload_len > 0 && payload_out && payload_buf_size > 0) {
        size_t to_read = hdr_out->payload_len < payload_buf_size
                             ? hdr_out->payload_len
                             : payload_buf_size;
        if (recv_all(client->vsock_fd, payload_out, to_read) < 0) return -1;
        if (to_read < hdr_out->payload_len) {
            uint8_t discard[256];
            size_t leftover = hdr_out->payload_len - to_read;
            while (leftover > 0) {
                size_t chunk = leftover < sizeof(discard) ? leftover : sizeof(discard);
                if (recv_all(client->vsock_fd, discard, chunk) < 0) return -1;
                leftover -= chunk;
            }
        }
    }
    return 0;
}
