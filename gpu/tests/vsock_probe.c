/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * vsock_probe -- tiny connectivity probe used by the e2e scripts to wait for
 * cube-gpu-daemon to start listening, without a fixed sleep.
 *
 * Usage:
 *   vsock_probe [--cid N] [--port N] [--timeout MS]
 *
 * Defaults: CID from CUBE_GPU_VSOCK_CID (else 1 = VMADDR_CID_LOCAL) and port
 * from CUBE_GPU_VSOCK_PORT (else 0x5055).  Exit 0 when connect() succeeds.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Local integer parsing: avoid glibc's __isoc23_strtoull redirect (GLIBC_2.38)
 * so the library stays loadable on older guest userspaces (e.g. glibc 2.36). */
static unsigned long long cube_parse_u64(const char *s, int base) {
    unsigned long long v = 0;
    int neg = 0;
    if (!s) return 0;
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;
    if (base == 0) {
        if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
        else if (s[0] == '0') { base = 8; s++; }
        else base = 10;
    } else if (base == 16 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
    }
    for (;; s++) {
        int d;
        char c = *s;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else break;
        if (d >= base) break;
        v = v * (unsigned)base + (unsigned)d;
    }
    return neg ? (unsigned long long)(-(long long)v) : v;
}
#define strtoull(s, e, b) cube_parse_u64((s), (b))
#define strtoul(s, e, b) ((unsigned long)cube_parse_u64((s), (b)))

#include <unistd.h>
#include <sys/socket.h>
#include <linux/vm_sockets.h>

#ifndef VMADDR_CID_LOCAL
#define VMADDR_CID_LOCAL 1
#endif

static int probe(uint32_t cid, uint32_t port, int timeout_ms)
{
    struct sockaddr_vm addr;
    struct pollfd pfd;
    int fd, err = 0;
    socklen_t len = sizeof(err);
    int flags;

    fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;

    flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0)
        (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    memset(&addr, 0, sizeof(addr));
    addr.svm_family = AF_VSOCK;
    addr.svm_cid = cid;
    addr.svm_port = port;

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
        close(fd);
        return 0;
    }
    if (errno != EINPROGRESS && errno != EINTR) {
        close(fd);
        return -1;
    }

    pfd.fd = fd;
    pfd.events = POLLOUT;
    pfd.revents = 0;
    if (poll(&pfd, 1, timeout_ms) <= 0) {
        close(fd);
        return -1;
    }
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) < 0 || err != 0) {
        close(fd);
        return -1;
    }
    close(fd);
    return 0;
}

int main(int argc, char **argv)
{
    uint32_t cid = VMADDR_CID_LOCAL;
    uint32_t port = 0x5055;
    int timeout_ms = 500;
    int i;
    const char *env;

    env = getenv("CUBE_GPU_VSOCK_CID");
    if (env && env[0])
        cid = (uint32_t)strtoul(env, NULL, 10);
    env = getenv("CUBE_GPU_VSOCK_PORT");
    if (env && env[0])
        port = (uint32_t)strtoul(env, NULL, 10);

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cid") == 0 && i + 1 < argc)
            cid = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc)
            timeout_ms = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--cid N] [--port N] [--timeout MS]\n", argv[0]);
            return 2;
        }
    }

    if (probe(cid, port, timeout_ms) != 0) {
        fprintf(stderr, "[probe] CID %u port %u not reachable\n", cid, port);
        return 1;
    }
    printf("[probe] CID %u port %u reachable\n", cid, port);
    return 0;
}
