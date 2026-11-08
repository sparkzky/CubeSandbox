// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0
//

package filter

import (
	"context"
	"testing"

	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/constants"
	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/node"
	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/scheduler/selctx"
)

func gpuNode(id string, labels map[string]string) *node.Node {
	return &node.Node{InsID: id, IP: "10.0.0.1", NodeLabels: labels}
}

func healthyGPULabels() map[string]string {
	return map[string]string{
		gpuLabelPresent:     "true",
		gpuLabelCount:       "1",
		gpuLabelMemTotalMiB: "24576",
		gpuLabelMemUsedMiB:  "4096",
		gpuLabelSessions:    "1",
		gpuLabelMaxSessions: "4",
		gpuLabelDriver:      "615.71.09",
		gpuLabelModels:      "NVIDIA A100-SXM4-40GB",
	}
}

func gpuLabelsWith(overrides map[string]string) map[string]string {
	labels := healthyGPULabels()
	for k, v := range overrides {
		if v == "" {
			delete(labels, k)
			continue
		}
		labels[k] = v
	}
	return labels
}

func newGPUFilterCtx(annotation string, nodes node.NodeList) *selctx.SelectorCtx {
	ctx := selctx.New("random")
	ctx.Ctx = context.Background()
	ctx.ReqRes = &selctx.RequestResource{}
	if annotation != "" {
		ctx.ReqRes.Annotations = map[string]string{constants.AnnotationsGPU: annotation}
	}
	ctx.SetNodes(nodes)
	return ctx
}

func TestGpuFilterID(t *testing.T) {
	if got := NewGpuFilter().ID(); got != constants.SelectorFilterID+"/gpu" {
		t.Fatalf("unexpected filter ID: %s", got)
	}
}

func TestGpuFilterSelect(t *testing.T) {
	healthy := gpuNode("healthy", healthyGPULabels())

	tests := []struct {
		name       string
		annotation string
		nodes      node.NodeList
		wantIDs    []string
		wantErr    bool
	}{
		{
			name:       "non gpu request passes all nodes",
			annotation: "",
			nodes:      node.NodeList{healthy, gpuNode("no-gpu", nil)},
			wantIDs:    []string{"healthy", "no-gpu"},
		},
		{
			name:       "disabled request passes all nodes",
			annotation: `{"enable":false}`,
			nodes:      node.NodeList{healthy, gpuNode("no-gpu", nil)},
			wantIDs:    []string{"healthy", "no-gpu"},
		},
		{
			name:       "enabled request skips nodes without gpu labels",
			annotation: `{"enable":true}`,
			nodes:      node.NodeList{healthy, gpuNode("no-gpu", nil)},
			wantIDs:    []string{"healthy"},
		},
		{
			name:       "present must be true",
			annotation: `{"enable":true}`,
			nodes:      node.NodeList{gpuNode("absent", gpuLabelsWith(map[string]string{gpuLabelPresent: "false"}))},
			wantIDs:    nil,
		},
		{
			name:       "count zero rejects",
			annotation: `{"enable":true}`,
			nodes:      node.NodeList{gpuNode("count-zero", gpuLabelsWith(map[string]string{gpuLabelCount: "0"}))},
			wantIDs:    nil,
		},
		{
			name:       "default memory quota is 4096 and exact free passes",
			annotation: `{"enable":true}`,
			nodes: node.NodeList{
				gpuNode("free-4096", gpuLabelsWith(map[string]string{gpuLabelMemTotalMiB: "8192", gpuLabelMemUsedMiB: "4096"})),
				gpuNode("free-4095", gpuLabelsWith(map[string]string{gpuLabelMemTotalMiB: "8191", gpuLabelMemUsedMiB: "4096"})),
			},
			wantIDs: []string{"free-4096"},
		},
		{
			name:       "memory quota from annotation",
			annotation: `{"enable":true,"memory_quota_mb":8192}`,
			nodes: node.NodeList{
				healthy,
				gpuNode("free-8191", gpuLabelsWith(map[string]string{gpuLabelMemTotalMiB: "16383", gpuLabelMemUsedMiB: "8192"})),
			},
			wantIDs: []string{"healthy"},
		},
		{
			name:       "used above total rejects",
			annotation: `{"enable":true}`,
			nodes:      node.NodeList{gpuNode("over", gpuLabelsWith(map[string]string{gpuLabelMemTotalMiB: "4096", gpuLabelMemUsedMiB: "8192"}))},
			wantIDs:    nil,
		},
		{
			name:       "sessions below max passes",
			annotation: `{"enable":true}`,
			nodes:      node.NodeList{gpuNode("sessions", gpuLabelsWith(map[string]string{gpuLabelSessions: "3", gpuLabelMaxSessions: "4"}))},
			wantIDs:    []string{"sessions"},
		},
		{
			name:       "sessions at max rejects",
			annotation: `{"enable":true}`,
			nodes:      node.NodeList{gpuNode("sessions-full", gpuLabelsWith(map[string]string{gpuLabelSessions: "4", gpuLabelMaxSessions: "4"}))},
			wantIDs:    nil,
		},
		{
			name:       "driver requirement satisfied",
			annotation: `{"enable":true,"required_driver_min":"575.0"}`,
			nodes:      node.NodeList{healthy},
			wantIDs:    []string{"healthy"},
		},
		{
			name:       "driver requirement unmet",
			annotation: `{"enable":true,"required_driver_min":"615.71.10"}`,
			nodes:      node.NodeList{healthy},
			wantIDs:    nil,
		},
		{
			name:       "driver requirement missing label rejects",
			annotation: `{"enable":true,"required_driver_min":"575.0"}`,
			nodes:      node.NodeList{gpuNode("no-driver", gpuLabelsWith(map[string]string{gpuLabelDriver: ""}))},
			wantIDs:    nil,
		},
		{
			name:       "driver requirement malformed node version rejects",
			annotation: `{"enable":true,"required_driver_min":"575.0"}`,
			nodes:      node.NodeList{gpuNode("bad-driver", gpuLabelsWith(map[string]string{gpuLabelDriver: "abc"}))},
			wantIDs:    nil,
		},
		{
			name:       "driver requirement empty ignores driver",
			annotation: `{"enable":true,"required_driver_min":""}`,
			nodes:      node.NodeList{gpuNode("no-driver", gpuLabelsWith(map[string]string{gpuLabelDriver: ""}))},
			wantIDs:    []string{"no-driver"},
		},
		{
			name:       "model match case insensitive substring",
			annotation: `{"enable":true,"models":["nvidia a100"]}`,
			nodes:      node.NodeList{healthy},
			wantIDs:    []string{"healthy"},
		},
		{
			name:       "model match when request is more specific",
			annotation: `{"enable":true,"models":["NVIDIA A100-SXM4"]}`,
			nodes:      node.NodeList{healthy},
			wantIDs:    []string{"healthy"},
		},
		{
			name:       "model mismatch rejects",
			annotation: `{"enable":true,"models":["RTX 3060"]}`,
			nodes:      node.NodeList{healthy},
			wantIDs:    nil,
		},
		{
			name:       "model requirement with missing label rejects",
			annotation: `{"enable":true,"models":["A100"]}`,
			nodes:      node.NodeList{gpuNode("no-models", gpuLabelsWith(map[string]string{gpuLabelModels: ""}))},
			wantIDs:    nil,
		},
		{
			name:       "one of several models matches",
			annotation: `{"enable":true,"models":["H100","RTX 3060"]}`,
			nodes:      node.NodeList{gpuNode("rtx", gpuLabelsWith(map[string]string{gpuLabelModels: "RTX 3060,RTX 4090"}))},
			wantIDs:    []string{"rtx"},
		},
		{
			name:       "malformed json rejects all nodes",
			annotation: `{"enable":`,
			nodes:      node.NodeList{healthy, gpuNode("no-gpu", nil)},
			wantIDs:    nil,
			wantErr:    true,
		},
		{
			name:       "malformed numeric label rejects",
			annotation: `{"enable":true}`,
			nodes:      node.NodeList{gpuNode("bad-total", gpuLabelsWith(map[string]string{gpuLabelMemTotalMiB: "twenty"}))},
			wantIDs:    nil,
		},
		{
			name:       "non gpu nodes coexist with gpu request",
			annotation: `{"enable":true,"memory_quota_mb":1024}`,
			nodes: node.NodeList{
				gpuNode("gpu-a", gpuLabelsWith(nil)),
				gpuNode("no-gpu", nil),
				gpuNode("gpu-full", gpuLabelsWith(map[string]string{gpuLabelSessions: "8", gpuLabelMaxSessions: "8"})),
			},
			wantIDs: []string{"gpu-a"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewGpuFilter().Select(newGPUFilterCtx(tt.annotation, tt.nodes))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if len(got) != 0 {
					t.Fatalf("expected no nodes on error, got %d", len(got))
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			gotIDs := make([]string, 0, len(got))
			for _, n := range got {
				gotIDs = append(gotIDs, n.ID())
			}
			if len(gotIDs) != len(tt.wantIDs) {
				t.Fatalf("want nodes %v, got %v", tt.wantIDs, gotIDs)
			}
			for i := range tt.wantIDs {
				if gotIDs[i] != tt.wantIDs[i] {
					t.Fatalf("want nodes %v, got %v", tt.wantIDs, gotIDs)
				}
			}
		})
	}
}

func TestRequestNeedsGPU(t *testing.T) {
	tests := []struct {
		name       string
		annotation string
		want       bool
	}{
		{name: "missing annotation", annotation: "", want: false},
		{name: "disabled", annotation: `{"enable":false}`, want: false},
		{name: "enabled", annotation: `{"enable":true}`, want: true},
		{name: "malformed", annotation: `{`, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqRes := &selctx.RequestResource{}
			if tt.annotation != "" {
				reqRes.Annotations = map[string]string{constants.AnnotationsGPU: tt.annotation}
			}
			if got := RequestNeedsGPU(reqRes); got != tt.want {
				t.Fatalf("RequestNeedsGPU() = %v, want %v", got, tt.want)
			}
		})
	}
	if RequestNeedsGPU(nil) {
		t.Fatal("RequestNeedsGPU(nil) = true, want false")
	}
}

func TestCompareDriverVersion(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"615.71.09", "575.0", 1},
		{"575.0", "575.0", 0},
		{"575.0", "575.0.1", -1},
		{"575.1", "575.0.9", 1},
		{"615.71.09", "615.71.9", 0},
	}
	for _, tt := range tests {
		got, err := compareDriverVersion(tt.a, tt.b)
		if err != nil {
			t.Fatalf("compareDriverVersion(%q, %q) error: %v", tt.a, tt.b, err)
		}
		if got != tt.want {
			t.Fatalf("compareDriverVersion(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}

	for _, bad := range []string{"", "abc", "1..2", "1.-2"} {
		if _, err := compareDriverVersion(bad, "575.0"); err == nil {
			t.Fatalf("compareDriverVersion(%q) expected error", bad)
		}
	}
}
