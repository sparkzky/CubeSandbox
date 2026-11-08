// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0
//

package gpu

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func testConfig() Config {
	return Config{Enabled: true, MaxSessionsPerGPU: 64}
}

func discoverDevices(devices ...Device) func() ([]Device, error) {
	return func() ([]Device, error) {
		return devices, nil
	}
}

func reserveReq(memoryMB uint64) Request {
	return Request{Enable: true, MemoryQuotaMB: memoryMB}
}

func TestParseRequest(t *testing.T) {
	t.Parallel()

	req, found, err := ParseRequest(nil)
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, Request{MemoryQuotaMB: DefaultMemoryQuotaMB}, req)

	req, found, err = ParseRequest(map[string]string{AnnotationKey: "   "})
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, DefaultMemoryQuotaMB, req.MemoryQuotaMB)

	req, found, err = ParseRequest(map[string]string{
		AnnotationKey: `{"enable":true,"memory_quota_mb":8192,"sm_percent":50,"required_driver_min":"575.0","models":["A100"," A100 ","rTx 3060"]}`,
	})
	require.NoError(t, err)
	require.True(t, found)
	require.True(t, req.Enable)
	require.Equal(t, uint64(8192), req.MemoryQuotaMB)
	require.Equal(t, 50, req.SMPercent)
	require.Equal(t, "575.0", req.RequiredDriverMin)
	require.Equal(t, []string{"A100", "rTx 3060"}, req.Models)

	req, found, err = ParseRequest(map[string]string{AnnotationKey: `{"enable":true}`})
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, DefaultMemoryQuotaMB, req.MemoryQuotaMB)
	require.Equal(t, 0, req.SMPercent)
	require.Empty(t, req.RequiredDriverMin)
	require.Empty(t, req.Models)

	_, _, err = ParseRequest(map[string]string{AnnotationKey: `{"enable":`})
	require.Error(t, err)

	_, _, err = ParseRequest(map[string]string{AnnotationKey: `{"enable":true,"sm_percent":101}`})
	require.Error(t, err)

	_, _, err = ParseRequest(map[string]string{AnnotationKey: `{"enable":true,"required_driver_min":"v575"}`})
	require.Error(t, err)
}

func TestParseNVIDIASMIOutput(t *testing.T) {
	t.Parallel()

	data := []byte(
		"1, GPU-b, NVIDIA GeForce RTX 3060, 12288, 535.104.05\n" +
			"0, GPU-a, NVIDIA A100-SXM4-40GB, 40960, 535.104.05\n")
	devices, err := ParseNVIDIASMIOutput(data)
	require.NoError(t, err)
	require.Len(t, devices, 2)
	require.Equal(t, Device{UUID: "GPU-a", Name: "NVIDIA A100-SXM4-40GB", DriverVersion: "535.104.05", MemoryMB: 40960}, devices[0])
	require.Equal(t, Device{UUID: "GPU-b", Name: "NVIDIA GeForce RTX 3060", DriverVersion: "535.104.05", MemoryMB: 12288}, devices[1])

	_, err = ParseNVIDIASMIOutput([]byte("not,a,valid,row\n"))
	require.Error(t, err)
}

func TestDiscoverNVIDIASMIWithRunner(t *testing.T) {
	t.Parallel()

	var gotName string
	var gotArgs []string
	runner := func(_ context.Context, name string, args ...string) ([]byte, error) {
		gotName = name
		gotArgs = append([]string(nil), args...)
		return []byte("0, GPU-0, NVIDIA A100, 40960, 550.54.14\n"), nil
	}
	devices, err := DiscoverNVIDIASMIWithRunner(runner)
	require.NoError(t, err)
	require.Equal(t, "nvidia-smi", gotName)
	require.Equal(t, []string{nvidiaSMIQuery, "--format=csv,noheader,nounits"}, gotArgs)
	require.Len(t, devices, 1)
	require.Equal(t, "GPU-0", devices[0].UUID)

	_, err = DiscoverNVIDIASMIWithRunner(func(context.Context, string, ...string) ([]byte, error) {
		return nil, errors.New("command not found")
	})
	require.Error(t, err)
}

func TestReservePicksDeviceWithMostFreeMemoryAndLabels(t *testing.T) {
	t.Parallel()

	m := NewManager(testConfig(), discoverDevices(
		Device{UUID: "GPU-a", Name: "NVIDIA GeForce RTX 3060", DriverVersion: "535.104.05", MemoryMB: 8192},
		Device{UUID: "GPU-b", Name: "NVIDIA GeForce RTX 3060", DriverVersion: "535.104.05", MemoryMB: 24576},
	))

	labels := m.Labels()
	require.Equal(t, "true", labels[LabelPresent])
	require.Equal(t, "2", labels[LabelCount])
	require.Equal(t, "32768", labels[LabelMemoryTotalMiB])
	require.Equal(t, "0", labels[LabelMemoryUsedMiB])
	require.Equal(t, "0", labels[LabelSessions])
	require.Equal(t, "128", labels[LabelMaxSessions])
	require.Equal(t, "535.104.05", labels[LabelDriver])
	require.Equal(t, "NVIDIA GeForce RTX 3060", labels[LabelModels])

	assignment, err := m.Reserve("sandbox-1", reserveReq(4096))
	require.NoError(t, err)
	require.Equal(t, "GPU-b", assignment.DeviceUUID)
	require.Equal(t, uint64(4096), assignment.MemoryQuotaMB)

	// Reserving the same sandbox again is idempotent.
	again, err := m.Reserve("sandbox-1", reserveReq(4096))
	require.NoError(t, err)
	require.Equal(t, assignment, again)

	// Still the card with the most free memory wins.
	second, err := m.Reserve("sandbox-2", reserveReq(4096))
	require.NoError(t, err)
	require.Equal(t, "GPU-b", second.DeviceUUID)

	_, err = m.Reserve("sandbox-1", reserveReq(8192))
	require.ErrorIs(t, err, ErrAlreadyReserved)

	labels = m.Labels()
	require.Equal(t, "8192", labels[LabelMemoryUsedMiB])
	require.Equal(t, "2", labels[LabelSessions])

	require.NoError(t, m.Attach("sandbox-1"))
	require.NoError(t, m.Attach("sandbox-1"))
	require.ErrorIs(t, m.Attach("missing"), ErrUnknownSandbox)

	require.NoError(t, m.Release("sandbox-1"))
	require.NoError(t, m.Release("sandbox-1"))
	require.NoError(t, m.Release("missing"))
	labels = m.Labels()
	require.Equal(t, "4096", labels[LabelMemoryUsedMiB])
	require.Equal(t, "1", labels[LabelSessions])
}

func TestReserveCapacityAndMaxSessions(t *testing.T) {
	t.Parallel()

	t.Run("memory", func(t *testing.T) {
		t.Parallel()
		m := NewManager(Config{Enabled: true, MaxSessionsPerGPU: 64}, discoverDevices(
			Device{UUID: "GPU-a", Name: "GPU", DriverVersion: "535.104.05", MemoryMB: 4096},
		))
		_, err := m.Reserve("too-big", reserveReq(8192))
		require.ErrorIs(t, err, ErrNoCapacity)

		_, err = m.Reserve("fits", reserveReq(4096))
		require.NoError(t, err)
		_, err = m.Reserve("second", reserveReq(1024))
		require.ErrorIs(t, err, ErrNoCapacity)

		require.NoError(t, m.Release("fits"))
		_, err = m.Reserve("second", reserveReq(1024))
		require.NoError(t, err)
	})

	t.Run("max sessions", func(t *testing.T) {
		t.Parallel()
		m := NewManager(Config{Enabled: true, MaxSessionsPerCard: 1}, discoverDevices(
			Device{UUID: "GPU-a", Name: "GPU", DriverVersion: "535.104.05", MemoryMB: 8192},
			Device{UUID: "GPU-b", Name: "GPU", DriverVersion: "535.104.05", MemoryMB: 8192},
		))
		_, err := m.Reserve("one", reserveReq(1024))
		require.NoError(t, err)
		_, err = m.Reserve("two", reserveReq(1024))
		require.NoError(t, err)
		// Both cards still have plenty of memory, but each hit its session cap.
		_, err = m.Reserve("three", reserveReq(1024))
		require.ErrorIs(t, err, ErrNoCapacity)

		require.Equal(t, "2", m.Labels()[LabelMaxSessions])
		require.NoError(t, m.Release("one"))
		_, err = m.Reserve("three", reserveReq(1024))
		require.NoError(t, err)
	})
}

func TestReserveCompatibility(t *testing.T) {
	t.Parallel()

	devices := discoverDevices(
		Device{UUID: "GPU-a", Name: "NVIDIA GeForce RTX 3060", DriverVersion: "535.104.05", MemoryMB: 12288},
	)

	t.Run("driver below minimum", func(t *testing.T) {
		t.Parallel()
		m := NewManager(testConfig(), devices)
		req := reserveReq(1024)
		req.RequiredDriverMin = "575.0"
		_, err := m.Reserve("sb", req)
		require.ErrorIs(t, err, ErrIncompatible)
		require.Contains(t, err.Error(), "required_driver_min")
	})

	t.Run("driver satisfied", func(t *testing.T) {
		t.Parallel()
		m := NewManager(testConfig(), devices)
		req := reserveReq(1024)
		req.RequiredDriverMin = "535.104"
		_, err := m.Reserve("sb", req)
		require.NoError(t, err)
	})

	t.Run("unknown model", func(t *testing.T) {
		t.Parallel()
		m := NewManager(testConfig(), devices)
		req := reserveReq(1024)
		req.Models = []string{"A100"}
		_, err := m.Reserve("sb", req)
		require.ErrorIs(t, err, ErrIncompatible)
		require.Contains(t, err.Error(), "models")
	})

	t.Run("case insensitive substring model", func(t *testing.T) {
		t.Parallel()
		m := NewManager(testConfig(), devices)
		req := reserveReq(1024)
		req.Models = []string{"rtx 3060"}
		_, err := m.Reserve("sb", req)
		require.NoError(t, err)
	})

	t.Run("invalid driver requirement", func(t *testing.T) {
		t.Parallel()
		m := NewManager(testConfig(), devices)
		req := reserveReq(1024)
		req.RequiredDriverMin = "v575"
		_, err := m.Reserve("sb", req)
		require.Error(t, err)
		require.NotErrorIs(t, err, ErrIncompatible)
	})
}

func TestQuarantineBlocksAllocation(t *testing.T) {
	t.Parallel()

	m := NewManager(Config{Enabled: true, MaxSessionsPerGPU: 8}, discoverDevices(
		Device{UUID: "GPU-a", Name: "GPU", DriverVersion: "535.104.05", MemoryMB: 4096},
	))
	_, err := m.Reserve("quarantined", reserveReq(4096))
	require.NoError(t, err)
	m.Quarantine("quarantined")

	_, err = m.Reserve("quarantined", reserveReq(4096))
	require.ErrorIs(t, err, ErrQuarantined)
	require.ErrorIs(t, m.Attach("quarantined"), ErrQuarantined)

	// Quarantined quota is still charged to the node, so nothing else fits.
	_, err = m.Reserve("other", reserveReq(4096))
	require.ErrorIs(t, err, ErrNoCapacity)
	labels := m.Labels()
	require.Equal(t, "4096", labels[LabelMemoryUsedMiB])
	require.Equal(t, "1", labels[LabelSessions])

	// Releasing is an explicit recovery path and returns the capacity.
	require.NoError(t, m.Release("quarantined"))
	_, err = m.Reserve("other", reserveReq(4096))
	require.NoError(t, err)

	// Unknown IDs can be quarantined to stop a racing retry.
	m.Quarantine("ghost")
	_, err = m.Reserve("ghost", reserveReq(1024))
	require.ErrorIs(t, err, ErrQuarantined)
}

func TestConcurrentReserveDoesNotOversell(t *testing.T) {
	t.Parallel()

	const (
		quota      = 1024
		capacity   = 4096
		concurrent = 32
	)
	m := NewManager(Config{Enabled: true, MaxSessionsPerGPU: 256}, discoverDevices(
		Device{UUID: "GPU-a", Name: "GPU", DriverVersion: "535.104.05", MemoryMB: capacity},
	))

	var (
		wg        sync.WaitGroup
		mu        sync.Mutex
		succeeded []string
	)
	for i := range concurrent {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := "sandbox-" + string(rune('a'+i%26)) + string(rune('a'+i/26))
			if _, err := m.Reserve(id, reserveReq(quota)); err == nil {
				mu.Lock()
				succeeded = append(succeeded, id)
				mu.Unlock()
				return
			}
		}(i)
	}
	wg.Wait()

	require.Len(t, succeeded, int(capacity/quota))
	labels := m.Labels()
	require.Equal(t, "4096", labels[LabelMemoryUsedMiB])
	require.Equal(t, "4", labels[LabelSessions])

	for _, id := range succeeded {
		require.NoError(t, m.Release(id))
	}
	require.Equal(t, "0", m.Labels()[LabelMemoryUsedMiB])
}

func TestReconcileRebuildsLedger(t *testing.T) {
	t.Parallel()

	m := NewManager(Config{Enabled: true, MaxSessionsPerGPU: 8}, discoverDevices(
		Device{UUID: "GPU-a", Name: "GPU", DriverVersion: "535.104.05", MemoryMB: 8192},
	))
	_, err := m.Reserve("stale", reserveReq(4096))
	require.NoError(t, err)

	m.Reconcile(map[string]Request{
		"alive-1": {Enable: true, MemoryQuotaMB: 4096},
		"alive-2": {Enable: true, MemoryQuotaMB: 4096},
		"paused":  {Enable: false},
	})
	labels := m.Labels()
	require.Equal(t, "2", labels[LabelSessions])
	require.Equal(t, "8192", labels[LabelMemoryUsedMiB])
	require.NoError(t, m.Attach("alive-1"))

	// The stale session is gone, so its quota is no longer available.
	_, err = m.Reserve("new", reserveReq(1024))
	require.ErrorIs(t, err, ErrNoCapacity)
	require.NoError(t, m.Release("alive-1"))
	_, err = m.Reserve("new", reserveReq(1024))
	require.NoError(t, err)
}

func TestDisabledManagerReportsNothing(t *testing.T) {
	t.Parallel()

	discovered := false
	m := NewManager(Config{Enabled: false}, func() ([]Device, error) {
		discovered = true
		return []Device{{UUID: "GPU-a", MemoryMB: 4096}}, nil
	})
	require.False(t, discovered)
	require.Empty(t, m.Labels())

	_, err := m.Reserve("sb", reserveReq(1024))
	require.ErrorIs(t, err, ErrDisabled)
}

func TestDiscoveryFailureDegradesToNoGPU(t *testing.T) {
	t.Parallel()

	m := NewManager(testConfig(), func() ([]Device, error) {
		return nil, errors.New("nvidia-smi: not found")
	})
	require.Empty(t, m.Labels())

	_, err := m.Reserve("sb", reserveReq(1024))
	require.ErrorIs(t, err, ErrNoGPU)
	require.True(t, strings.Contains(err.Error(), "not found"))

	// No devices discovered is also a clean "no GPU" state.
	empty := NewManager(testConfig(), discoverDevices())
	require.Empty(t, empty.Labels())
	_, err = empty.Reserve("sb", reserveReq(1024))
	require.ErrorIs(t, err, ErrNoGPU)
}

func TestDriverVersionComparison(t *testing.T) {
	t.Parallel()

	require.True(t, driverSatisfies("615.71.09", "575.0"))
	require.True(t, driverSatisfies("575", "575.0"))
	require.True(t, driverSatisfies("575.0.1", "575.0"))
	require.False(t, driverSatisfies("535.104.05", "575.0"))
	require.False(t, driverSatisfies("garbage", "575.0"))
}
