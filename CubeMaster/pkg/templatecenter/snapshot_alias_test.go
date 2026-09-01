// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

package templatecenter

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/db/models"
)

func TestValidateSnapshotAliasKey(t *testing.T) {
	valid := []string{
		"foo:default",
		"foo:v2",
		"a1-b:tag-1",
		strings.Repeat("a", 64) + ":" + strings.Repeat("b", 63), // 128 = column width
	}
	for _, key := range valid {
		if err := ValidateSnapshotAliasKey(key); err != nil {
			t.Errorf("ValidateSnapshotAliasKey(%q) = %v, want nil", key, err)
		}
	}

	invalid := []string{
		"",
		"foo",              // missing tag
		":v2",              // empty alias
		"foo:",             // empty tag
		"foo:bar:baz",      // multiple separators
		"FOO:default",      // uppercase alias
		"foo:DEFAULT",      // uppercase tag
		"foo_bar:v2",       // underscore not in charset
		"snap-foo:default", // reserved id prefix in alias
		"tpl-foo:default",  // reserved id prefix in alias
		"foo:snap-bar",     // reserved id prefix in tag
		"foo :default",     // inner space in alias
		"foo:de fault",     // inner space in tag
		strings.Repeat("a", 64) + ":" + strings.Repeat("b", 64), // 129 > 128
		strings.Repeat("a", 65) + ":v",                          // alias segment > 64
	}
	for _, key := range invalid {
		if err := ValidateSnapshotAliasKey(key); !errors.Is(err, ErrSnapshotAliasInvalid) {
			t.Errorf("ValidateSnapshotAliasKey(%q) = %v, want ErrSnapshotAliasInvalid", key, err)
		}
	}
}

func TestStripAliasNamespace(t *testing.T) {
	cases := map[string]string{
		"foo":             "foo",
		"foo:default":     "foo:default",
		"team/foo":        "foo",
		"team/sub/foo:v2": "foo:v2",
		"  team/foo  ":    "foo",
		"/leading":        "leading",
		"":                "",
		"/":               "",
	}
	for in, want := range cases {
		if got := stripAliasNamespace(in); got != want {
			t.Errorf("stripAliasNamespace(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestResolveSnapshotAliasExactAndDefaultDerivation verifies the lookup
// semantics: exact key match always; "<id>:default" only when deriveDefault
// is set and the identifier carries no tag. getSnapshotRecordByAlias is
// stubbed so the test exercises candidate construction, not gorm.
func TestResolveSnapshotAliasExactAndDefaultDerivation(t *testing.T) {
	oldDB := store.db
	store.db = nil
	defer func() { store.db = oldDB }()

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	aliases := map[string]string{
		"foo:default": "snap-1",
		"foo:v2":      "snap-2",
	}
	patches.ApplyFunc(getSnapshotRecordByAlias, func(_ context.Context, aliasKey string) (*models.SnapshotRecord, error) {
		if id, ok := aliases[aliasKey]; ok {
			return &models.SnapshotRecord{SnapshotID: id, Alias: &aliasKey}, nil
		}
		return nil, ErrSnapshotNotFound
	})

	ctx := context.Background()

	// Exact qualified key resolves regardless of deriveDefault.
	for _, derive := range []bool{true, false} {
		got, err := ResolveSnapshotAlias(ctx, "foo:v2", derive)
		if err != nil || got != "snap-2" {
			t.Fatalf("ResolveSnapshotAlias(foo:v2, derive=%v) = %q, %v; want snap-2", derive, got, err)
		}
	}

	// Tag-less identifier only resolves when the default tag is derived.
	got, err := ResolveSnapshotAlias(ctx, "foo", true)
	if err != nil || got != "snap-1" {
		t.Fatalf("ResolveSnapshotAlias(foo, derive=true) = %q, %v; want snap-1", got, err)
	}
	if _, err := ResolveSnapshotAlias(ctx, "foo", false); !errors.Is(err, ErrSnapshotNotFound) {
		t.Fatalf("ResolveSnapshotAlias(foo, derive=false) err = %v, want ErrSnapshotNotFound", err)
	}

	// Namespace is stripped before lookup.
	got, err = ResolveSnapshotAlias(ctx, "team/foo:default", false)
	if err != nil || got != "snap-1" {
		t.Fatalf("ResolveSnapshotAlias(team/foo:default) = %q, %v; want snap-1", got, err)
	}

	// Unknown identifier misses.
	if _, err := ResolveSnapshotAlias(ctx, "bar:default", true); !errors.Is(err, ErrSnapshotNotFound) {
		t.Fatalf("ResolveSnapshotAlias(bar:default) err = %v, want ErrSnapshotNotFound", err)
	}
}

// TestNormalizeSnapshotRefPassthrough verifies that tpl-/snap- prefixed
// references never hit the alias table on snapshot endpoints.
func TestNormalizeSnapshotRefPassthrough(t *testing.T) {
	oldDB := store.db
	store.db = nil
	defer func() { store.db = oldDB }()

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	called := false
	patches.ApplyFunc(getSnapshotRecordByAlias, func(_ context.Context, aliasKey string) (*models.SnapshotRecord, error) {
		called = true
		return nil, ErrSnapshotNotFound
	})

	for _, ref := range []string{"snap-abc123", "tpl-abc123", "  snap-abc123  "} {
		got, err := NormalizeSnapshotRef(context.Background(), ref)
		if err != nil || got != strings.TrimSpace(ref) {
			t.Fatalf("NormalizeSnapshotRef(%q) = %q, %v; want passthrough", ref, got, err)
		}
	}
	if called {
		t.Fatal("prefixed references must not reach the alias lookup")
	}
}
