package state

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEnsureExplicitStateDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "state")
	got, err := Ensure(dir)
	if err != nil {
		t.Fatalf("Ensure returned error: %v", err)
	}
	if got != dir {
		t.Fatalf("Ensure returned %q, want %q", got, dir)
	}
	if !dirExists(got) {
		t.Fatalf("state dir was not created: %s", got)
	}
}

func TestEnsureMigratesLegacyCacheState(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))

	legacy, ok := LegacyDir()
	if !ok {
		t.Skip("legacy cache dir unavailable")
	}
	model := filepath.Join(legacy, "models", "ggml-medium.bin")
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		t.Fatalf("mkdir legacy model dir: %v", err)
	}
	if err := os.WriteFile(model, []byte("model"), 0o644); err != nil {
		t.Fatalf("write legacy model: %v", err)
	}

	stateDir, err := Ensure("")
	if err != nil {
		t.Fatalf("Ensure returned error: %v", err)
	}
	migrated := filepath.Join(stateDir, "models", "ggml-medium.bin")
	if _, err := os.Stat(migrated); err != nil {
		t.Fatalf("expected migrated model at %s: %v", migrated, err)
	}
}
