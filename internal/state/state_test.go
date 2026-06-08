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

func TestEnsureMigratesMissingLegacyFilesWhenStateDirExists(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))

	stateDir := filepath.Join(home, ".scriby")
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		t.Fatalf("mkdir state dir: %v", err)
	}
	existing := filepath.Join(stateDir, "models", "ggml-medium.bin")
	if err := os.MkdirAll(filepath.Dir(existing), 0o755); err != nil {
		t.Fatalf("mkdir existing model dir: %v", err)
	}
	if err := os.WriteFile(existing, []byte("current"), 0o644); err != nil {
		t.Fatalf("write existing model: %v", err)
	}

	legacy, ok := LegacyDir()
	if !ok {
		t.Skip("legacy cache dir unavailable")
	}
	legacyTiny := filepath.Join(legacy, "models", "ggml-tiny.bin")
	if err := os.MkdirAll(filepath.Dir(legacyTiny), 0o755); err != nil {
		t.Fatalf("mkdir legacy model dir: %v", err)
	}
	if err := os.WriteFile(legacyTiny, []byte("tiny"), 0o644); err != nil {
		t.Fatalf("write legacy tiny model: %v", err)
	}
	legacyMedium := filepath.Join(legacy, "models", "ggml-medium.bin")
	if err := os.WriteFile(legacyMedium, []byte("legacy"), 0o644); err != nil {
		t.Fatalf("write legacy medium model: %v", err)
	}

	got, err := Ensure("")
	if err != nil {
		t.Fatalf("Ensure returned error: %v", err)
	}
	if got != stateDir {
		t.Fatalf("Ensure returned %q, want %q", got, stateDir)
	}
	migratedTiny := filepath.Join(stateDir, "models", "ggml-tiny.bin")
	if b, err := os.ReadFile(migratedTiny); err != nil || string(b) != "tiny" {
		t.Fatalf("expected migrated tiny model, got %q, err %v", string(b), err)
	}
	if b, err := os.ReadFile(existing); err != nil || string(b) != "current" {
		t.Fatalf("expected existing model to remain unchanged, got %q, err %v", string(b), err)
	}
}
