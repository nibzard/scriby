package main

import "testing"

func TestHandleValidateMissingInputContract(t *testing.T) {
	env, code := handleValidate([]string{})
	assertContractBasics(t, env, "validate")
	if code != exitInput {
		t.Fatalf("exit code = %d, want %d", code, exitInput)
	}
	if env.Status != "failed" {
		t.Fatalf("status = %s, want failed", env.Status)
	}
	if len(env.Errors) == 0 || env.Errors[0].Code != "MISSING_INPUT" {
		t.Fatalf("expected MISSING_INPUT error, got %#v", env.Errors)
	}
}

func TestHandleRunMissingInputContract(t *testing.T) {
	env, code := handleRun([]string{})
	assertContractBasics(t, env, "run")
	if code != exitInput {
		t.Fatalf("exit code = %d, want %d", code, exitInput)
	}
	if env.Status != "failed" {
		t.Fatalf("status = %s, want failed", env.Status)
	}
	if len(env.Errors) == 0 || env.Errors[0].Code != "MISSING_INPUT" {
		t.Fatalf("expected MISSING_INPUT error, got %#v", env.Errors)
	}
}

func TestHandleModelsPruneRequiresYesContract(t *testing.T) {
	env, code := handleModels([]string{"prune", "--state-dir", t.TempDir()})
	assertContractBasics(t, env, "models.prune")
	if code != exitInput {
		t.Fatalf("exit code = %d, want %d", code, exitInput)
	}
	if env.Status != "failed" {
		t.Fatalf("status = %s, want failed", env.Status)
	}
	if len(env.Errors) == 0 || env.Errors[0].Code != "CONFIRMATION_REQUIRED" {
		t.Fatalf("expected CONFIRMATION_REQUIRED error, got %#v", env.Errors)
	}
}

func assertContractBasics(t *testing.T, env Envelope, command string) {
	t.Helper()
	if env.SchemaVersion != schemaVersion {
		t.Fatalf("schema_version = %s, want %s", env.SchemaVersion, schemaVersion)
	}
	if env.Command != command {
		t.Fatalf("command = %s, want %s", env.Command, command)
	}
	if env.RunID == "" {
		t.Fatal("run_id must be set")
	}
	if env.Metrics == nil {
		t.Fatal("metrics must be present")
	}
	if _, ok := env.Metrics["duration_ms"]; !ok {
		t.Fatal("metrics.duration_ms must be present")
	}
}
