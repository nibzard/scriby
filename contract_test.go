package main

import "testing"

func TestHelpShortFlagReturnsSuccessEnvelope(t *testing.T) {
	tests := []struct {
		name    string
		command string
		run     func() (Envelope, int)
		want    string
	}{
		{
			name:    "run",
			command: "run",
			run: func() (Envelope, int) {
				return handleRun([]string{"-h"})
			},
			want: runHelp(),
		},
		{
			name:    "validate",
			command: "validate",
			run: func() (Envelope, int) {
				return handleValidate([]string{"-h"})
			},
			want: validateHelp(),
		},
		{
			name:    "doctor",
			command: "doctor",
			run: func() (Envelope, int) {
				return handleDoctor([]string{"-h"})
			},
			want: doctorHelp(),
		},
		{
			name:    "replay",
			command: "replay",
			run: func() (Envelope, int) {
				return handleReplay([]string{"-h"})
			},
			want: replayHelp(),
		},
		{
			name:    "models pull",
			command: "models.pull",
			run: func() (Envelope, int) {
				return handleModels([]string{"pull", "-h"})
			},
			want: modelsHelp(),
		},
		{
			name:    "models list",
			command: "models.list",
			run: func() (Envelope, int) {
				return handleModels([]string{"list", "-h"})
			},
			want: modelsHelp(),
		},
		{
			name:    "models prune",
			command: "models.prune",
			run: func() (Envelope, int) {
				return handleModels([]string{"prune", "-h"})
			},
			want: modelsHelp(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env, code := tt.run()
			assertContractBasics(t, env, tt.command)
			if code != exitOK {
				t.Fatalf("exit code = %d, want %d", code, exitOK)
			}
			if env.Status != "succeeded" {
				t.Fatalf("status = %s, want succeeded", env.Status)
			}
			if len(env.Errors) != 0 {
				t.Fatalf("expected no errors, got %#v", env.Errors)
			}
			got, ok := env.Data.(string)
			if !ok {
				t.Fatalf("help data type = %T, want string", env.Data)
			}
			if got != tt.want {
				t.Fatalf("help data mismatch")
			}
		})
	}
}

func TestRootHelpEnvelopeContract(t *testing.T) {
	env := rootHelpEnvelope()
	assertContractBasics(t, env, "root")
	if env.Status != "succeeded" {
		t.Fatalf("status = %s, want succeeded", env.Status)
	}
	got, ok := env.Data.(string)
	if !ok {
		t.Fatalf("help data type = %T, want string", env.Data)
	}
	if got != rootHelp() {
		t.Fatal("root help data mismatch")
	}
}

func TestRootMissingCommandContract(t *testing.T) {
	env := rootMissingCommandEnvelope()
	assertContractBasics(t, env, "root")
	if env.Status != "failed" {
		t.Fatalf("status = %s, want failed", env.Status)
	}
	if len(env.Errors) == 0 || env.Errors[0].Code != "MISSING_COMMAND" {
		t.Fatalf("expected MISSING_COMMAND error, got %#v", env.Errors)
	}
}

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
