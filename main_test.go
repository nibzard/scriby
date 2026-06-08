package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"scriby/internal/clipboard"
	history "scriby/internal/history"
)

func TestModelFilenameNormalization(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "default", in: "", want: "ggml-medium.bin"},
		{name: "plain", in: "small", want: "ggml-small.bin"},
		{name: "prefixed", in: "ggml-large-v3.bin", want: "ggml-large-v3.bin"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := modelFilename(tt.in)
			if got != tt.want {
				t.Fatalf("modelFilename(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestModelDownloadNoticeExplainsFirstRunCost(t *testing.T) {
	notice := modelDownloadNotice("ggml-medium.bin", "/tmp/scriby/models/ggml-medium.bin")
	for _, want := range []string{"First-time model download", "ggml-medium.bin", "about 1.4 GiB", "several minutes", "caches"} {
		if !strings.Contains(notice, want) {
			t.Fatalf("modelDownloadNotice() = %q, want substring %q", notice, want)
		}
	}
	if got := modelSizeHint("medium"); got != "about 1.4 GiB" {
		t.Fatalf("modelSizeHint(medium) = %q", got)
	}
}

func TestRunArtifactPathsModes(t *testing.T) {
	stateDir := t.TempDir()
	mediaDir := t.TempDir()
	media := filepath.Join(mediaDir, "meeting.wav")

	both, err := runArtifactPaths(stateDir, "run-1", media, "", "both")
	if err != nil {
		t.Fatalf("runArtifactPaths both returned error: %v", err)
	}
	if both.TranscriptTarget != both.ArtifactTranscript {
		t.Fatalf("both transcript target = %q, want artifact %q", both.TranscriptTarget, both.ArtifactTranscript)
	}
	if both.ArtifactTranscript == "" || both.LatestTranscript != filepath.Join(mediaDir, "meeting.md") {
		t.Fatalf("both paths = %#v", both)
	}
	if !strings.Contains(both.ArtifactTranscript, filepath.Join(stateDir, "runs", "run-1")) {
		t.Fatalf("artifact transcript = %q, want under run dir", both.ArtifactTranscript)
	}

	versioned, err := runArtifactPaths(stateDir, "run-1", media, "", "versioned")
	if err != nil {
		t.Fatalf("runArtifactPaths versioned returned error: %v", err)
	}
	if versioned.LatestTranscript != "" || versioned.ArtifactTranscript == "" {
		t.Fatalf("versioned paths = %#v", versioned)
	}
	if !strings.Contains(versioned.ArtifactTranscript, filepath.Join(stateDir, "runs", "run-1")) {
		t.Fatalf("versioned artifact transcript = %q, want under state dir", versioned.ArtifactTranscript)
	}

	outDir := t.TempDir()
	versionedOut, err := runArtifactPaths(stateDir, "run-2", media, outDir, "versioned")
	if err != nil {
		t.Fatalf("runArtifactPaths versioned output-dir returned error: %v", err)
	}
	if versionedOut.LatestTranscript != "" {
		t.Fatalf("versioned output-dir latest transcript = %q, want empty", versionedOut.LatestTranscript)
	}
	if !strings.Contains(versionedOut.ArtifactTranscript, filepath.Join(outDir, "runs", "run-2")) {
		t.Fatalf("versioned output-dir artifact transcript = %q, want under output dir", versionedOut.ArtifactTranscript)
	}

	latestDir := t.TempDir()
	latest, err := runArtifactPaths(stateDir, "run-1", media, latestDir, "latest")
	if err != nil {
		t.Fatalf("runArtifactPaths latest returned error: %v", err)
	}
	if latest.ArtifactTranscript != "" || latest.TranscriptTarget != filepath.Join(latestDir, "meeting.md") {
		t.Fatalf("latest paths = %#v", latest)
	}
}

func TestNormalizeArch(t *testing.T) {
	if got := normalizeArch("x86_64"); got != "amd64" {
		t.Fatalf("normalizeArch(x86_64) = %q, want amd64", got)
	}
	if got := normalizeArch("aarch64"); got != "arm64" {
		t.Fatalf("normalizeArch(aarch64) = %q, want arm64", got)
	}
	if got := normalizeArch("arm64"); got != "arm64" {
		t.Fatalf("normalizeArch(arm64) = %q, want arm64", got)
	}
}

func TestSaveHistoryRecordUsesArtifactPathAsCanonical(t *testing.T) {
	stateDir := t.TempDir()
	dir := t.TempDir()
	latest := filepath.Join(dir, "meeting.md")
	artifact := filepath.Join(stateDir, "runs", "run-1", "meeting.transcript.md")
	if err := os.MkdirAll(filepath.Dir(artifact), 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(latest, []byte("latest text"), 0o644); err != nil {
		t.Fatalf("write latest: %v", err)
	}
	if err := os.WriteFile(artifact, []byte("artifact text"), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	env := newEnvelope("run")
	env.RunID = "20260224-130000-artifact"
	env.Data = RunData{
		Input:        "meeting.wav",
		Engine:       "whisper",
		ArtifactMode: "both",
		Files: []FileResult{{
			File:               "meeting.wav",
			Transcript:         latest,
			LatestTranscript:   latest,
			ArtifactTranscript: artifact,
			Status:             "succeeded",
		}},
	}
	env.Metrics["duration_ms"] = int64(1)
	env.Metrics["files_total"] = int64(1)
	env.Metrics["files_succeeded"] = int64(1)
	env.Metrics["files_failed"] = int64(0)

	if err := saveHistoryRecord(stateDir, env); err != nil {
		t.Fatalf("saveHistoryRecord returned error: %v", err)
	}
	db, err := history.Open(stateDir)
	if err != nil {
		t.Fatalf("history.Open returned error: %v", err)
	}
	defer db.Close()
	_, files, _, err := history.GetRun(db, env.RunID, true)
	if err != nil {
		t.Fatalf("history.GetRun returned error: %v", err)
	}
	if len(files) != 1 || files[0].TranscriptPath != artifact || files[0].Transcript != "artifact text" {
		t.Fatalf("history file = %#v", files)
	}
}

func TestDefaultClipboardAndInteractivity(t *testing.T) {
	runCfg := defaultRunConfig()
	if runCfg.Clipboard != "ask" {
		t.Fatalf("defaultRunConfig().Clipboard = %q, want ask", runCfg.Clipboard)
	}
	if runCfg.Engine != "whisper" {
		t.Fatalf("defaultRunConfig().Engine = %q, want whisper", runCfg.Engine)
	}

	global := defaultGlobalOptions()
	if global.NonInteractive {
		t.Fatal("defaultGlobalOptions().NonInteractive = true, want false")
	}
	if global.Output != "text" {
		t.Fatalf("defaultGlobalOptions().Output = %q, want text", global.Output)
	}
}

func TestApplyAgentMode(t *testing.T) {
	global := defaultGlobalOptions()
	global.Agent = true
	cfg := defaultRunConfig()

	applyAgentMode(&global, &cfg)

	if global.Output != "json" {
		t.Fatalf("agent output = %q, want json", global.Output)
	}
	if !global.NonInteractive || !global.Yes {
		t.Fatalf("agent global = %#v, want non-interactive and yes", global)
	}
	if cfg.Clipboard != "never" {
		t.Fatalf("agent clipboard = %q, want never", cfg.Clipboard)
	}
	if cfg.StreamTranscript {
		t.Fatal("agent mode should disable transcript streaming")
	}
}

func TestNormalizeEngine(t *testing.T) {
	if got := normalizeEngine(""); got != "whisper" {
		t.Fatalf("normalizeEngine(\"\") = %q, want whisper", got)
	}
	if got := normalizeEngine("COHERE"); got != "cohere" {
		t.Fatalf("normalizeEngine(\"COHERE\") = %q, want cohere", got)
	}
}

func TestValidateRunInputsRejectsUnknownEngine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Engine = "other"
	_, err := validateRunInputs(cfg, validationOptions{})
	if err == nil || err.Code != "INVALID_ENGINE" {
		t.Fatalf("expected INVALID_ENGINE, got %#v", err)
	}
}

func TestValidateRunInputsRejectsUnsupportedWhisperLanguage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Language = "zzz"
	_, err := validateRunInputs(cfg, validationOptions{})
	if err == nil || err.Code != "UNSUPPORTED_LANGUAGE" {
		t.Fatalf("expected UNSUPPORTED_LANGUAGE, got %#v", err)
	}
}

func TestAggregateRunStatusTreatsDescriptionFailureAsPartialNotFailed(t *testing.T) {
	status := aggregateRunStatus(1, 1, 0)
	if status != "partial" {
		t.Fatalf("aggregateRunStatus(1, 1, 0) = %q, want partial", status)
	}
}

func TestHandleValidateRejectsUnsupportedFileType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "notes.md")
	if err := os.WriteFile(path, []byte("not media"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	env, code := handleValidate([]string{"--state-dir", t.TempDir(), path})
	if code != exitInput {
		t.Fatalf("validate code = %d, want %d; env = %#v", code, exitInput, env)
	}
	if env.Status != "failed" || len(env.Errors) != 1 || env.Errors[0].Code != "UNSUPPORTED_FILE_TYPE" {
		t.Fatalf("validate env = %#v", env)
	}
}

func TestGenerateDescriptionRemovesOutputOnFailure(t *testing.T) {
	falsePath, err := exec.LookPath("false")
	if err != nil {
		t.Skip("false command unavailable")
	}
	dir := t.TempDir()
	transcript := filepath.Join(dir, "transcript.md")
	prompt := filepath.Join(dir, "prompt.md")
	out := filepath.Join(dir, "description.md")
	if err := os.WriteFile(transcript, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
	if err := os.WriteFile(prompt, []byte("summarize"), 0o644); err != nil {
		t.Fatalf("write prompt: %v", err)
	}

	appErr := generateDescription(context.Background(), falsePath, transcript, prompt, out)
	if appErr == nil || appErr.Code != "LLM_DESCRIPTION_FAILED" {
		t.Fatalf("generateDescription error = %#v", appErr)
	}
	if _, err := os.Stat(out); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed description output should be removed, stat err = %v", err)
	}
}

func TestIsAppleSilicon(t *testing.T) {
	got := isAppleSilicon()
	want := runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
	if got != want {
		t.Fatalf("isAppleSilicon() = %v, want %v (GOOS=%s, GOARCH=%s)", got, want, runtime.GOOS, runtime.GOARCH)
	}
}

func TestValidateRunInputsRejectsCohereOnNonAppleSilicon(t *testing.T) {
	if isAppleSilicon() {
		t.Skip("test only runs on non-Apple-Silicon platforms")
	}

	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Engine = "cohere"
	_, err := validateRunInputs(cfg, validationOptions{})
	if err == nil || err.Code != "COHERE_REQUIRES_APPLE_SILICON" {
		t.Fatalf("expected COHERE_REQUIRES_APPLE_SILICON, got %#v", err)
	}
}

func TestValidateRunInputsRejectsCohereTimestamps(t *testing.T) {
	if !isAppleSilicon() {
		t.Skip("cohere validation requires Apple Silicon")
	}

	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Engine = "cohere"
	cfg.Timestamps = true
	_, err := validateRunInputs(cfg, validationOptions{})
	if err == nil || err.Code != "COHERE_TIMESTAMPS_UNSUPPORTED" {
		t.Fatalf("expected COHERE_TIMESTAMPS_UNSUPPORTED, got %#v", err)
	}
}

func TestValidateRunInputsAcceptsCohereLanguageAliases(t *testing.T) {
	if !isAppleSilicon() {
		t.Skip("cohere validation requires Apple Silicon")
	}
	t.Setenv("HF_TOKEN", "test-token")

	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Engine = "cohere"
	cfg.Language = "zh-CN"
	warnings, err := validateRunInputs(cfg, validationOptions{})
	if err != nil {
		t.Fatalf("validateRunInputs returned error: %#v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
}

func TestValidateRunInputsStrictCohereRequiresHFToken(t *testing.T) {
	if !isAppleSilicon() {
		t.Skip("cohere validation requires Apple Silicon")
	}
	t.Setenv("HF_TOKEN", "")

	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Engine = "cohere"
	_, err := validateRunInputs(cfg, validationOptions{Strict: true})
	if err == nil || err.Code != "HF_TOKEN_NOT_SET" {
		t.Fatalf("expected HF_TOKEN_NOT_SET, got %#v", err)
	}
}

func TestHandleValidateReportsNoPromptWhenUnset(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	env, code := handleValidate([]string{"--state-dir", t.TempDir(), path})
	if code != exitOK {
		t.Fatalf("validate code = %d, want %d; env = %#v", code, exitOK, env)
	}
	data, ok := env.Data.(map[string]any)
	if !ok {
		t.Fatalf("validate data = %#v", env.Data)
	}
	checks, ok := data["checks"].(map[string]any)
	if !ok {
		t.Fatalf("validate checks = %#v", data["checks"])
	}
	if got, _ := checks["prompt_exists"].(bool); got {
		t.Fatalf("prompt_exists = true, want false when no prompt was provided")
	}
}

func TestFailureHintsAreSanitized(t *testing.T) {
	ffmpegHint := ffmpegFailureHint("ffmpeg version 8.0\n  built with Apple clang\nconfiguration: lots\ninput.mp3: Invalid data found when processing input")
	if strings.Contains(ffmpegHint, "ffmpeg version") || !strings.Contains(ffmpegHint, "valid audio/video") {
		t.Fatalf("ffmpegFailureHint returned %q", ffmpegHint)
	}

	cohereHint := cohereFailureHint("Fetching 14 files: 0%\nTraceback (most recent call last):\nRepository Not Found for url: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026")
	if strings.Contains(cohereHint, "Traceback") || !strings.Contains(cohereHint, "HF_TOKEN") {
		t.Fatalf("cohereFailureHint returned %q", cohereHint)
	}
}

func TestSelectRuntimeAsset(t *testing.T) {
	manifest := RuntimeManifest{
		Version: "v0.1.0",
		Assets: []RuntimeAsset{
			{Name: "whisper-cli", OS: "darwin", Arch: "arm64", URL: "https://example.invalid/darwin-arm64.tar.gz"},
			{Name: "whisper-cli", OS: "linux", Arch: "x86_64", URL: "https://example.invalid/linux-amd64.tar.gz"},
		},
	}

	asset, err := selectRuntimeAsset(manifest, "linux", "amd64")
	if err != nil {
		t.Fatalf("selectRuntimeAsset returned error: %v", err)
	}
	if asset.URL != "https://example.invalid/linux-amd64.tar.gz" {
		t.Fatalf("unexpected asset URL: %s", asset.URL)
	}
}

func TestVerifyFileSHA256(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sample.txt")
	if err := os.WriteFile(path, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	const helloSHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
	if err := verifyFileSHA256(path, helloSHA256); err != nil {
		t.Fatalf("verifyFileSHA256 should pass: %v", err)
	}
	if err := verifyFileSHA256(path, "0000000000000000000000000000000000000000000000000000000000000000"); err == nil {
		t.Fatal("verifyFileSHA256 should fail for wrong checksum")
	}
}

func TestIsSupportedMediaFile(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{path: "meeting.mp4", want: true},
		{path: "podcast.m4a", want: true},
		{path: "voice.mp3", want: true},
		{path: "screen-recording.mov", want: true},
		{path: "capture.MOV", want: true},
		{path: "notes.wav", want: true},
		{path: "transcript.txt", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			if got := isSupportedMediaFile(tt.path); got != tt.want {
				t.Fatalf("isSupportedMediaFile(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestListInputFilesAcceptsMOV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "Screen Recording.mov")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	files, err := listInputFiles(path)
	if err != nil {
		t.Fatalf("listInputFiles returned error: %#v", err)
	}
	if len(files) != 1 || files[0] != path {
		t.Fatalf("listInputFiles(%q) = %#v, want [%q]", path, files, path)
	}
}

func TestListInputFilesDirectoryIncludesMOV(t *testing.T) {
	dir := t.TempDir()
	want := []string{
		filepath.Join(dir, "a.mov"),
		filepath.Join(dir, "b.mp4"),
	}
	for _, path := range want {
		if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
			t.Fatalf("write file %q: %v", path, err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "ignore.txt"), []byte("fake"), 0o644); err != nil {
		t.Fatalf("write ignore file: %v", err)
	}

	files, err := listInputFiles(dir)
	if err != nil {
		t.Fatalf("listInputFiles returned error: %#v", err)
	}
	if len(files) != len(want) {
		t.Fatalf("listInputFiles(%q) returned %d files, want %d: %#v", dir, len(files), len(want), files)
	}
	for i := range want {
		if files[i] != want[i] {
			t.Fatalf("listInputFiles(%q)[%d] = %q, want %q", dir, i, files[i], want[i])
		}
	}
}

func TestValidateRunInputsInvalidClipboardMode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "meeting.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Clipboard = "sometimes"

	_, err := validateRunInputs(cfg, validationOptions{})
	if err == nil {
		t.Fatal("validateRunInputs should fail for invalid clipboard mode")
	}
	if err.Code != "INVALID_CLIPBOARD_MODE" {
		t.Fatalf("error code = %s, want INVALID_CLIPBOARD_MODE", err.Code)
	}
}

func TestClipboardTranscriptPath(t *testing.T) {
	path, warn := clipboardTranscriptPath([]FileResult{
		{File: "one.mov", Transcript: "/tmp/one.md", Status: "succeeded"},
	})
	if warn != nil {
		t.Fatalf("unexpected warning: %#v", warn)
	}
	if path != "/tmp/one.md" {
		t.Fatalf("clipboardTranscriptPath returned %q, want /tmp/one.md", path)
	}

	path, warn = clipboardTranscriptPath([]FileResult{
		{File: "one.mov", Transcript: "/tmp/one.md", Status: "succeeded"},
		{File: "two.mov", Transcript: "/tmp/two.md", Status: "succeeded"},
	})
	if path != "" {
		t.Fatalf("clipboardTranscriptPath returned %q, want empty path", path)
	}
	if warn == nil || warn.Code != "CLIPBOARD_SKIPPED_MULTIPLE_TRANSCRIPTS" {
		t.Fatalf("expected CLIPBOARD_SKIPPED_MULTIPLE_TRANSCRIPTS, got %#v", warn)
	}
}

func TestClipboardAskSkipsInJSONLWhenNonInteractive(t *testing.T) {
	transcript := filepath.Join(t.TempDir(), "one.md")
	if err := os.WriteFile(transcript, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}

	warnings := maybeHandleClipboard(
		GlobalOptions{Output: "jsonl", NonInteractive: true},
		RunConfig{Clipboard: "ask"},
		[]FileResult{{File: "one.wav", Transcript: transcript, Status: "succeeded"}},
		newProgressReporter("jsonl", "run", "run-1"),
	)
	if len(warnings) != 1 || warnings[0].Code != "CLIPBOARD_PROMPT_SKIPPED" {
		t.Fatalf("warnings = %#v, want CLIPBOARD_PROMPT_SKIPPED", warnings)
	}
}

func TestHistoryRecordStoresTranscriptAndSearches(t *testing.T) {
	stateDir := t.TempDir()
	mediaDir := t.TempDir()
	transcriptPath := filepath.Join(mediaDir, "meeting.md")
	descriptionPath := filepath.Join(mediaDir, "meeting_description.md")
	if err := os.WriteFile(transcriptPath, []byte("Discussed quarterly planning and customer discovery."), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
	if err := os.WriteFile(descriptionPath, []byte("Planning summary."), 0o644); err != nil {
		t.Fatalf("write description: %v", err)
	}

	env := newEnvelope("run")
	env.RunID = "20260224-120000-test"
	env.Status = "succeeded"
	env.Data = RunData{
		Input:    mediaDir,
		Engine:   "whisper",
		ModelRef: "ggml-medium.bin",
		Files: []FileResult{{
			File:        filepath.Join(mediaDir, "meeting.wav"),
			Transcript:  transcriptPath,
			Description: descriptionPath,
			Status:      "succeeded",
		}},
	}
	env.Metrics["duration_ms"] = int64(42)
	env.Metrics["files_total"] = int64(1)
	env.Metrics["files_succeeded"] = int64(1)
	env.Metrics["files_failed"] = int64(0)

	if err := saveHistoryRecord(stateDir, env); err != nil {
		t.Fatalf("saveHistoryRecord returned error: %v", err)
	}

	db, err := history.Open(stateDir)
	if err != nil {
		t.Fatalf("history.Open returned error: %v", err)
	}
	defer db.Close()

	runs, err := history.ListRuns(db, 10, "", nil)
	if err != nil {
		t.Fatalf("history.ListRuns returned error: %v", err)
	}
	if len(runs) != 1 || runs[0].RunID != env.RunID {
		t.Fatalf("history runs = %#v, want run_id %s", runs, env.RunID)
	}

	run, files, _, err := history.GetRun(db, env.RunID, true)
	if err != nil {
		t.Fatalf("history.GetRun returned error: %v", err)
	}
	if run.Engine != "whisper" || run.FilesSucceeded != 1 {
		t.Fatalf("history run = %#v", run)
	}
	if len(files) != 1 || !strings.Contains(files[0].Transcript, "quarterly planning") {
		t.Fatalf("history files = %#v", files)
	}

	matches, err := history.Search(db, "customer discovery", 5, nil, false)
	if err != nil {
		t.Fatalf("history.Search returned error: %v", err)
	}
	if len(matches) != 1 || matches[0].RunID != env.RunID {
		t.Fatalf("history matches = %#v, want run_id %s", matches, env.RunID)
	}
	if matches[0].Transcript != "" || !strings.Contains(matches[0].TranscriptPreview, "quarterly planning") || matches[0].TranscriptChars == 0 {
		t.Fatalf("compact history match = %#v", matches[0])
	}
	fullMatches, err := history.Search(db, "customer discovery", 5, nil, true)
	if err != nil {
		t.Fatalf("history.Search include transcript returned error: %v", err)
	}
	if len(fullMatches) != 1 || !strings.Contains(fullMatches[0].Transcript, "quarterly planning") {
		t.Fatalf("full history matches = %#v", fullMatches)
	}

	latest, err := history.LatestRunID(db)
	if err != nil {
		t.Fatalf("history.LatestRunID returned error: %v", err)
	}
	if latest != env.RunID {
		t.Fatalf("latest run = %q, want %q", latest, env.RunID)
	}

	schema, err := history.Schema(db)
	if err != nil {
		t.Fatalf("history.Schema returned error: %v", err)
	}
	if _, ok := schema["runs"]; !ok {
		t.Fatalf("schema missing runs table: %#v", schema)
	}

	md := history.ExportMarkdown(run, files)
	if !strings.Contains(md, "Scriby Run") || !strings.Contains(md, "quarterly planning") {
		t.Fatalf("markdown export = %q", md)
	}
}

func TestHistorySQLReadOnlyGuard(t *testing.T) {
	if !history.LooksReadOnlySQL("select run_id from runs") {
		t.Fatal("select should be accepted")
	}
	if !history.LooksReadOnlySQL("WITH recent AS (select * from runs) select * from recent") {
		t.Fatal("with query should be accepted")
	}
	if history.LooksReadOnlySQL("delete from runs") {
		t.Fatal("delete should be rejected")
	}
	if history.LooksReadOnlySQL("select * from runs; delete from runs") {
		t.Fatal("multi-statement query should be rejected")
	}
}

func TestHistorySQLReadOnlyConnectionBlocksWithMutation(t *testing.T) {
	stateDir := t.TempDir()
	db, err := history.Open(stateDir)
	if err != nil {
		t.Fatalf("history.Open returned error: %v", err)
	}
	_, err = db.Exec(`INSERT INTO runs (
		run_id, created_at, command, status, envelope_json
	) VALUES ('seed', '2026-06-08T00:00:00Z', 'run', 'succeeded', '{}')`)
	_ = db.Close()
	if err != nil {
		t.Fatalf("seed insert returned error: %v", err)
	}

	ro, err := history.OpenReadOnly(stateDir)
	if err != nil {
		t.Fatalf("history.OpenReadOnly returned error: %v", err)
	}
	defer ro.Close()

	if _, err := history.QuerySQL(ro, `WITH t AS (SELECT 1) DELETE FROM runs`); err == nil {
		t.Fatal("read-only history SQL should reject WITH ... DELETE")
	}
	if _, err := history.QuerySQL(ro, `WITH t AS (SELECT 1) INSERT INTO runs (
		run_id, created_at, command, status, envelope_json
	) VALUES ('mutated', '2026-06-08T00:00:00Z', 'run', 'succeeded', '{}')`); err == nil {
		t.Fatal("read-only history SQL should reject WITH ... INSERT")
	}

	check, err := history.Open(stateDir)
	if err != nil {
		t.Fatalf("history.Open returned error: %v", err)
	}
	defer check.Close()
	var count int
	if err := check.QueryRow(`SELECT count(*) FROM runs`).Scan(&count); err != nil {
		t.Fatalf("count query returned error: %v", err)
	}
	if count != 1 {
		t.Fatalf("runs count = %d, want 1", count)
	}
}

func TestHandleHistoryLatestExportSchema(t *testing.T) {
	stateDir := t.TempDir()
	mediaDir := t.TempDir()
	transcriptPath := filepath.Join(mediaDir, "meeting.md")
	if err := os.WriteFile(transcriptPath, []byte("Latest transcript for agent workflows."), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}

	env := newEnvelope("run")
	env.RunID = "20260224-130000-test"
	env.Status = "succeeded"
	env.Data = RunData{
		Input:            mediaDir,
		Engine:           "whisper",
		Language:         "en",
		SampleRate:       16000,
		StreamTranscript: false,
		Files: []FileResult{{
			File:       filepath.Join(mediaDir, "meeting.wav"),
			Transcript: transcriptPath,
			Status:     "succeeded",
		}},
	}
	env.Metrics["duration_ms"] = int64(10)
	env.Metrics["files_total"] = int64(1)
	env.Metrics["files_succeeded"] = int64(1)
	env.Metrics["files_failed"] = int64(0)
	if err := saveHistoryRecord(stateDir, env); err != nil {
		t.Fatalf("saveHistoryRecord returned error: %v", err)
	}

	compactEnv, code := handleHistory([]string{"--agent", "latest", "--state-dir", stateDir})
	if code != exitOK {
		t.Fatalf("history latest compact code = %d env = %#v", code, compactEnv)
	}
	compactData, ok := compactEnv.Data.(map[string]any)
	if !ok {
		t.Fatalf("compact latest data = %#v", compactEnv.Data)
	}
	compactFiles, ok := compactData["files"].([]history.Transcription)
	if !ok || len(compactFiles) != 1 {
		t.Fatalf("compact files = %#v", compactData["files"])
	}
	if compactFiles[0].Transcript != "" || !strings.Contains(compactFiles[0].TranscriptPreview, "Latest transcript") {
		t.Fatalf("compact latest should omit full transcript and include preview: %#v", compactFiles[0])
	}

	latestEnv, code := handleHistory([]string{"--agent", "latest", "--state-dir", stateDir, "--transcript-only"})
	if code != exitOK {
		t.Fatalf("history latest code = %d env = %#v", code, latestEnv)
	}
	latestData, ok := latestEnv.Data.(map[string]any)
	if !ok || latestData["run_id"] != env.RunID {
		t.Fatalf("latest data = %#v", latestEnv.Data)
	}

	exportEnv, code := handleHistory([]string{"export", "--state-dir", stateDir, "--latest", "--format", "markdown"})
	if code != exitOK {
		t.Fatalf("history export code = %d env = %#v", code, exportEnv)
	}
	exportData, ok := exportEnv.Data.(map[string]any)
	if !ok || !strings.Contains(exportData["content"].(string), "Latest transcript") {
		t.Fatalf("export data = %#v", exportEnv.Data)
	}

	schemaEnv, code := handleHistory([]string{"schema", "--state-dir", stateDir})
	if code != exitOK {
		t.Fatalf("history schema code = %d env = %#v", code, schemaEnv)
	}
	schemaData, ok := schemaEnv.Data.(map[string]any)
	if !ok || schemaData["schema"] == nil {
		t.Fatalf("schema data = %#v", schemaEnv.Data)
	}
}

func TestParseSince(t *testing.T) {
	now := time.Date(2026, 6, 8, 12, 0, 0, 0, time.UTC)
	got, err := history.ParseSince("7d", now)
	if err != nil {
		t.Fatalf("history.ParseSince returned error: %v", err)
	}
	want := now.Add(-7 * 24 * time.Hour)
	if !got.Equal(want) {
		t.Fatalf("history.ParseSince(7d) = %s, want %s", got, want)
	}
	if _, err := history.ParseSince("nope", now); err == nil {
		t.Fatal("history.ParseSince should reject invalid values")
	}
}

func TestRetryRunArgsAndFailedInputs(t *testing.T) {
	global := defaultGlobalOptions()
	global.Agent = true
	global.Output = "json"
	global.NonInteractive = true
	global.Yes = true
	global.StateDir = "/tmp/scriby-state"
	data := RunData{
		Input:            "/tmp/input",
		Engine:           "whisper",
		Language:         "en",
		Clipboard:        "ask",
		MonoMode:         "average",
		SampleRate:       16000,
		StreamTranscript: true,
		ModelName:        "medium",
		Files: []FileResult{
			{File: "/tmp/a.wav", Status: "failed"},
			{File: "/tmp/b.wav", Status: "succeeded"},
		},
	}

	inputs := failedRunInputs(data.Files)
	if len(inputs) != 1 || inputs[0] != "/tmp/a.wav" {
		t.Fatalf("failedRunInputs = %#v", inputs)
	}

	args := retryRunArgs(global, data, inputs[0])
	joined := strings.Join(args, " ")
	for _, want := range []string{"--agent", "--clipboard never", "--stream-transcript=false", "/tmp/a.wav"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("retry args %q missing %q", joined, want)
		}
	}
}

func TestPromptYesNo(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{name: "yes", input: "yes\n", want: true},
		{name: "short yes", input: "y\n", want: true},
		{name: "no", input: "n\n", want: false},
		{name: "empty", input: "", want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			got, err := promptYesNo(strings.NewReader(tt.input), &out, "Copy? ")
			if err != nil {
				t.Fatalf("promptYesNo returned error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("promptYesNo(%q) = %v, want %v", tt.input, got, tt.want)
			}
			if out.String() != "Copy? " {
				t.Fatalf("prompt output = %q, want %q", out.String(), "Copy? ")
			}
		})
	}
}

func TestClipboardCommandFor(t *testing.T) {
	lookPath := func(paths map[string]string) func(string) (string, error) {
		return func(name string) (string, error) {
			if path, ok := paths[name]; ok {
				return path, nil
			}
			return "", fmt.Errorf("%s not found", name)
		}
	}
	getenv := func(values map[string]string) func(string) string {
		return func(key string) string {
			return values[key]
		}
	}

	tests := []struct {
		name     string
		goos     string
		env      map[string]string
		paths    map[string]string
		wantPath string
		wantArgs []string
		wantErr  string
	}{
		{
			name:     "darwin pbcopy",
			goos:     "darwin",
			paths:    map[string]string{"pbcopy": "/usr/bin/pbcopy"},
			wantPath: "/usr/bin/pbcopy",
		},
		{
			name:     "windows clip",
			goos:     "windows",
			paths:    map[string]string{"clip.exe": "C:\\Windows\\System32\\clip.exe"},
			wantPath: "C:\\Windows\\System32\\clip.exe",
		},
		{
			name:     "linux wayland wl-copy",
			goos:     "linux",
			env:      map[string]string{"WAYLAND_DISPLAY": "wayland-1"},
			paths:    map[string]string{"wl-copy": "/usr/bin/wl-copy"},
			wantPath: "/usr/bin/wl-copy",
		},
		{
			name:     "linux xclip fallback",
			goos:     "linux",
			paths:    map[string]string{"xclip": "/usr/bin/xclip"},
			wantPath: "/usr/bin/xclip",
			wantArgs: []string{"-selection", "clipboard"},
		},
		{
			name:    "linux unavailable",
			goos:    "linux",
			paths:   map[string]string{},
			wantErr: "clipboard unavailable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd, err := clipboard.CommandFor(tt.goos, getenv(tt.env), lookPath(tt.paths))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("clipboard.CommandFor error = %v, want substring %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("clipboard.CommandFor returned error: %v", err)
			}
			if cmd.Path != tt.wantPath {
				t.Fatalf("cmd.Path = %q, want %q", cmd.Path, tt.wantPath)
			}
			if strings.Join(cmd.Args, " ") != strings.Join(tt.wantArgs, " ") {
				t.Fatalf("cmd.Args = %#v, want %#v", cmd.Args, tt.wantArgs)
			}
		})
	}
}

func TestSplitRootArgs(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		wantLeading []string
		wantRest    []string
		wantHelp    bool
		wantErr     string
	}{
		{
			name:        "command first",
			args:        []string{"run", "--help"},
			wantLeading: []string{},
			wantRest:    []string{"run", "--help"},
		},
		{
			name:        "leading global flags",
			args:        []string{"--output", "text", "--non-interactive", "run", "clip.wav"},
			wantLeading: []string{"--output", "text", "--non-interactive"},
			wantRest:    []string{"run", "clip.wav"},
		},
		{
			name:        "root help with output",
			args:        []string{"--output=json", "--help"},
			wantLeading: []string{"--output=json", "--help"},
			wantRest:    nil,
			wantHelp:    true,
		},
		{
			name:    "unknown leading flag",
			args:    []string{"--bogus", "run"},
			wantErr: "flag provided but not defined: -bogus",
		},
		{
			name:    "missing flag value",
			args:    []string{"--output"},
			wantErr: "flag needs an argument: --output",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotLeading, gotRest, gotHelp, err := splitRootArgs(tt.args)
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("splitRootArgs(%q) error = %v, want %q", tt.args, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("splitRootArgs(%q) returned error: %v", tt.args, err)
			}
			if fmt.Sprintf("%q", gotLeading) != fmt.Sprintf("%q", tt.wantLeading) {
				t.Fatalf("leading = %q, want %q", gotLeading, tt.wantLeading)
			}
			if fmt.Sprintf("%q", gotRest) != fmt.Sprintf("%q", tt.wantRest) {
				t.Fatalf("rest = %q, want %q", gotRest, tt.wantRest)
			}
			if gotHelp != tt.wantHelp {
				t.Fatalf("help = %v, want %v", gotHelp, tt.wantHelp)
			}
		})
	}
}

func TestOutputPreference(t *testing.T) {
	t.Setenv("SCRIBY_OUTPUT", "")

	mode, source := outputPreference([]string{"--output", "text", "run"})
	if mode != "text" || source != "flag" {
		t.Fatalf("outputPreference flag = (%q, %q), want (text, flag)", mode, source)
	}

	t.Setenv("SCRIBY_OUTPUT", "jsonl")
	mode, source = outputPreference([]string{"run"})
	if mode != "jsonl" || source != "env" {
		t.Fatalf("outputPreference env = (%q, %q), want (jsonl, env)", mode, source)
	}

	t.Setenv("SCRIBY_OUTPUT", "")
	t.Setenv("SCRIBY_AGENT", "1")
	mode, source = outputPreference([]string{"run"})
	if mode != "json" || source != "env" {
		t.Fatalf("outputPreference SCRIBY_AGENT = (%q, %q), want (json, env)", mode, source)
	}

	t.Setenv("SCRIBY_AGENT", "")
	mode, source = outputPreference([]string{"run"})
	if mode != "text" || source != "default" {
		t.Fatalf("outputPreference default = (%q, %q), want (text, default)", mode, source)
	}
}

func TestHoistGlobalFlagsAllowsTrailingAgent(t *testing.T) {
	got := hoistGlobalFlags([]string{"developer tools", "--agent"})
	want := []string{"--agent", "developer tools"}
	if fmt.Sprintf("%q", got) != fmt.Sprintf("%q", want) {
		t.Fatalf("hoistGlobalFlags = %q, want %q", got, want)
	}
}

func TestHoistRunFlagsAllowsTrailingFlags(t *testing.T) {
	got := hoistRunFlags([]string{"meeting.wav", "--clipboard", "never", "--language=en"})
	want := []string{"--clipboard", "never", "--language=en", "meeting.wav"}
	if fmt.Sprintf("%q", got) != fmt.Sprintf("%q", want) {
		t.Fatalf("hoistRunFlags = %q, want %q", got, want)
	}
}

func TestHoistRetryFlagsAllowsTrailingFailedOnly(t *testing.T) {
	got := hoistRetryFlags([]string{"run-123", "--failed-only"})
	want := []string{"--failed-only", "run-123"}
	if fmt.Sprintf("%q", got) != fmt.Sprintf("%q", want) {
		t.Fatalf("hoistRetryFlags = %q, want %q", got, want)
	}
}

func TestHandleRunAcceptsFlagsAfterInput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	env, code := handleRun([]string{path, "--language", "zzz", "--clipboard", "never", "--state-dir", t.TempDir()})
	if code != exitInput {
		t.Fatalf("run code = %d, want %d; env = %#v", code, exitInput, env)
	}
	if env.Status != "failed" || len(env.Errors) != 1 || env.Errors[0].Code != "UNSUPPORTED_LANGUAGE" {
		t.Fatalf("run env = %#v", env)
	}
}

func TestHistorySearchAcceptsTrailingAgent(t *testing.T) {
	stateDir := t.TempDir()
	env := newEnvelope("run")
	env.RunID = "20260224-130000-search"
	env.Status = "succeeded"
	env.Data = RunData{
		Input:  "meeting.wav",
		Engine: "whisper",
		Files: []FileResult{{
			File:       "meeting.wav",
			Transcript: "missing.md",
			Status:     "succeeded",
		}},
	}
	env.Metrics["duration_ms"] = int64(10)
	env.Metrics["files_total"] = int64(1)
	env.Metrics["files_succeeded"] = int64(1)
	env.Metrics["files_failed"] = int64(0)
	if err := history.Save(stateDir, history.Record{
		RunID:          env.RunID,
		CreatedAt:      "2026-02-24T13:00:00Z",
		Command:        "run",
		Status:         "succeeded",
		Input:          "meeting.wav",
		Engine:         "whisper",
		EnvelopeJSON:   `{"schema_version":"1.0"}`,
		DurationMS:     10,
		FilesTotal:     1,
		FilesSucceeded: 1,
		Files: []history.FileRecord{{
			File:       "meeting.wav",
			Status:     "succeeded",
			Transcript: "developer tools strategy",
		}},
	}); err != nil {
		t.Fatalf("save history: %v", err)
	}

	got, code := handleHistory([]string{"search", "developer tools", "--agent", "--state-dir", stateDir})
	if code != exitOK {
		t.Fatalf("history search code = %d env = %#v", code, got)
	}
	data, ok := got.Data.(map[string]any)
	if !ok {
		t.Fatalf("history search data = %T", got.Data)
	}
	matches, ok := data["matches"].([]history.Transcription)
	if !ok || len(matches) != 1 {
		t.Fatalf("matches = %#v", data["matches"])
	}
	if matches[0].Transcript != "" || !strings.Contains(matches[0].TranscriptPreview, "developer tools") {
		t.Fatalf("compact agent match = %#v", matches[0])
	}
}

func TestHistoryShowAndExportAcceptFlagsAfterRunID(t *testing.T) {
	stateDir := t.TempDir()
	env := newEnvelope("run")
	env.RunID = "20260224-130000-export"
	env.Status = "succeeded"
	env.Data = RunData{
		Input:  "meeting.wav",
		Engine: "whisper",
		Files: []FileResult{{
			File:       "meeting.wav",
			Transcript: "missing.md",
			Status:     "succeeded",
		}},
	}
	env.Metrics["duration_ms"] = int64(10)
	env.Metrics["files_total"] = int64(1)
	env.Metrics["files_succeeded"] = int64(1)
	env.Metrics["files_failed"] = int64(0)
	if err := history.Save(stateDir, history.Record{
		RunID:          env.RunID,
		CreatedAt:      "2026-02-24T13:00:00Z",
		Command:        "run",
		Status:         "succeeded",
		Input:          "meeting.wav",
		Engine:         "whisper",
		EnvelopeJSON:   `{"schema_version":"1.0"}`,
		DurationMS:     10,
		FilesTotal:     1,
		FilesSucceeded: 1,
		Files: []history.FileRecord{{
			File:       "meeting.wav",
			Status:     "succeeded",
			Transcript: "developer tools strategy",
		}},
	}); err != nil {
		t.Fatalf("save history: %v", err)
	}

	showEnv, code := handleHistory([]string{"show", env.RunID, "--include-transcript", "--state-dir", stateDir})
	if code != exitOK {
		t.Fatalf("history show code = %d env = %#v", code, showEnv)
	}
	showData, ok := showEnv.Data.(map[string]any)
	if !ok {
		t.Fatalf("history show data = %T", showEnv.Data)
	}
	files, ok := showData["files"].([]history.Transcription)
	if !ok || len(files) != 1 || files[0].Transcript != "developer tools strategy" {
		t.Fatalf("history show files = %#v", showData["files"])
	}

	exportEnv, code := handleHistory([]string{"export", env.RunID, "--format", "markdown", "--state-dir", stateDir})
	if code != exitOK {
		t.Fatalf("history export code = %d env = %#v", code, exportEnv)
	}
	exportData, ok := exportEnv.Data.(map[string]any)
	if !ok {
		t.Fatalf("history export data = %T", exportEnv.Data)
	}
	content, _ := exportData["content"].(string)
	if !strings.Contains(content, "developer tools strategy") {
		t.Fatalf("history export content = %q", content)
	}
}

func TestHistoryExportTextPrintsMarkdown(t *testing.T) {
	env := newEnvelope("history.export")
	env.Data = map[string]any{
		"run_id":  "run-1",
		"format":  "markdown",
		"content": "# Scriby Run run-1\n\nTranscript body.\n",
	}
	finishEnvelope(&env, time.Now(), 0, 0, 0)

	out := captureStdout(t, func() {
		if err := printEnvelope(env, "text"); err != nil {
			t.Fatalf("printEnvelope returned error: %v", err)
		}
	})
	if !strings.HasPrefix(out, "# Scriby Run run-1\n") {
		t.Fatalf("text export output = %q", out)
	}
	if strings.Contains(out, `"schema_version"`) || strings.Contains(out, `"content"`) {
		t.Fatalf("text export should not print JSON envelope: %q", out)
	}
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	fn()
	if err := w.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}
	os.Stdout = old
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	return string(b)
}
