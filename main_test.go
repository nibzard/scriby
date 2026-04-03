package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	_, err := validateRunInputs(cfg)
	if err == nil || err.Code != "INVALID_ENGINE" {
		t.Fatalf("expected INVALID_ENGINE, got %#v", err)
	}
}

func TestValidateRunInputsRejectsCohereTimestamps(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Engine = "cohere"
	cfg.Timestamps = true
	_, err := validateRunInputs(cfg)
	if err == nil || err.Code != "COHERE_TIMESTAMPS_UNSUPPORTED" {
		t.Fatalf("expected COHERE_TIMESTAMPS_UNSUPPORTED, got %#v", err)
	}
}

func TestValidateRunInputsAcceptsCohereLanguageAliases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.wav")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	cfg := defaultRunConfig()
	cfg.Input = path
	cfg.Engine = "cohere"
	cfg.Language = "zh-CN"
	warnings, err := validateRunInputs(cfg)
	if err != nil {
		t.Fatalf("validateRunInputs returned error: %#v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
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

	_, err := validateRunInputs(cfg)
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
			cmd, err := clipboardCommandFor(tt.goos, getenv(tt.env), lookPath(tt.paths))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("clipboardCommandFor error = %v, want substring %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("clipboardCommandFor returned error: %v", err)
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
	mode, source = outputPreference([]string{"run"})
	if mode != "json" || source != "default" {
		t.Fatalf("outputPreference default = (%q, %q), want (json, default)", mode, source)
	}
}
