package main

import (
	"os"
	"path/filepath"
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
