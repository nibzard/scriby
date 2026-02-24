# Scriby

AI-native CLI for media transcription with `whisper-cli`, optional description generation via `llm`, and deterministic machine-parseable output.

## Quick Start

Build:

```bash
make build
```

Run:

```bash
./scriby run /path/to/file-or-directory
```

Validate only:

```bash
./scriby validate /path/to/file-or-directory
```

## Release And Runtime Packaging

Scriby release binaries:

```bash
make test
make dist VERSION=v0.1.0
```

This generates cross-platform archives under `dist/scriby`.

Whisper runtime assets + manifest (published in `nibzard/scriby`):

```bash
make runtime-assets VERSION=v0.1.0
```

This packages `runtime/bin/<os>_<arch>/whisper-cli(.exe)` into `dist/runtime` and generates:

- `dist/runtime/runtime-manifest.json`
- checksummed platform runtime artifacts (tar.gz/zip)

GitHub Actions workflows:

- `.github/workflows/release.yml` publishes Scriby CLI assets on `v*` tags.
- `.github/workflows/runtime-release.yml` builds `whisper-cli` matrix artifacts, generates `runtime-manifest.json`, and publishes runtime assets.

## Runtime Bootstrap Source

By default, Scriby resolves runtime assets from:

`https://github.com/nibzard/scriby/releases/download/v0.1.0/runtime-manifest.json`

Override when needed:

```bash
./scriby run --runtime-manifest-url <manifest-url> <input>
```
