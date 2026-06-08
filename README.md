# Scriby

AI-native CLI for media transcription with `whisper-cli` by default, optional Cohere Transcribe via mlx-audio on Apple Silicon, optional description generation via `llm`, and deterministic machine-parseable output.

## Install from GitHub Release (recommended)

Releases:

`https://github.com/nibzard/scriby/releases`

Quick install (macOS/Linux):

```bash
VERSION="v2.1.4"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH="amd64" ;;
  arm64|aarch64) ARCH="arm64" ;;
esac

curl -L -o /tmp/scriby.tar.gz "https://github.com/nibzard/scriby/releases/download/${VERSION}/scriby-${VERSION}-${OS}-${ARCH}.tar.gz"
tar -xzf /tmp/scriby.tar.gz -C /tmp
sudo install /tmp/scriby /usr/local/bin/scriby
scriby --help
```

macOS Apple Silicon (`arm64`) direct example:

```bash
curl -L -o /tmp/scriby.tar.gz "https://github.com/nibzard/scriby/releases/download/v2.1.4/scriby-v2.1.4-darwin-arm64.tar.gz"
tar -xzf /tmp/scriby.tar.gz -C /tmp
sudo install /tmp/scriby /usr/local/bin/scriby
scriby --help
```

## Quickstart (first setup + first run)

Prerequisite:

```bash
ffmpeg -version
```

First run (downloads Whisper runtime + model automatically):

```bash
scriby run \
  --model medium \
  --language en \
  --stream-transcript=false \
  /path/to/audio-or-video-file
```

What happens on first run:

- Scriby creates durable local state in `~/.scriby`.
- Scriby installs `whisper-cli` to `~/.scriby/runtime`.
- Scriby downloads the selected model to `~/.scriby/models`.
- Transcript history is written to an immutable run folder under `~/.scriby/runs/<run_id>/`.
- By default, Scriby also writes a latest convenience copy next to your input file as `<name>.md`.
- Run metadata plus transcript/description text are indexed in `~/.scriby/scriby.db`.
- If an older cache state exists, Scriby copies it into `~/.scriby` on first use.

Artifact modes:

```bash
scriby run ./meeting.wav --artifact-mode both       # default: immutable history + latest copy
scriby run ./meeting.wav --artifact-mode versioned  # only ~/.scriby/runs/<run_id>/ artifacts
scriby run ./meeting.wav --artifact-mode latest     # only latest output files
scriby run ./meeting.wav --output-dir ./out         # latest copies go to ./out
scriby run ./meeting.wav --artifact-mode versioned --output-dir ./out  # versioned artifacts go to ./out/runs/<run_id>/
```

Example:

```bash
scriby run --model medium --language en --stream-transcript=false ./meeting.wav
```

Agent-friendly run:

```bash
scriby run --agent ./meeting.wav
```

`--agent` is a shortcut for machine use: JSON output, non-interactive prompts, `--yes`, no clipboard prompt during `run`, and compact transcript handling.

## Cohere Transcribe engine

Scriby can run `CohereLabs/cohere-transcribe-03-2026` via [mlx-audio](https://github.com/Blaizzy/mlx-audio) on Apple Silicon. Scriby uses [uv](https://docs.astral.sh/uv/) to manage the Python runtime and dependencies automatically.

Requirements:

- Apple Silicon Mac (M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/) installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Accept the model access conditions on Hugging Face for `CohereLabs/cohere-transcribe-03-2026`
- Export `HF_TOKEN` or configure Hugging Face auth locally

Run with the Cohere engine:

```bash
export HF_TOKEN=...
scriby run \
  --engine cohere \
  --language en \
  --clipboard never \
  ./meeting.wav
```

Tested flow with Cohere's demo audio:

```bash
curl -fsSL \
  -H "Authorization: Bearer $HF_TOKEN" \
  -o /tmp/voxpopuli_test_en_demo.wav \
  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/resolve/main/demo/voxpopuli_test_en_demo.wav

scriby run \
  --engine cohere \
  --language en \
  --clipboard never \
  /tmp/voxpopuli_test_en_demo.wav
```

Notes:

- `--timestamps` is not supported by the Cohere engine.
- `scriby models ...` still manages Whisper `ggml-*` files only.
- First run may take a while because uv installs mlx-audio and downloads model weights from Hugging Face.

Supported input formats include `.mp4`, `.m4a`, `.mp3`, `.mov`, and `.wav`.

## History database

Scriby keeps a local SQLite history database at:

```bash
~/.scriby/scriby.db
```

The database stores run metadata and the transcript/description text captured during each run, while still leaving Markdown files next to the original media.

Useful commands:

```bash
scriby history path
scriby history list --limit 10
scriby history list --since 7d
scriby history latest --transcript-only
scriby history show <run_id>
scriby history search "customer discovery"
scriby history search "customer discovery" --since 7d
scriby history export --latest --format markdown
scriby history schema
scriby history sql "select run_id, status, input from runs order by created_at desc limit 5"
```

Override the location with `--state-dir <path>` or `SCRIBY_STATE_DIR`.
`history sql` opens the database through a read-only SQLite connection and accepts only single-statement `SELECT`, `WITH`, or `PRAGMA` forms as a pre-filter.

Recovery:

```bash
scriby retry <run_id>
scriby retry <run_id> --failed-only --agent
```

Clipboard prompting is enabled by default for interactive runs. Use these overrides when needed:

```bash
scriby run --clipboard always /path/to/file.mov
scriby run --clipboard never /path/to/file.mov
scriby run --non-interactive /path/to/file.mov
```

## Build from source

```bash
make build
./scriby run /path/to/file-or-directory
./scriby validate /path/to/file-or-directory
```

## Release and runtime packaging

Scriby release binaries:

```bash
make test
make dist VERSION=v2.1.4
```

This generates cross-platform archives under `dist/scriby`.

Whisper runtime assets + manifest (published in `nibzard/scriby`):

```bash
make runtime-assets VERSION=v2.1.4
```

This packages `runtime/bin/<os>_<arch>/whisper-cli(.exe)` into `dist/runtime` and generates:

- `dist/runtime/runtime-manifest.json`
- checksummed platform runtime artifacts (tar.gz/zip)

GitHub Actions workflows:

- `.github/workflows/release.yml` publishes Scriby CLI assets on `v*` tags.
- `.github/workflows/runtime-release.yml` builds `whisper-cli` matrix artifacts, generates `runtime-manifest.json`, and publishes runtime assets.

## Runtime bootstrap source

By default, Scriby resolves runtime assets from:

`https://github.com/nibzard/scriby/releases/download/v2.1.4/runtime-manifest.json`

Override when needed:

```bash
./scriby run --runtime-manifest-url <manifest-url> <input>
```
