# Scriby

AI-native CLI for media transcription with `whisper-cli` by default, optional Cohere Transcribe via Hugging Face, optional description generation via `llm`, and deterministic machine-parseable output.

## Install from GitHub Release (recommended)

Releases:

`https://github.com/nibzard/scriby/releases`

Quick install (macOS/Linux):

```bash
VERSION="v0.1.2"
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
curl -L -o /tmp/scriby.tar.gz "https://github.com/nibzard/scriby/releases/download/v0.1.2/scriby-v0.1.2-darwin-arm64.tar.gz"
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

- Scriby installs `whisper-cli` to `~/Library/Caches/scriby/runtime` on macOS.
- Scriby downloads the selected model to `~/Library/Caches/scriby/models`.
- Transcript is written next to your input file as `<name>.md`.

Example:

```bash
scriby run --model medium --language en --stream-transcript=false ./meeting.wav
```

## Cohere Transcribe engine

Scriby can also run `CohereLabs/cohere-transcribe-03-2026` with an embedded Python helper.

Requirements:

- Accept the model access conditions on Hugging Face for `CohereLabs/cohere-transcribe-03-2026`
- Configure Hugging Face auth locally or export `HF_TOKEN`
- Use a Python version with current `torch` wheels available. `python3.12` worked in local testing.
- Install Python dependencies:

```bash
python3 -m pip install "transformers>=5.4.0" torch huggingface_hub soundfile librosa sentencepiece protobuf
```

Run with the Cohere engine:

```bash
scriby run \
  --engine cohere \
  --python-path /path/to/python3.12 \
  --language en \
  --hf-model-id CohereLabs/cohere-transcribe-03-2026 \
  --clipboard never \
  ./meeting.wav
```

Notes:

- `--timestamps` is not supported by the Cohere engine.
- `scriby models ...` still manages Whisper `ggml-*` files only.
- The helper writes `cohere_transcribe.py` into Scriby's runtime cache on first use.
- First run may take a while because Hugging Face model weights need to be downloaded and loaded.

Supported input formats include `.mp4`, `.m4a`, `.mp3`, `.mov`, and `.wav`.

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
make dist VERSION=v0.1.2
```

This generates cross-platform archives under `dist/scriby`.

Whisper runtime assets + manifest (published in `nibzard/scriby`):

```bash
make runtime-assets VERSION=v0.1.2
```

This packages `runtime/bin/<os>_<arch>/whisper-cli(.exe)` into `dist/runtime` and generates:

- `dist/runtime/runtime-manifest.json`
- checksummed platform runtime artifacts (tar.gz/zip)

GitHub Actions workflows:

- `.github/workflows/release.yml` publishes Scriby CLI assets on `v*` tags.
- `.github/workflows/runtime-release.yml` builds `whisper-cli` matrix artifacts, generates `runtime-manifest.json`, and publishes runtime assets.

## Runtime bootstrap source

By default, Scriby resolves runtime assets from:

`https://github.com/nibzard/scriby/releases/download/v0.1.2/runtime-manifest.json`

Override when needed:

```bash
./scriby run --runtime-manifest-url <manifest-url> <input>
```
