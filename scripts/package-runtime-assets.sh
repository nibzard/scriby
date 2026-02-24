#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
OUT_DIR="${2:-dist/runtime}"
BASE_URL="${3:-}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <version> [output_dir] [base_url]" >&2
  exit 1
fi

if [[ -z "$BASE_URL" ]]; then
  BASE_URL="https://github.com/nibzard/scriby/releases/download/${VERSION}"
fi

TARGETS=(
  "darwin/amd64"
  "darwin/arm64"
  "linux/amd64"
  "linux/arm64"
  "windows/amd64"
)

hash_file() {
  local file="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    sha256sum "$file" | awk '{print $1}'
  fi
}

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$OUT_DIR"
assets_json=""
sep=""

for target in "${TARGETS[@]}"; do
  os="${target%/*}"
  arch="${target#*/}"

  src="runtime/bin/${os}_${arch}/whisper-cli"
  format="tar.gz"
  binary="whisper-cli"
  artifact="whisper-cli-${os}-${arch}.tar.gz"

  if [[ "$os" == "windows" ]]; then
    src="runtime/bin/${os}_${arch}/whisper-cli.exe"
    format="zip"
    binary="whisper-cli.exe"
    artifact="whisper-cli-${os}-${arch}.zip"
  fi

  if [[ ! -f "$src" ]]; then
    echo "Missing runtime binary: $src" >&2
    exit 1
  fi

  stage="$WORK_DIR/${os}_${arch}"
  mkdir -p "$stage"
  cp "$src" "$stage/$binary"

  if [[ "$format" == "zip" ]]; then
    ( cd "$stage" && zip -q "$OLDPWD/$OUT_DIR/$artifact" "$binary" )
  else
    ( cd "$stage" && tar -czf "$OLDPWD/$OUT_DIR/$artifact" "$binary" )
  fi

  sha256="$(hash_file "$OUT_DIR/$artifact")"
  assets_json+="${sep}{\"name\":\"whisper-cli\",\"os\":\"${os}\",\"arch\":\"${arch}\",\"format\":\"${format}\",\"binary\":\"${binary}\",\"url\":\"${BASE_URL}/${artifact}\",\"sha256\":\"${sha256}\"}"
  sep=","
done

generated_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
cat > "$OUT_DIR/runtime-manifest.json" <<JSON
{
  "version": "${VERSION}",
  "generated_at": "${generated_at}",
  "assets": [${assets_json}]
}
JSON

echo "Runtime assets packaged in $OUT_DIR" >&2
echo "Manifest: $OUT_DIR/runtime-manifest.json" >&2
