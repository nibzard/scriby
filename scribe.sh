#!/usr/bin/env bash
set -euo pipefail

#########################################################################
# scribe
#
# 1) Converts any audio/video file (mp4, m4a, mp3, wav, etc.) → 16 kHz mono WAV
# 2) Uses whisper.cpp (via whisper-cli) to produce a text transcript
#    – transcript is written incrementally as whisper prints segments
# 3) If a prompt is found (either a user-passed prompt file OR a “prompt.md”
#    in the same folder), pipes that transcript into llm to generate a Markdown
#    description.
# 4) Cleans up the intermediate WAV, leaving only:
#      • original media file
#      • <basename>.md                   (the transcript)
#      • <basename>_description.md       (only if a prompt is used)
#
# ────────────────────────────────────────────────────────────────────────
# PREREQUISITES:
#   • ffmpeg           (e.g. brew install ffmpeg or download from https://ffmpeg.org)
#   • whisper-cli      (build from https://github.com/ggerganov/whisper.cpp)
#   • llm CLI          (https://github.com/llm-cli/llm; e.g. “pip install llm” or “brew install llm”)
#        – After installing llm, set an API key or default model. For example:
#            llm keys set openai
#            llm set-default-model gpt-4o-mini
#        – If you want to force a specific model for description, edit generate_description()
#          (see “-m <model-name>” example below).
#
# CONFIG (edit these to match your machine):
#   FFMPEG_PATH   — path to ffmpeg binary (e.g. /opt/homebrew/bin/ffmpeg)
#   WHISPER_PATH  — path to whisper-cli binary (e.g. /Users/nikola/dev/whisper.cpp/build/bin/whisper-cli)
#   MODELS_DIR    — directory containing whisper.cpp model files (e.g. “…/whisper.cpp/models”)
#   SCRIBE_STREAM_TRANSCRIPT — set to 0 to disable streaming output (default: 1)
#
# USAGE:
#   1) Single file, no explicit prompt given → looks for “prompt.md” next to that file:
#        ./scribe lecture.mp4
#      → Produces:
#        lecture.wav         (intermediate, then deleted)
#        lecture.md          (transcript)
#        lecture_description.md  (only if “lecture_directory/prompt.md” existed)
#
#   2) Single file + custom prompt:
#        ./scribe podcast.wav youtube-description.md
#      → Produces:
#        podcast_converted.wav   (intermediate, then deleted)
#        podcast.md              (transcript)
#        podcast_description.md  (based on youtube-description.md)
#
#   3) Directory of files:
#        ./scribe "/Users/you/AudioFolder"         (uses prompt.md if present in that folder)
#      OR:
#        ./scribe "/Users/you/AudioFolder" blog-prompt.md
#      → Loops through all top-level *.mp4, *.m4a, *.wav in that folder.
#        For each file:
#          • Converts → <basename>_converted.wav
#          • Transcribes → <basename>.md
#          • If “prompt.md” (or user-passed prompt) exists → <basename>_description.md
#          • Deletes <basename>_converted.wav (and temporary converted audio)
#
# EXAMPLES:
#   • Transcribe & (if prompt.md exists) describe one file:
#       ./scribe Andrija.WAV
#   • Transcribe & describe using a different prompt:
#       ./scribe Andrija.WAV youtube-description.md
#   • Batch mode (uses prompt.md if present in folder):
#       ./scribe "/Users/nikola/Documents/AudioNotes"
#
# NOTES:
#   – Whisper always needs a 16 kHz mono WAV → we avoid clobbering raw WAV by naming it
#     <basename>_converted.wav. After transcription, we clean that up.
#   – The transcript is written as Markdown (.md). You can rename or edit as needed.
#   – Description files always end in “_description.md.”
#   – If you want a specific LLM model for description, edit generate_description() to:
#         llm -m "<your-model-here>" -s "$prompt"
# ────────────────────────────────────────────────────────────────────────

# ──────────────────────────────  CONFIG  ──────────────────────────────
FFMPEG_PATH="/opt/homebrew/bin/ffmpeg"                               # ← adjust if needed
WHISPER_PATH="/Users/nikola/dev/whisper.cpp/build/bin/whisper-cli"   # ← adjust
MODELS_DIR="/Users/nikola/dev/whisper.cpp/models"                    # ← adjust
LANGUAGE="${WHISPER_LANGUAGE:-en}"                                    # ← default to English, override with WHISPER_LANGUAGE env var
SAMPLE_RATE="${SCRIBE_SAMPLE_RATE:-16000}"                            # ← sample rate for Whisper input
# mono strategy: left | right | average
SCRIBE_MONO_MODE="${SCRIBE_MONO_MODE:-average}"
# set to 0 to disable live transcript writes and use whisper CLI file output mode
SCRIBE_STREAM_TRANSCRIPT="${SCRIBE_STREAM_TRANSCRIPT:-1}"
# set to 1 to keep timestamps in transcript output (default: 0)
SCRIBE_WITH_TIMESTAMPS="${SCRIBE_WITH_TIMESTAMPS:-0}"

# ──────────────────────────────  FUNCTIONS  ──────────────────────────────

# convert():
#   • Input: any media file (mp4, m4a, wav, etc.)
#   • Output (echoed): path to a 16 kHz mono WAV
#       – If input extension ≠ .wav → “<basename>.wav”
#       – If input extension = .wav  → “<basename>_converted.wav” (to avoid overwriting)
#   • Caller is responsible for deleting the returned WAV.
convert() {
  local input="$1"
  local ext="${input##*.}"
  local base="${input%.*}"
  local output_wav
  local channel_count
  local mono_filter

  if [[ "${ext,,}" == "wav" ]]; then
    output_wav="${base}_converted.wav"
  else
    output_wav="${base}.wav"
  fi

  echo "Converting → ${output_wav} (${SAMPLE_RATE} Hz mono) ..." >&2

  # Prefer an explicit mono source channel choice over ffmpeg's default downmix behavior.
  # Default is "left" to avoid phase-cancellation artifacts from stereo merge.
  channel_count="$("$FFMPEG_PATH" -v error -select_streams a:0 \
    -show_entries stream=channels \
    -of csv=p=0 "$input" 2>/dev/null | tr -dc '0-9')"
  channel_count="${channel_count:-1}"

  if (( channel_count > 1 )); then
    case "$SCRIBE_MONO_MODE" in
      right)
        mono_filter="pan=mono|c0=c1"
        ;;
      average|avg|blend)
        # Safer than default -ac 1 for many stereo recordings; avoids hard clipping.
        mono_filter="pan=mono|c0=0.5*c0+0.5*c1"
        ;;
      *)
        mono_filter="pan=mono|c0=c0"
        ;;
    esac
  else
    # mono input, no channel merge needed.
    mono_filter="pan=mono|c0=c0"
  fi

  "$FFMPEG_PATH" -y -i "$input" \
    -af "${mono_filter},aresample=${SAMPLE_RATE}:resampler=soxr:precision=28" \
    -c:a pcm_s16le "$output_wav"
  echo "$output_wav"
}

# transcribe():
#   • Inputs:
#       1) wav_path  → the 16 kHz mono WAV from convert()
#       2) orig_file → the original media path (so we name transcript correctly)
#   • Output: writes “<orig-basename>.md” directly (streamed as whisper prints)
#   • Returns: the final transcript path (“<orig-basename>.md”)
transcribe() {
  local wav_path="$1"
  local orig_file="$2"
  local orig_base="${orig_file%.*}"
  local md_path="${orig_base}.md"
  local -a whisper_args=(
    -m "${MODELS_DIR}/ggml-medium.bin"
    -l "$LANGUAGE"
    -f "$wav_path"
    -np
  )

  echo "Transcribing → ${md_path} ..." >&2

  if [[ "$SCRIBE_WITH_TIMESTAMPS" != "1" ]]; then
    whisper_args+=(-nt)
  fi

  : > "$md_path"
  if [[ "${SCRIBE_STREAM_TRANSCRIPT}" == "0" && "$SCRIBE_WITH_TIMESTAMPS" != "1" ]]; then
    "$WHISPER_PATH" \
      "${whisper_args[@]}" \
      -otxt \
      -of "$orig_base"

    if [[ ! -f "${orig_base}.txt" ]]; then
      echo "Error: Whisper did not write ${orig_base}.txt" >&2
      exit 1
    fi
    mv "${orig_base}.txt" "$md_path"
    echo "Wrote transcript → ${md_path}" >&2
    echo "$md_path"
    return
  fi

  "$WHISPER_PATH" \
    "${whisper_args[@]}" \
    | tee -a "$md_path"

  echo "Wrote transcript (streamed) → ${md_path}" >&2
  echo "$md_path"
}

# generate_description():
#   • Inputs:
#       1) prompt_file → a text file containing the LLM prompt
#       2) orig_file   → the original media path (so the transcript is “<orig-basename>.md”)
#   • Output: "<orig-basename>_description.md"
#   • Uses: llm CLI, piping transcript into -s "$prompt"
generate_description() {
  local prompt_file="$1"
  local orig_file="$2"
  local transcript_md="${orig_file%.*}.md"
  local output_md="${orig_file%.*}_description.md"

  if [[ ! -f "$prompt_file" ]]; then
    echo "Warning: Prompt file “$prompt_file” not found. Skipping description." >&2
    return 0
  fi

  if [[ ! -f "$transcript_md" ]]; then
    echo "Error: Transcript “${transcript_md}” not found; cannot generate description." >&2
    return 1
  fi

  echo "Generating description → ${output_md} ..." >&2
  local prompt
  prompt=$(<"$prompt_file")

  cat "$transcript_md" | llm -s "$prompt" > "$output_md"
  echo "Wrote description → ${output_md}" >&2
}

# find_default_prompt():
#   • Given an orig_file (absolute path), looks in the same folder for “prompt.md.”
#   • If found, echoes that path; otherwise, echoes empty string.
find_default_prompt() {
  local dir
  dir="$(dirname "$1")"
  if [[ -f "${dir}/prompt.md" ]]; then
    echo "${dir}/prompt.md"
  else
    echo ""
  fi
}

# process_file():
#   • Inputs:
#       1) file_path      → absolute path to media (mp4/m4a/wav)
#       2) optional_prompt→ explicit prompt file (can be empty string)
#   • Workflow:
#       a) convert()      → get 16kHz WAV
#       b) transcribe()   → write "<basename>.md"
#       c) decide prompt  → explicit or default “prompt.md”
#       d) generate_description() if a prompt was chosen
#       e) delete the intermediate WAV
#          • always delete the returned converted file
#          • also delete any <basename>_converted.wav helper file for that input
process_file() {
  local abs_file
  abs_file="$(realpath "$1")"
  local explicit_prompt="$2"
  local base_no_ext="${abs_file%.*}"

  # Step a: convert
  local wav_path
  wav_path="$(convert "$abs_file")" || exit 1

  # Step b: transcribe
  transcribe "$wav_path" "$abs_file"

  # Step c: choose prompt
  local prompt_to_use=""
  if [[ -n "$explicit_prompt" && -f "$explicit_prompt" ]]; then
    prompt_to_use="$explicit_prompt"
  fi

  if [[ -z "$prompt_to_use" ]]; then
    prompt_to_use="$(find_default_prompt "$abs_file")"
  fi

  # Step d: generate description (if we have a prompt)
  if [[ -n "$prompt_to_use" ]]; then
    generate_description "$prompt_to_use" "$abs_file"
  else
    echo "No prompt found for “$abs_file” → skipping description." >&2
  fi

  # Step e: clean up WAV
  local converted_wav="${base_no_ext}_converted.wav"

  if [[ -f "$wav_path" ]]; then
    rm -f "$wav_path"
    echo "Deleted intermediate WAV: $wav_path" >&2
  fi

  if [[ "$converted_wav" != "$wav_path" && -f "$converted_wav" ]]; then
    rm -f "$converted_wav"
    echo "Deleted helper converted WAV: $converted_wav" >&2
  fi
}

# ───────────────────────────────  MAIN  ───────────────────────────────

if (( $# < 1 )); then
  cat <<EOF
Usage: $(basename "$0") <file-or-directory> [prompt_file]

Options:
  --mono-mode <left|right|average>  Which input channel to use for mono conversion.
                                     (default: ${SCRIBE_MONO_MODE})
                                     - left: use left channel
                                     - right: use right channel
                                     - average: 0.5*left + 0.5*right
  --timestamps                        Include timestamps in transcript output
  --no-timestamps                     Omit timestamps in transcript output
  --sample-rate <hz>                  Sample rate for conversion (default: ${SAMPLE_RATE})

  <file-or-directory> :
      • If you supply a single file (mp4/m4a/wav), scribe processes that
        one file.
      • If you supply a directory, scribe finds all top-level *.mp4, *.m4a, *.mp3, *.wav
        (case-insensitive) in it and processes each in turn.

  [prompt_file] :
      • (Optional) Path to a custom prompt (e.g. “youtube-description.md”).
        If given, that file is always used for description.
      • Otherwise, scribe looks for “prompt.md” in the same folder as each input.

Examples:
  1) Transcribe “lecture.mp4” and, if “prompt.md” exists next to it, also make a description:
       ./scribe lecture.mp4

  2) Transcribe “podcast.wav” and use “youtube-description.md” for the description:
       ./scribe podcast.wav youtube-description.md

  3) Batch mode on a directory (uses prompt.md if present):
       ./scribe "/Users/you/AudioFolder"

  4) Batch mode on a directory but force “blog-prompt.md” for every file:
       ./scribe "/Users/you/AudioFolder" blog-prompt.md

  5) Force mono conversion from right channel:
       ./scribe --mono-mode right recording.wav

  6) Keep transcript with timestamps (default: off):
       ./scribe --timestamps recording.wav
  7) Force mono conversion from average with 16kHz:
       ./scribe --mono-mode average --sample-rate 16000 recording.wav
EOF
  exit 1
fi

mono_mode="${SCRIBE_MONO_MODE}"
sample_rate="${SAMPLE_RATE}"
input=""
prompt_arg=""

while (( $# > 0 )); do
  case "$1" in
  --mono-mode)
      if (( $# < 2 )); then
        echo "Error: --mono-mode requires an argument." >&2
        exit 1
      fi
      mono_mode="$2"
      shift 2
      ;;
    --mono-mode=*)
      mono_mode="${1#*=}"
      shift
      ;;
    --sample-rate)
      if (( $# < 2 )); then
        echo "Error: --sample-rate requires an argument." >&2
        exit 1
      fi
      sample_rate="$2"
      shift 2
      ;;
    --sample-rate=*)
      sample_rate="${1#*=}"
      shift
      ;;
    --timestamps)
      SCRIBE_WITH_TIMESTAMPS="1"
      shift
      ;;
    --no-timestamps)
      SCRIBE_WITH_TIMESTAMPS="0"
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [options] <file-or-directory> [prompt_file]

Options:
  --mono-mode <left|right|average>  Which input channel to use for mono conversion.
                                     (default: ${SCRIBE_MONO_MODE})
                                     - left: use left channel
                                     - right: use right channel
                                     - average: 0.5*left + 0.5*right
  --timestamps                       Include timestamps in transcript output
  --no-timestamps                    Omit timestamps in transcript output
  --sample-rate <hz>                 Sample rate for conversion (default: ${SAMPLE_RATE})
  -h, --help                         Show this help text.
EOF
      exit 0
      ;;
    *)
      if [[ -z "$input" ]]; then
        input="$1"
      elif [[ -z "$prompt_arg" ]]; then
        prompt_arg="$1"
      else
        echo "Error: too many positional arguments: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

case "$mono_mode" in
  left|right|average|avg|blend)
    ;;
  *)
    echo "Error: --mono-mode must be one of: left, right, average" >&2
    exit 1
    ;;
esac

SAMPLE_RATE="$sample_rate"
case "$mono_mode" in
  average|avg|blend)
    SCRIBE_MONO_MODE="average"
    ;;
  right)
    SCRIBE_MONO_MODE="right"
    ;;
  *)
    SCRIBE_MONO_MODE="left"
    ;;
esac

if [[ -z "${input}" ]]; then
  echo "Error: missing required <file-or-directory> argument." >&2
  exit 1
fi

if [[ ! "$SAMPLE_RATE" =~ ^[0-9]+$ ]] || (( SAMPLE_RATE <= 0 )); then
  echo "Error: --sample-rate must be a positive integer." >&2
  exit 1
fi

if [[ -d "$input" ]]; then
  # Directory mode: loop through top-level mp4/m4a/wav only
  while IFS= read -r -d '' media; do
    echo "===============================" >&2
    echo "Processing: $media" >&2
    process_file "$media" "$prompt_arg"
    echo "===============================" >&2
  done < <(find "$input" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.m4a' -o -iname '*.wav' -o -iname '*.mp3' \) -print0)

elif [[ -f "$input" ]]; then
  echo "===============================" >&2
  echo "Processing single file: $input" >&2
  process_file "$input" "$prompt_arg"
  echo "===============================" >&2

else
  echo "Error: '$input' is not a valid file or directory." >&2
  exit 1
fi

echo "All done!" >&2
