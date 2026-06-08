package main

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"compress/gzip"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"scriby/internal/clipboard"
	history "scriby/internal/history"
	"scriby/internal/state"
)

const (
	appVersion                = "2.1.1"
	schemaVersion             = "1.0"
	defaultEngineName         = "whisper"
	defaultModelName          = "medium"
	defaultCohereModelID      = "CohereLabs/cohere-transcribe-03-2026"
	defaultModelURLMedium     = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
	defaultRuntimeManifestURL = "https://github.com/nibzard/scriby/releases/download/v2.1.1/runtime-manifest.json"

	exitOK         = 0
	exitInput      = 2
	exitDependency = 3
	exitRuntime    = 4
	exitPartial    = 5
	exitInternal   = 10
)

var knownModelURLs = map[string]string{
	"tiny":     "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
	"base":     "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
	"small":    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
	"medium":   defaultModelURLMedium,
	"large-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
}

var cohereSupportedLanguages = map[string]string{
	"ar":    "ar",
	"de":    "de",
	"el":    "el",
	"en":    "en",
	"es":    "es",
	"fr":    "fr",
	"gr":    "el",
	"it":    "it",
	"ja":    "ja",
	"ko":    "ko",
	"nl":    "nl",
	"pl":    "pl",
	"pt":    "pt",
	"vi":    "vi",
	"zh":    "zh",
	"zh-cn": "zh",
}

type AppError struct {
	Class     string `json:"class"`
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	Hint      string `json:"hint,omitempty"`
}

type Warning struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type Envelope struct {
	SchemaVersion string         `json:"schema_version"`
	Command       string         `json:"command"`
	Status        string         `json:"status"`
	RunID         string         `json:"run_id"`
	SessionID     string         `json:"session_id,omitempty"`
	Data          any            `json:"data,omitempty"`
	Errors        []AppError     `json:"errors,omitempty"`
	Warnings      []Warning      `json:"warnings,omitempty"`
	Metrics       map[string]any `json:"metrics,omitempty"`
}

type GlobalOptions struct {
	Output         string
	Agent          bool
	Strict         bool
	NonInteractive bool
	Yes            bool
	TimeoutMS      int
	MaxRetries     int
	IdempotencyKey string
	SessionPolicy  string
	SessionID      string
	StateDir       string
}

type RunConfig struct {
	Input              string
	Prompt             string
	Clipboard          string
	MonoMode           string
	SampleRate         int
	Timestamps         bool
	Engine             string
	Language           string
	StreamTranscript   bool
	ModelName          string
	ModelURL           string
	HFModelID          string
	PythonPath         string
	WhisperPath        string
	WhisperURL         string
	RuntimeManifestURL string
	FFmpegPath         string
	LLMPath            string
	KeepTemp           bool
}

type RuntimeManifest struct {
	Version string         `json:"version"`
	Assets  []RuntimeAsset `json:"assets"`
}

type RuntimeAsset struct {
	Name   string `json:"name"`
	OS     string `json:"os"`
	Arch   string `json:"arch"`
	Format string `json:"format,omitempty"`
	Binary string `json:"binary,omitempty"`
	URL    string `json:"url"`
	SHA256 string `json:"sha256,omitempty"`
}

type ProgressReporter struct {
	OutputMode string
	Command    string
	RunID      string
}

func newProgressReporter(outputMode string, command string, runID string) *ProgressReporter {
	return &ProgressReporter{
		OutputMode: outputMode,
		Command:    command,
		RunID:      runID,
	}
}

func (p *ProgressReporter) Step(event string, message string, data map[string]any) {
	if p == nil {
		return
	}

	if p.OutputMode == "jsonl" {
		payload := map[string]any{}
		for k, v := range data {
			payload[k] = v
		}
		if strings.TrimSpace(message) != "" {
			payload["message"] = message
		}
		emitJSONLEvent(p.Command, p.RunID, event, payload)
		return
	}

	if strings.TrimSpace(message) != "" {
		fmt.Fprintf(os.Stderr, "[scriby] %s\n", message)
	}
}

type RunData struct {
	Input            string       `json:"input"`
	Prompt           string       `json:"prompt,omitempty"`
	Engine           string       `json:"engine"`
	Language         string       `json:"language,omitempty"`
	Clipboard        string       `json:"clipboard,omitempty"`
	MonoMode         string       `json:"mono_mode,omitempty"`
	SampleRate       int          `json:"sample_rate,omitempty"`
	Timestamps       bool         `json:"timestamps,omitempty"`
	StreamTranscript bool         `json:"stream_transcript,omitempty"`
	FFmpegPath       string       `json:"ffmpeg_path"`
	TranscriberPath  string       `json:"transcriber_path,omitempty"`
	ModelRef         string       `json:"model_ref,omitempty"`
	ModelName        string       `json:"model_name,omitempty"`
	HFModelID        string       `json:"hf_model_id,omitempty"`
	WhisperPath      string       `json:"whisper_path,omitempty"`
	ModelPath        string       `json:"model_path,omitempty"`
	Files            []FileResult `json:"files"`
}

type FileResult struct {
	File        string    `json:"file"`
	Transcript  string    `json:"transcript,omitempty"`
	Description string    `json:"description,omitempty"`
	Status      string    `json:"status"`
	Warnings    []Warning `json:"warnings,omitempty"`
	Error       *AppError `json:"error,omitempty"`
}

type DoctorCheck struct {
	Name    string `json:"name"`
	Status  string `json:"status"`
	Details string `json:"details"`
}

type StoredRunEnvelope struct {
	SchemaVersion string         `json:"schema_version"`
	Command       string         `json:"command"`
	Status        string         `json:"status"`
	RunID         string         `json:"run_id"`
	SessionID     string         `json:"session_id,omitempty"`
	Data          RunData        `json:"data"`
	Errors        []AppError     `json:"errors,omitempty"`
	Warnings      []Warning      `json:"warnings,omitempty"`
	Metrics       map[string]any `json:"metrics,omitempty"`
}

func main() {
	leadingGlobalArgs, remainingArgs, rootHelp, rootParseErr := splitRootArgs(os.Args[1:])
	if rootParseErr != nil {
		env := newEnvelope("root")
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", rootParseErr.Error(), false, "Run 'scriby --help' for usage")}
		finishEnvelope(&env, time.Now(), 0, 0, 0)
		_ = printCommandResult(env, os.Args[1:])
		os.Exit(exitInput)
	}

	if len(remainingArgs) < 1 {
		if rootHelp {
			env := rootHelpEnvelope()
			_ = printCommandResult(env, leadingGlobalArgs)
			os.Exit(exitOK)
		}
		env := rootMissingCommandEnvelope()
		_ = printCommandResult(env, leadingGlobalArgs)
		os.Exit(exitInput)
	}

	command := remainingArgs[0]
	args := append(append([]string{}, leadingGlobalArgs...), remainingArgs[1:]...)

	switch command {
	case "version", "-V", "--version":
		env := newEnvelope("version")
		env.Status = "succeeded"
		env.Data = appVersion
		finishEnvelope(&env, time.Now(), 0, 0, 0)
		_ = printCommandResult(env, args)
		os.Exit(exitOK)
	case "help", "-h", "--help":
		env := rootHelpEnvelope()
		_ = printCommandResult(env, args)
		os.Exit(exitOK)
	case "run":
		env, code := handleRun(args)
		_ = printCommandResult(env, args)
		os.Exit(code)
	case "validate":
		env, code := handleValidate(args)
		_ = printCommandResult(env, args)
		os.Exit(code)
	case "doctor":
		env, code := handleDoctor(args)
		_ = printCommandResult(env, args)
		os.Exit(code)
	case "replay":
		env, code := handleReplay(args)
		_ = printCommandResult(env, args)
		os.Exit(code)
	case "retry":
		env, code := handleRetry(args)
		_ = printCommandResult(env, args)
		os.Exit(code)
	case "models":
		env, code := handleModels(args)
		_ = printCommandResult(env, args)
		os.Exit(code)
	case "history":
		env, code := handleHistory(args)
		_ = printCommandResult(env, args)
		os.Exit(code)
	default:
		env := newEnvelope("root")
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "UNKNOWN_COMMAND", fmt.Sprintf("unknown command: %s", command), false, "Use 'scriby help' to list commands")}
		env.Metrics["duration_ms"] = int64(0)
		_ = printCommandResult(env, args)
		os.Exit(exitInput)
	}
}

func rootHelp() string {
	return `Usage: scriby <command> [options]

Commands:
  run        Convert + transcribe (+ optional description generation)
  validate   Validate inputs and runtime readiness without running transcription
  doctor     Diagnose local environment and suggest deterministic remediations
  replay     Replay a saved run envelope by run_id
  retry      Retry a saved transcription run
  models     Manage local whisper model files (pull, list, prune)
  history    Explore local run/transcription history in SQLite

Global Contract:
  - Output modes: --output text|json|jsonl (default: text)
  - Agent shortcut: --agent implies --output json, --non-interactive, --yes, and disables run clipboard prompts
  - Stable envelope: schema_version, command, status, run_id, data, errors, warnings, metrics
  - Deterministic exit codes:
      0 success, 2 input error, 3 dependency/setup error, 4 runtime failure, 5 partial success, 10 internal error

Examples:
  scriby run ./lecture.mp4
  scriby run --engine cohere --language en ./lecture.mp4
  scriby run --agent ./lecture.mp4
  scriby run --prompt ./youtube-description.md ./podcast.wav
  scriby validate --strict ./recordings
  scriby doctor --output text
  scriby models pull --name medium
  scriby replay 20260224-abc123
  scriby retry 20260224-abc123 --failed-only
  scriby history search "quarterly planning"
`
}

func rootHelpEnvelope() Envelope {
	env := newEnvelope("root")
	env.Data = rootHelp()
	finishEnvelope(&env, time.Now(), 0, 0, 0)
	return env
}

func rootMissingCommandEnvelope() Envelope {
	env := newEnvelope("root")
	env.Status = "failed"
	env.Errors = []AppError{newError("input", "MISSING_COMMAND", "missing required <command>", false, "Use 'scriby --help' to list commands")}
	finishEnvelope(&env, time.Now(), 0, 0, 0)
	return env
}

func newEnvelope(command string) Envelope {
	return Envelope{
		SchemaVersion: schemaVersion,
		Command:       command,
		Status:        "succeeded",
		RunID:         newRunID(),
		Metrics:       map[string]any{},
	}
}

func defaultGlobalOptions() GlobalOptions {
	return GlobalOptions{
		Output:         envOr("SCRIBY_OUTPUT", "text"),
		Agent:          envBool("SCRIBY_AGENT", false),
		Strict:         envBool("SCRIBY_STRICT", false),
		NonInteractive: envBool("SCRIBY_NON_INTERACTIVE", false),
		Yes:            envBool("SCRIBY_YES", false),
		TimeoutMS:      envInt("SCRIBY_TIMEOUT_MS", 0),
		MaxRetries:     envInt("SCRIBY_MAX_RETRIES", 2),
		IdempotencyKey: envOr("SCRIBY_IDEMPOTENCY_KEY", ""),
		SessionPolicy:  envOr("SCRIBY_SESSION_POLICY", "ephemeral"),
		SessionID:      envOr("SCRIBY_SESSION_ID", ""),
		StateDir:       envOr("SCRIBY_STATE_DIR", ""),
	}
}

func defaultRunConfig() RunConfig {
	return RunConfig{
		Clipboard:          envOr("SCRIBY_CLIPBOARD", "ask"),
		MonoMode:           envOr("SCRIBE_MONO_MODE", "average"),
		SampleRate:         envInt("SCRIBE_SAMPLE_RATE", 16000),
		Timestamps:         envBool("SCRIBE_WITH_TIMESTAMPS", false),
		Engine:             envOr("SCRIBY_ENGINE", defaultEngineName),
		Language:           envOr("SCRIBY_LANGUAGE", envOr("WHISPER_LANGUAGE", "en")),
		StreamTranscript:   envBool("SCRIBE_STREAM_TRANSCRIPT", true),
		ModelName:          envOr("SCRIBY_MODEL", defaultModelName),
		ModelURL:           envOr("SCRIBY_MODEL_URL", ""),
		HFModelID:          envOr("SCRIBY_HF_MODEL_ID", defaultCohereModelID),
		PythonPath:         envOr("SCRIBY_PYTHON_PATH", ""),
		WhisperPath:        envOr("SCRIBY_WHISPER_PATH", ""),
		WhisperURL:         envOr("SCRIBY_WHISPER_URL", ""),
		RuntimeManifestURL: envOr("SCRIBY_RUNTIME_MANIFEST_URL", defaultRuntimeManifestURL),
		FFmpegPath:         envOr("SCRIBY_FFMPEG_PATH", ""),
		LLMPath:            envOr("SCRIBY_LLM_PATH", "llm"),
		KeepTemp:           envBool("SCRIBY_KEEP_TEMP", false),
	}
}

func addGlobalFlags(fs *flag.FlagSet, g *GlobalOptions) {
	fs.StringVar(&g.Output, "output", g.Output, "Output mode: json|jsonl|text")
	fs.BoolVar(&g.Agent, "agent", g.Agent, "Agent mode: JSON output, non-interactive prompts, clipboard disabled for run")
	fs.BoolVar(&g.Strict, "strict", g.Strict, "Disable silent fallbacks and enforce strict contract behavior")
	fs.BoolVar(&g.NonInteractive, "non-interactive", g.NonInteractive, "Disable interactive prompts")
	fs.BoolVar(&g.Yes, "yes", g.Yes, "Assume yes for destructive operations")
	fs.IntVar(&g.TimeoutMS, "timeout-ms", g.TimeoutMS, "Command timeout in milliseconds (0 = no timeout)")
	fs.IntVar(&g.MaxRetries, "max-retries", g.MaxRetries, "Maximum retries for retryable operations")
	fs.StringVar(&g.IdempotencyKey, "idempotency-key", g.IdempotencyKey, "Stable idempotency key for command replay")
	fs.StringVar(&g.SessionPolicy, "session-policy", g.SessionPolicy, "Session policy: ephemeral|sticky|resume")
	fs.StringVar(&g.SessionID, "session-id", g.SessionID, "Session identifier for sticky/resume sessions")
	fs.StringVar(&g.StateDir, "state-dir", g.StateDir, "State directory (models, runtime, runs)")
}

func applyAgentMode(global *GlobalOptions, cfg *RunConfig) {
	if global == nil || !global.Agent {
		return
	}
	global.Output = "json"
	global.NonInteractive = true
	global.Yes = true
	if global.MaxRetries < 1 {
		global.MaxRetries = 1
	}
	if cfg != nil {
		cfg.Clipboard = "never"
		cfg.StreamTranscript = false
	}
}

func addRunFlags(fs *flag.FlagSet, cfg *RunConfig) {
	fs.StringVar(&cfg.Prompt, "prompt", cfg.Prompt, "Prompt file for description generation")
	fs.StringVar(&cfg.Clipboard, "clipboard", cfg.Clipboard, "Clipboard behavior: never|ask|always")
	fs.StringVar(&cfg.MonoMode, "mono-mode", cfg.MonoMode, "Mono channel strategy: left|right|average")
	fs.IntVar(&cfg.SampleRate, "sample-rate", cfg.SampleRate, "Sample rate for converted WAV")
	fs.BoolVar(&cfg.Timestamps, "timestamps", cfg.Timestamps, "Include timestamps in transcript output")
	fs.StringVar(&cfg.Engine, "engine", cfg.Engine, "ASR engine: whisper|cohere")
	fs.StringVar(&cfg.Language, "language", cfg.Language, "Language code for the active ASR engine")
	fs.BoolVar(&cfg.StreamTranscript, "stream-transcript", cfg.StreamTranscript, "Stream transcription from whisper stdout (whisper engine only)")
	fs.StringVar(&cfg.ModelName, "model", cfg.ModelName, "Whisper model name (tiny|base|small|medium|large-v3)")
	fs.StringVar(&cfg.ModelURL, "model-url", cfg.ModelURL, "Override Whisper model download URL")
	fs.StringVar(&cfg.HFModelID, "hf-model-id", cfg.HFModelID, "Hugging Face model ID for the cohere engine")
	fs.StringVar(&cfg.PythonPath, "python-path", cfg.PythonPath, "Python executable (unused by cohere engine, which uses uv)")
	fs.StringVar(&cfg.WhisperPath, "whisper-path", cfg.WhisperPath, "Path to whisper-cli binary")
	fs.StringVar(&cfg.WhisperURL, "whisper-url", cfg.WhisperURL, "URL for whisper runtime binary/archive")
	fs.StringVar(&cfg.RuntimeManifestURL, "runtime-manifest-url", cfg.RuntimeManifestURL, "Runtime manifest URL for deterministic whisper bootstrap")
	fs.StringVar(&cfg.FFmpegPath, "ffmpeg-path", cfg.FFmpegPath, "Path to ffmpeg binary")
	fs.StringVar(&cfg.LLMPath, "llm-path", cfg.LLMPath, "Path to llm CLI binary")
	fs.BoolVar(&cfg.KeepTemp, "keep-temp", cfg.KeepTemp, "Keep intermediate WAV files")
}

func addHelpFlags(fs *flag.FlagSet, help *bool) {
	fs.BoolVar(help, "help", false, "Show help")
	fs.BoolVar(help, "h", false, "Show help")
}

func finishHelp(env *Envelope, started time.Time, helpText string) (Envelope, int) {
	env.Data = helpText
	finishEnvelope(env, started, 0, 0, 0)
	return *env, exitOK
}

func handleRun(args []string) (Envelope, int) {
	started := time.Now()
	args = hoistGlobalFlags(args)
	env := newEnvelope("run")
	global := defaultGlobalOptions()
	cfg := defaultRunConfig()
	var help bool

	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addRunFlags(fs, &cfg)
	addHelpFlags(fs, &help)

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, runHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby run --help' for usage")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, runHelp())
	}
	applyAgentMode(&global, &cfg)

	if !isValidOutputMode(global.Output) {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_OUTPUT_MODE", "--output must be one of json|jsonl|text", false, "Set --output json, --output jsonl, or --output text")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	pos := fs.Args()
	if len(pos) == 0 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_INPUT", "missing required <file-or-directory>", false, "Usage: scriby run [flags] <file-or-directory> [prompt_file]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if len(pos) > 2 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "TOO_MANY_ARGUMENTS", "too many positional arguments", false, "Usage: scriby run [flags] <file-or-directory> [prompt_file]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	cfg.Input = pos[0]
	if cfg.Prompt == "" && len(pos) == 2 {
		cfg.Prompt = pos[1]
	}

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	global.StateDir = stateDir

	sid, serr := ensureSession(stateDir, global.SessionPolicy, global.SessionID)
	if serr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*serr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	env.SessionID = sid

	if global.IdempotencyKey != "" {
		if cached, ok := loadIdempotencyRecord(stateDir, "run", global.IdempotencyKey); ok {
			cached.Warnings = append(cached.Warnings, Warning{Code: "IDEMPOTENT_REPLAY", Message: "Returning cached result for idempotency key"})
			finishEnvelope(&cached, started, int64(asInt(cached.Metrics["files_total"])), int64(asInt(cached.Metrics["files_succeeded"])), int64(asInt(cached.Metrics["files_failed"])))
			return cached, exitFromStatus(cached.Status)
		}
	}

	validationWarnings, validationErr := validateRunInputs(cfg)
	env.Warnings = append(env.Warnings, validationWarnings...)
	if validationErr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*validationErr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	ctx, cancel := commandContext(global.TimeoutMS)
	defer cancel()
	progress := newProgressReporter(global.Output, "run", env.RunID)
	progress.Step("run.start", fmt.Sprintf("Starting run for %s", cfg.Input), map[string]any{"input": cfg.Input})

	ffmpegPath, ferr := ensureFFmpegPath(cfg.FFmpegPath)
	if ferr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*ferr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}

	engine := normalizeEngine(cfg.Engine)
	var transcriberPath string
	var modelRef string
	var whisperPath string
	var modelPath string

	switch engine {
	case "whisper":
		progress.Step("runtime.ensure", "Preparing whisper runtime", nil)
		wPath, werr := ensureWhisperPath(ctx, stateDir, cfg.WhisperPath, cfg.WhisperURL, cfg.RuntimeManifestURL, global.MaxRetries, progress)
		if werr != nil {
			env.Status = "failed"
			env.Errors = []AppError{*werr}
			finishEnvelope(&env, started, 0, 0, 0)
			return env, exitDependency
		}

		progress.Step("model.ensure", fmt.Sprintf("Preparing model %s", modelFilename(cfg.ModelName)), map[string]any{"model": modelFilename(cfg.ModelName)})
		mPath, merr := ensureModel(ctx, stateDir, cfg.ModelName, cfg.ModelURL, global.MaxRetries, progress)
		if merr != nil {
			env.Status = "failed"
			env.Errors = []AppError{*merr}
			finishEnvelope(&env, started, 0, 0, 0)
			return env, exitDependency
		}
		whisperPath = wPath
		modelPath = mPath
		transcriberPath = wPath
		modelRef = modelFilename(cfg.ModelName)
	case "cohere":
		progress.Step("runtime.ensure", "Checking uv and mlx-audio runtime", nil)
		uvPath, cerr := ensureMLXAudioRuntime()
		if cerr != nil {
			env.Status = "failed"
			env.Errors = []AppError{*cerr}
			finishEnvelope(&env, started, 0, 0, 0)
			return env, exitDependency
		}
		transcriberPath = uvPath
		modelRef = strings.TrimSpace(cfg.HFModelID)
		progress.Step("model.ensure", fmt.Sprintf("Using Hugging Face model %s", modelRef), map[string]any{"model": modelRef})
	}

	files, lerr := listInputFiles(cfg.Input)
	if lerr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*lerr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	if cfg.Prompt != "" {
		if !fileExists(cfg.Prompt) {
			env.Warnings = append(env.Warnings, Warning{Code: "PROMPT_NOT_FOUND", Message: fmt.Sprintf("Prompt file not found: %s. Falling back to prompt.md next to each file.", cfg.Prompt)})
			cfg.Prompt = ""
		} else {
			ap, aerr := filepath.Abs(cfg.Prompt)
			if aerr == nil {
				cfg.Prompt = ap
			}
		}
	}

	progress.Step("run.plan", fmt.Sprintf("Processing %d file(s)", len(files)), map[string]any{"files_total": len(files)})

	llmPath := cfg.LLMPath
	haveLLM := false
	if llmPath != "" {
		if p, err := exec.LookPath(llmPath); err == nil {
			llmPath = p
			haveLLM = true
		}
	}

	runData := RunData{
		Input:            cfg.Input,
		Prompt:           cfg.Prompt,
		Engine:           engine,
		Language:         cfg.Language,
		Clipboard:        cfg.Clipboard,
		MonoMode:         cfg.MonoMode,
		SampleRate:       cfg.SampleRate,
		Timestamps:       cfg.Timestamps,
		StreamTranscript: cfg.StreamTranscript,
		FFmpegPath:       ffmpegPath,
		TranscriberPath:  transcriberPath,
		ModelRef:         modelRef,
		ModelName:        cfg.ModelName,
		HFModelID:        cfg.HFModelID,
		WhisperPath:      whisperPath,
		ModelPath:        modelPath,
		Files:            make([]FileResult, 0, len(files)),
	}

	var successes int64
	var failures int64

	for i, media := range files {
		progress.Step(
			"file.start",
			fmt.Sprintf("Processing file %d/%d: %s", i+1, len(files), media),
			map[string]any{"file": media, "index": i + 1, "total": len(files)},
		)
		fr, warns, perr := processMediaFile(ctx, cfg, media, ffmpegPath, transcriberPath, modelPath, llmPath, haveLLM, global.Output, env.RunID, progress)
		runData.Files = append(runData.Files, fr)
		env.Warnings = append(env.Warnings, warns...)
		if perr != nil {
			failures++
			progress.Step(
				"file.failed",
				fmt.Sprintf("Failed file %d/%d: %s", i+1, len(files), media),
				map[string]any{"file": media, "index": i + 1, "total": len(files), "error_code": perr.Code},
			)
		} else {
			successes++
			progress.Step(
				"file.done",
				fmt.Sprintf("Completed file %d/%d: %s", i+1, len(files), media),
				map[string]any{"file": media, "index": i + 1, "total": len(files)},
			)
		}
	}

	env.Data = runData
	env.Warnings = append(env.Warnings, maybeHandleClipboard(global, cfg, runData.Files, progress)...)
	if failures == 0 {
		env.Status = "succeeded"
	} else if successes > 0 {
		env.Status = "partial"
	} else {
		env.Status = "failed"
	}

	finishEnvelope(&env, started, int64(len(files)), successes, failures)
	if global.IdempotencyKey != "" {
		_ = saveIdempotencyRecord(stateDir, "run", global.IdempotencyKey, env)
	}
	_ = saveRunRecord(stateDir, env)
	_ = saveHistoryRecord(stateDir, env)
	progress.Step(
		"run.done",
		fmt.Sprintf("Run finished with status %s (%d succeeded, %d failed)", env.Status, successes, failures),
		map[string]any{"status": env.Status, "files_succeeded": successes, "files_failed": failures},
	)
	return env, exitFromStatus(env.Status)
}

func handleValidate(args []string) (Envelope, int) {
	started := time.Now()
	args = hoistGlobalFlags(args)
	env := newEnvelope("validate")
	global := defaultGlobalOptions()
	cfg := defaultRunConfig()
	var help bool

	fs := flag.NewFlagSet("validate", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addRunFlags(fs, &cfg)
	addHelpFlags(fs, &help)

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, validateHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby validate --help' for usage")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, validateHelp())
	}
	applyAgentMode(&global, &cfg)
	if !isValidOutputMode(global.Output) {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_OUTPUT_MODE", "--output must be one of json|jsonl|text", false, "Set --output json, --output jsonl, or --output text")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	pos := fs.Args()
	if len(pos) == 0 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_INPUT", "missing required <file-or-directory>", false, "Usage: scriby validate [flags] <file-or-directory> [prompt_file]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if len(pos) > 2 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "TOO_MANY_ARGUMENTS", "too many positional arguments", false, "Usage: scriby validate [flags] <file-or-directory> [prompt_file]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	cfg.Input = pos[0]
	if cfg.Prompt == "" && len(pos) == 2 {
		cfg.Prompt = pos[1]
	}

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}

	warnings, verr := validateRunInputs(cfg)
	env.Warnings = append(env.Warnings, warnings...)
	if verr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*verr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	checks := map[string]any{}
	checks["state_dir"] = stateDir
	checks["engine"] = normalizeEngine(cfg.Engine)
	checks["input_exists"] = fileExists(cfg.Input) || dirExists(cfg.Input)
	checks["prompt_exists"] = cfg.Prompt == "" || fileExists(cfg.Prompt)
	checks["sample_rate"] = cfg.SampleRate
	checks["mono_mode"] = cfg.MonoMode

	if cfg.FFmpegPath != "" {
		checks["ffmpeg"] = fileExists(cfg.FFmpegPath)
	} else {
		_, err := exec.LookPath("ffmpeg")
		checks["ffmpeg"] = err == nil
	}

	switch normalizeEngine(cfg.Engine) {
	case "whisper":
		if cfg.WhisperPath != "" {
			checks["whisper"] = fileExists(cfg.WhisperPath)
		} else if p, err := exec.LookPath(binaryName("whisper-cli")); err == nil {
			checks["whisper"] = p != ""
		} else {
			runtimePath := filepath.Join(stateDir, "runtime", binaryName("whisper-cli"))
			checks["whisper"] = fileExists(runtimePath)
		}

		modelPath := filepath.Join(stateDir, "models", modelFilename(cfg.ModelName))
		checks["model_present"] = fileExists(modelPath)
	case "cohere":
		checks["apple_silicon"] = isAppleSilicon()
		if uvPath, err := exec.LookPath("uv"); err == nil {
			checks["uv"] = true
			checks["uv_path"] = uvPath
		} else {
			checks["uv"] = false
		}
		checks["hf_model_id"] = strings.TrimSpace(cfg.HFModelID)
		checks["hf_token_present"] = strings.TrimSpace(os.Getenv("HF_TOKEN")) != ""
	}

	if global.Strict {
		if ok, _ := checks["ffmpeg"].(bool); !ok {
			env.Status = "failed"
			env.Errors = append(env.Errors, newError("dependency", "FFMPEG_NOT_FOUND", "ffmpeg not found", false, "Install ffmpeg or pass --ffmpeg-path"))
		}
		switch normalizeEngine(cfg.Engine) {
		case "whisper":
			if ok, _ := checks["whisper"].(bool); !ok && cfg.WhisperURL == "" {
				env.Status = "failed"
				env.Errors = append(env.Errors, newError("dependency", "WHISPER_NOT_READY", "whisper runtime missing and no --whisper-url provided", false, "Provide --whisper-url or install whisper-cli"))
			}
		case "cohere":
			if ok, _ := checks["uv"].(bool); !ok {
				env.Status = "failed"
				env.Errors = append(env.Errors, newError("dependency", "UV_NOT_FOUND", "uv not found for cohere engine", false, "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"))
			}
		}
	}

	if env.Status == "failed" {
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitFromStatus(env.Status)
	}

	env.Data = map[string]any{
		"valid":  true,
		"checks": checks,
	}
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleDoctor(args []string) (Envelope, int) {
	started := time.Now()
	args = hoistGlobalFlags(args)
	env := newEnvelope("doctor")
	global := defaultGlobalOptions()
	var help bool

	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, doctorHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby doctor --help' for usage")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, doctorHelp())
	}
	applyAgentMode(&global, nil)

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}

	checks := []DoctorCheck{}
	if p, err := exec.LookPath("ffmpeg"); err == nil {
		checks = append(checks, DoctorCheck{Name: "ffmpeg", Status: "ok", Details: p})
	} else {
		checks = append(checks, DoctorCheck{Name: "ffmpeg", Status: "missing", Details: "Install ffmpeg or pass --ffmpeg-path to run/validate"})
	}

	whisperRuntimePath := filepath.Join(stateDir, "runtime", binaryName("whisper-cli"))
	if fileExists(whisperRuntimePath) {
		checks = append(checks, DoctorCheck{Name: "whisper", Status: "ok", Details: whisperRuntimePath})
	} else if p, err := exec.LookPath(binaryName("whisper-cli")); err == nil {
		checks = append(checks, DoctorCheck{Name: "whisper", Status: "ok", Details: p})
	} else {
		checks = append(checks, DoctorCheck{Name: "whisper", Status: "missing", Details: "Will bootstrap from --whisper-url or SCRIBY_WHISPER_URL"})
	}

	modelPath := filepath.Join(stateDir, "models", modelFilename(defaultModelName))
	if fileExists(modelPath) {
		checks = append(checks, DoctorCheck{Name: "model.medium", Status: "ok", Details: modelPath})
	} else {
		checks = append(checks, DoctorCheck{Name: "model.medium", Status: "missing", Details: "Run: scriby models pull --name medium"})
	}

	if p, err := exec.LookPath("llm"); err == nil {
		checks = append(checks, DoctorCheck{Name: "llm", Status: "ok", Details: p})
	} else {
		checks = append(checks, DoctorCheck{Name: "llm", Status: "missing", Details: "Optional: needed only for description generation"})
	}

	if isAppleSilicon() {
		checks = append(checks, DoctorCheck{Name: "apple_silicon", Status: "ok", Details: "darwin/arm64"})
	} else {
		checks = append(checks, DoctorCheck{Name: "apple_silicon", Status: "missing", Details: "Cohere engine requires Apple Silicon (darwin/arm64)"})
	}

	if p, err := exec.LookPath("uv"); err == nil {
		checks = append(checks, DoctorCheck{Name: "uv", Status: "ok", Details: p})
	} else {
		checks = append(checks, DoctorCheck{Name: "uv", Status: "missing", Details: "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"})
	}

	if strings.TrimSpace(os.Getenv("HF_TOKEN")) != "" {
		checks = append(checks, DoctorCheck{Name: "hf_token", Status: "ok", Details: "HF_TOKEN is set"})
	} else {
		checks = append(checks, DoctorCheck{Name: "hf_token", Status: "missing", Details: "Recommended for gated Hugging Face models such as Cohere Transcribe"})
	}

	writable := false
	testFile := filepath.Join(stateDir, ".doctor-write-test")
	if err := os.WriteFile(testFile, []byte("ok"), 0o644); err == nil {
		_ = os.Remove(testFile)
		writable = true
	}
	if writable {
		checks = append(checks, DoctorCheck{Name: "state_dir", Status: "ok", Details: stateDir})
	} else {
		checks = append(checks, DoctorCheck{Name: "state_dir", Status: "failed", Details: "State directory is not writable"})
		env.Status = "failed"
		env.Errors = append(env.Errors, newError("filesystem", "STATE_DIR_NOT_WRITABLE", "state directory is not writable", false, "Set --state-dir to a writable directory"))
	}

	env.Data = map[string]any{
		"state_dir": stateDir,
		"checks":    checks,
	}
	if env.Status != "failed" {
		env.Status = "succeeded"
	}
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitFromStatus(env.Status)
}

func handleReplay(args []string) (Envelope, int) {
	started := time.Now()
	args = hoistGlobalFlags(args)
	env := newEnvelope("replay")
	global := defaultGlobalOptions()
	var help bool

	fs := flag.NewFlagSet("replay", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addHelpFlags(fs, &help)

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, replayHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby replay --help' for usage")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, replayHelp())
	}
	applyAgentMode(&global, nil)

	pos := fs.Args()
	if len(pos) != 1 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_RUN_ID", "replay requires exactly one <run_id>", false, "Usage: scriby replay [flags] <run_id>")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}

	runID := pos[0]
	runFile := filepath.Join(stateDir, "runs", runID+".json")
	b, rerr := os.ReadFile(runFile)
	if rerr != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "RUN_NOT_FOUND", fmt.Sprintf("run_id not found: %s", runID), false, "Use scriby run first, then replay the emitted run_id")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	var replayed Envelope
	if err := json.Unmarshal(b, &replayed); err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "RUN_RECORD_CORRUPTED", err.Error(), false, "Delete corrupted run record and rerun command")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitRuntime
	}

	env.Status = "succeeded"
	env.Data = map[string]any{
		"replayed_run": replayed,
	}
	env.Metrics["replayed_status"] = replayed.Status
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleRetry(args []string) (Envelope, int) {
	started := time.Now()
	args = hoistGlobalFlags(args)
	env := newEnvelope("retry")
	global := defaultGlobalOptions()
	failedOnly := false
	var help bool

	fs := flag.NewFlagSet("retry", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.BoolVar(&failedOnly, "failed-only", failedOnly, "Retry only files that failed in the saved run")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, retryHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby retry --help' for usage")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, retryHelp())
	}
	applyAgentMode(&global, nil)

	pos := fs.Args()
	if len(pos) != 1 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_RUN_ID", "retry requires exactly one <run_id>", false, "Usage: scriby retry [flags] <run_id>")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	global.StateDir = stateDir

	stored, err := loadStoredRunRecord(stateDir, pos[0])
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "RUN_NOT_FOUND", fmt.Sprintf("run_id not found or not retryable: %s", pos[0]), false, "Use scriby history list to find run IDs")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if stored.Data.Input == "" {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "RUN_NOT_RETRYABLE", "saved run does not include retryable run data", false, "Run with the current Scriby version, then retry the new run_id")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	inputs := []string{stored.Data.Input}
	if failedOnly {
		inputs = failedRunInputs(stored.Data.Files)
		if len(inputs) == 0 {
			env.Status = "succeeded"
			env.Warnings = []Warning{{Code: "NO_FAILED_FILES", Message: "No failed files were found in the saved run"}}
			env.Data = map[string]any{"retried_from": stored.RunID, "retried_runs": []Envelope{}}
			finishEnvelope(&env, started, 0, 0, 0)
			return env, exitOK
		}
	}

	retried := []Envelope{}
	successes := int64(0)
	failures := int64(0)
	for _, input := range inputs {
		runArgs := retryRunArgs(global, stored.Data, input)
		retryEnv, code := handleRun(runArgs)
		retried = append(retried, retryEnv)
		if code == exitOK {
			successes++
		} else {
			failures++
		}
	}

	if failures == 0 {
		env.Status = "succeeded"
	} else if successes > 0 {
		env.Status = "partial"
	} else {
		env.Status = "failed"
	}
	env.Data = map[string]any{"retried_from": stored.RunID, "failed_only": failedOnly, "retried_runs": retried}
	finishEnvelope(&env, started, int64(len(inputs)), successes, failures)
	_ = saveRunRecord(stateDir, env)
	return env, exitFromStatus(env.Status)
}

func handleModels(args []string) (Envelope, int) {
	started := time.Now()
	args = hoistGlobalFlags(args)
	env := newEnvelope("models")

	leadingGlobalArgs, remainingArgs, help, parseErr := splitRootArgs(args)
	if parseErr != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", parseErr.Error(), false, "Run 'scriby models --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help && len(remainingArgs) == 0 {
		env.Data = modelsHelp()
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitOK
	}
	if len(remainingArgs) == 0 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_MODELS_COMMAND", "models command requires one of: pull|list|prune", false, "Usage: scriby models <pull|list|prune> [flags]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	sub := remainingArgs[0]
	subArgs := append(append([]string{}, leadingGlobalArgs...), remainingArgs[1:]...)
	switch sub {
	case "pull":
		return handleModelsPull(subArgs, started)
	case "list":
		return handleModelsList(subArgs, started)
	case "prune":
		return handleModelsPrune(subArgs, started)
	case "help", "--help", "-h":
		env.Data = modelsHelp()
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitOK
	default:
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "UNKNOWN_MODELS_COMMAND", fmt.Sprintf("unknown models subcommand: %s", sub), false, "Usage: scriby models <pull|list|prune> [flags]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
}

func handleModelsPull(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("models.pull")
	global := defaultGlobalOptions()
	name := defaultModelName
	url := ""
	var help bool

	fs := flag.NewFlagSet("models pull", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.StringVar(&name, "name", name, "Model name (tiny|base|small|medium|large-v3)")
	fs.StringVar(&url, "url", url, "Override model URL")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, modelsHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby models pull --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, modelsHelp())
	}
	applyAgentMode(&global, nil)
	if !isValidOutputMode(global.Output) {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_OUTPUT_MODE", "--output must be one of json|jsonl|text", false, "Set --output json, --output jsonl, or --output text")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}

	ctx, cancel := commandContext(global.TimeoutMS)
	defer cancel()
	progress := newProgressReporter(global.Output, "models.pull", env.RunID)
	progress.Step("model.ensure", fmt.Sprintf("Preparing model %s", modelFilename(name)), map[string]any{"model": modelFilename(name)})

	path, merr := ensureModel(ctx, stateDir, name, url, global.MaxRetries, progress)
	if merr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*merr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	env.Data = map[string]any{"name": name, "path": path}
	_ = saveRunRecord(stateDir, env)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleModelsList(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("models.list")
	global := defaultGlobalOptions()
	var help bool

	fs := flag.NewFlagSet("models list", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, modelsHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby models list --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, modelsHelp())
	}
	applyAgentMode(&global, nil)
	if !isValidOutputMode(global.Output) {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_OUTPUT_MODE", "--output must be one of json|jsonl|text", false, "Set --output json, --output jsonl, or --output text")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}

	installed := []string{}
	modelsDir := filepath.Join(stateDir, "models")
	entries, err := os.ReadDir(modelsDir)
	if err == nil {
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			if strings.HasSuffix(strings.ToLower(e.Name()), ".bin") {
				installed = append(installed, filepath.Join(modelsDir, e.Name()))
			}
		}
	}
	sort.Strings(installed)

	known := make([]string, 0, len(knownModelURLs))
	for k := range knownModelURLs {
		known = append(known, k)
	}
	sort.Strings(known)

	env.Data = map[string]any{"known": known, "installed": installed}
	_ = saveRunRecord(stateDir, env)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleModelsPrune(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("models.prune")
	global := defaultGlobalOptions()
	name := ""
	var help bool

	fs := flag.NewFlagSet("models prune", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.StringVar(&name, "name", name, "Model name to prune (empty = all)")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, modelsHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby models prune --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, modelsHelp())
	}
	applyAgentMode(&global, nil)
	if !isValidOutputMode(global.Output) {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_OUTPUT_MODE", "--output must be one of json|jsonl|text", false, "Set --output json, --output jsonl, or --output text")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}

	if !global.Yes {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "CONFIRMATION_REQUIRED", "models prune requires --yes", false, "Re-run with --yes for non-interactive prune")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	modelsDir := filepath.Join(stateDir, "models")
	_ = os.MkdirAll(modelsDir, 0o755)
	removed := []string{}
	if name != "" {
		target := filepath.Join(modelsDir, modelFilename(name))
		if fileExists(target) {
			if err := os.Remove(target); err == nil {
				removed = append(removed, target)
			}
		}
	} else {
		entries, err := os.ReadDir(modelsDir)
		if err == nil {
			for _, e := range entries {
				if e.IsDir() {
					continue
				}
				if strings.HasSuffix(strings.ToLower(e.Name()), ".bin") {
					target := filepath.Join(modelsDir, e.Name())
					if err := os.Remove(target); err == nil {
						removed = append(removed, target)
					}
				}
			}
		}
	}
	sort.Strings(removed)
	env.Data = map[string]any{"removed": removed}
	_ = saveRunRecord(stateDir, env)
	finishEnvelope(&env, started, int64(len(removed)), int64(len(removed)), 0)
	return env, exitOK
}

func handleHistory(args []string) (Envelope, int) {
	started := time.Now()
	args = hoistGlobalFlags(args)
	env := newEnvelope("history")

	leadingGlobalArgs, remainingArgs, help, parseErr := splitRootArgs(args)
	if parseErr != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", parseErr.Error(), false, "Run 'scriby history --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help && len(remainingArgs) == 0 {
		env.Data = historyHelp()
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitOK
	}
	if len(remainingArgs) == 0 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_HISTORY_COMMAND", "history command requires one of: path|list|latest|show|search|export|schema|sql", false, "Usage: scriby history <path|list|latest|show|search|export|schema|sql> [flags]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	sub := remainingArgs[0]
	subArgs := hoistGlobalFlags(append(append([]string{}, leadingGlobalArgs...), remainingArgs[1:]...))
	switch sub {
	case "path":
		return handleHistoryPath(subArgs, started)
	case "list":
		return handleHistoryList(subArgs, started)
	case "latest":
		return handleHistoryLatest(subArgs, started)
	case "show":
		return handleHistoryShow(subArgs, started)
	case "search":
		return handleHistorySearch(subArgs, started)
	case "export":
		return handleHistoryExport(subArgs, started)
	case "schema":
		return handleHistorySchema(subArgs, started)
	case "sql":
		return handleHistorySQL(subArgs, started)
	case "help", "--help", "-h":
		env.Data = historyHelp()
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitOK
	default:
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "UNKNOWN_HISTORY_COMMAND", fmt.Sprintf("unknown history subcommand: %s", sub), false, "Usage: scriby history <path|list|latest|show|search|export|schema|sql> [flags]")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
}

func handleHistoryPath(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.path")
	global := defaultGlobalOptions()
	var help bool

	fs := flag.NewFlagSet("history path", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history path --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)

	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	db, err := history.Open(stateDir)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "HISTORY_DB_ERROR", err.Error(), false, "Check the SQLite database path or set --state-dir")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	_ = db.Close()
	env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir)}
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleHistoryList(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.list")
	global := defaultGlobalOptions()
	limit := 20
	status := ""
	since := ""
	var help bool

	fs := flag.NewFlagSet("history list", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.IntVar(&limit, "limit", limit, "Maximum runs to return")
	fs.StringVar(&status, "status", status, "Filter by status: succeeded|partial|failed")
	fs.StringVar(&since, "since", since, "Filter runs since RFC3339 date or duration like 7d, 24h")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history list --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)
	if limit <= 0 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_LIMIT", "--limit must be a positive integer", false, "Set --limit 20")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if status != "" && status != "succeeded" && status != "partial" && status != "failed" {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_STATUS", "--status must be one of succeeded|partial|failed", false, "Omit --status or use a supported value")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	sinceTime, serr := history.ParseSince(since, time.Now())
	if serr != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_SINCE", serr.Error(), false, "Use --since 7d, --since 24h, or an RFC3339 timestamp")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	db, stateDir, herr := openHistoryCommandDB(global)
	if herr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*herr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	defer db.Close()

	runs, err := history.ListRuns(db, limit, status, sinceTime)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "HISTORY_QUERY_FAILED", err.Error(), false, "Check the SQLite database")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitRuntime
	}
	env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir), "runs": runs}
	env.Metrics["rows_total"] = len(runs)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleHistoryLatest(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.latest")
	global := defaultGlobalOptions()
	includeTranscript := false
	transcriptOnly := false
	var help bool

	fs := flag.NewFlagSet("history latest", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.BoolVar(&includeTranscript, "include-transcript", includeTranscript, "Include full transcript and description text")
	fs.BoolVar(&transcriptOnly, "transcript-only", transcriptOnly, "Return only transcript text for the latest run")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history latest --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)
	if transcriptOnly {
		includeTranscript = true
	}

	db, stateDir, herr := openHistoryReadOnlyCommandDB(global)
	if herr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*herr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	defer db.Close()

	runID, err := history.LatestRunID(db)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "RUN_NOT_FOUND", "no history runs found", false, "Run scriby run first")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	run, files, envelopeJSON, err := history.GetRun(db, runID, includeTranscript)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "RUN_NOT_FOUND", fmt.Sprintf("run_id not found: %s", runID), false, "Use scriby history list to find run IDs")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if transcriptOnly {
		env.Data = map[string]any{"run_id": run.RunID, "transcripts": history.TranscriptsOnly(files)}
	} else {
		var replayed Envelope
		if envelopeJSON != "" {
			_ = json.Unmarshal([]byte(envelopeJSON), &replayed)
		}
		env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir), "run": run, "files": files, "envelope": replayed}
	}
	env.Metrics["rows_total"] = len(files)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleHistoryShow(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.show")
	global := defaultGlobalOptions()
	includeTranscript := false
	transcriptOnly := false
	var help bool

	fs := flag.NewFlagSet("history show", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.BoolVar(&includeTranscript, "include-transcript", includeTranscript, "Include full transcript and description text")
	fs.BoolVar(&transcriptOnly, "transcript-only", transcriptOnly, "Return only transcript text")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history show --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)
	if transcriptOnly {
		includeTranscript = true
	}
	pos := fs.Args()
	if len(pos) != 1 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_RUN_ID", "history show requires exactly one <run_id>", false, "Usage: scriby history show <run_id>")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	db, stateDir, herr := openHistoryCommandDB(global)
	if herr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*herr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	defer db.Close()

	run, files, envelopeJSON, err := history.GetRun(db, pos[0], includeTranscript)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "RUN_NOT_FOUND", fmt.Sprintf("run_id not found: %s", pos[0]), false, "Use scriby history list to find run IDs")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if transcriptOnly {
		env.Data = map[string]any{"run_id": run.RunID, "transcripts": history.TranscriptsOnly(files)}
	} else {
		var replayed Envelope
		if envelopeJSON != "" {
			_ = json.Unmarshal([]byte(envelopeJSON), &replayed)
		}
		env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir), "run": run, "files": files, "envelope": replayed}
	}
	env.Metrics["rows_total"] = len(files)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleHistorySearch(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.search")
	global := defaultGlobalOptions()
	limit := 20
	since := ""
	includeTranscript := false
	var help bool

	fs := flag.NewFlagSet("history search", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.IntVar(&limit, "limit", limit, "Maximum matches to return")
	fs.StringVar(&since, "since", since, "Filter matches since RFC3339 date or duration like 7d, 24h")
	fs.BoolVar(&includeTranscript, "include-transcript", includeTranscript, "Include full transcript and description text in matches")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history search --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)
	pos := fs.Args()
	if len(pos) != 1 || strings.TrimSpace(pos[0]) == "" {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_QUERY", "history search requires exactly one <query>", false, "Usage: scriby history search \"meeting topic\"")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if limit <= 0 {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_LIMIT", "--limit must be a positive integer", false, "Set --limit 20")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	sinceTime, serr := history.ParseSince(since, time.Now())
	if serr != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_SINCE", serr.Error(), false, "Use --since 7d, --since 24h, or an RFC3339 timestamp")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	db, stateDir, herr := openHistoryCommandDB(global)
	if herr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*herr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	defer db.Close()

	matches, err := history.Search(db, pos[0], limit, sinceTime, includeTranscript)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "HISTORY_QUERY_FAILED", err.Error(), false, "Check the SQLite database")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitRuntime
	}
	env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir), "query": pos[0], "matches": matches}
	env.Metrics["rows_total"] = len(matches)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleHistoryExport(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.export")
	global := defaultGlobalOptions()
	format := "markdown"
	latest := false
	var help bool

	fs := flag.NewFlagSet("history export", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	fs.StringVar(&format, "format", format, "Export format: markdown|json")
	fs.BoolVar(&latest, "latest", latest, "Export the latest run")
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history export --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)
	format = strings.ToLower(strings.TrimSpace(format))
	if format != "markdown" && format != "json" {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "INVALID_FORMAT", "--format must be markdown or json", false, "Use --format markdown or --format json")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	pos := fs.Args()
	if len(pos) > 1 || (!latest && len(pos) != 1) || (latest && len(pos) != 0) {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_RUN_ID", "history export requires <run_id> or --latest", false, "Usage: scriby history export <run_id> --format markdown")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	db, stateDir, herr := openHistoryCommandDB(global)
	if herr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*herr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	defer db.Close()

	runID := ""
	if latest {
		id, err := history.LatestRunID(db)
		if err != nil {
			env.Status = "failed"
			env.Errors = []AppError{newError("input", "RUN_NOT_FOUND", "no history runs found", false, "Run scriby run first")}
			finishEnvelope(&env, started, 0, 0, 0)
			return env, exitInput
		}
		runID = id
	} else {
		runID = pos[0]
	}
	run, files, _, err := history.GetRun(db, runID, true)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "RUN_NOT_FOUND", fmt.Sprintf("run_id not found: %s", runID), false, "Use scriby history list to find run IDs")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if format == "json" {
		env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir), "run": run, "files": files}
	} else {
		env.Data = map[string]any{"run_id": run.RunID, "format": format, "content": history.ExportMarkdown(run, files)}
	}
	env.Metrics["rows_total"] = len(files)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleHistorySchema(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.schema")
	global := defaultGlobalOptions()
	var help bool

	fs := flag.NewFlagSet("history schema", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history schema --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)

	db, stateDir, herr := openHistoryCommandDB(global)
	if herr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*herr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	defer db.Close()

	schema, err := history.Schema(db)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "HISTORY_SCHEMA_FAILED", err.Error(), false, "Check the SQLite database")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitRuntime
	}
	env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir), "schema": schema}
	env.Metrics["rows_total"] = len(schema)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func handleHistorySQL(args []string, started time.Time) (Envelope, int) {
	args = hoistGlobalFlags(args)
	env := newEnvelope("history.sql")
	global := defaultGlobalOptions()
	var help bool

	fs := flag.NewFlagSet("history sql", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	addGlobalFlags(fs, &global)
	addHelpFlags(fs, &help)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return finishHelp(&env, started, historyHelp())
		}
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "FLAG_PARSE_ERROR", err.Error(), false, "Run 'scriby history sql --help'")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if help {
		return finishHelp(&env, started, historyHelp())
	}
	applyAgentMode(&global, nil)
	query := strings.TrimSpace(strings.Join(fs.Args(), " "))
	if query == "" {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "MISSING_SQL", "history sql requires a read-only SQL statement", false, "Example: scriby history sql \"select run_id,status from runs limit 5\"")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}
	if !history.LooksReadOnlySQL(query) {
		env.Status = "failed"
		env.Errors = []AppError{newError("input", "UNSAFE_SQL", "history sql only accepts read-only SELECT, WITH, or PRAGMA statements", false, "Use sqlite directly if you need to mutate the database")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitInput
	}

	db, stateDir, herr := openHistoryReadOnlyCommandDB(global)
	if herr != nil {
		env.Status = "failed"
		env.Errors = []AppError{*herr}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitDependency
	}
	defer db.Close()

	rows, err := history.QuerySQL(db, query)
	if err != nil {
		env.Status = "failed"
		env.Errors = []AppError{newError("filesystem", "HISTORY_SQL_FAILED", err.Error(), false, "Check the SQL statement")}
		finishEnvelope(&env, started, 0, 0, 0)
		return env, exitRuntime
	}
	env.Data = map[string]any{"state_dir": stateDir, "database": history.DBPath(stateDir), "rows": rows}
	env.Metrics["rows_total"] = len(rows)
	finishEnvelope(&env, started, 0, 0, 0)
	return env, exitOK
}

func processMediaFile(
	ctx context.Context,
	cfg RunConfig,
	mediaPath string,
	ffmpegPath string,
	transcriberPath string,
	modelPath string,
	llmPath string,
	haveLLM bool,
	outputMode string,
	runID string,
	progress *ProgressReporter,
) (FileResult, []Warning, *AppError) {
	warnings := []Warning{}
	fr := FileResult{File: mediaPath, Status: "failed"}

	absMedia, err := filepath.Abs(mediaPath)
	if err != nil {
		ae := newError("input", "INVALID_MEDIA_PATH", err.Error(), false, "Use a valid media file path")
		fr.Error = &ae
		return fr, warnings, &ae
	}
	fr.File = absMedia

	progress.Step("convert.start", fmt.Sprintf("Converting audio: %s", filepath.Base(absMedia)), map[string]any{"file": absMedia, "sample_rate": cfg.SampleRate})
	wavPath, convertWarn, cerr := convertToTempWAV(ctx, ffmpegPath, absMedia, cfg.SampleRate, cfg.MonoMode)
	if convertWarn != nil {
		warnings = append(warnings, *convertWarn)
	}
	if cerr != nil {
		fr.Error = cerr
		return fr, warnings, cerr
	}
	progress.Step("convert.done", fmt.Sprintf("Converted audio: %s", filepath.Base(absMedia)), map[string]any{"file": absMedia})
	if !cfg.KeepTemp {
		defer os.Remove(wavPath)
	}

	transcript := strings.TrimSuffix(absMedia, filepath.Ext(absMedia)) + ".md"
	terr := transcribeAudio(ctx, cfg, transcriberPath, modelPath, wavPath, transcript, outputMode, runID, absMedia, progress)
	if terr != nil {
		fr.Error = terr
		return fr, warnings, terr
	}
	fr.Transcript = transcript

	promptPath := ""
	if cfg.Prompt != "" {
		if fileExists(cfg.Prompt) {
			promptPath = cfg.Prompt
		} else {
			warnings = append(warnings, Warning{Code: "PROMPT_NOT_FOUND", Message: fmt.Sprintf("Prompt missing: %s. Falling back to prompt.md", cfg.Prompt)})
		}
	}
	if promptPath == "" {
		defaultPrompt := filepath.Join(filepath.Dir(absMedia), "prompt.md")
		if fileExists(defaultPrompt) {
			promptPath = defaultPrompt
		}
	}

	if promptPath != "" {
		if !haveLLM {
			ae := newError("dependency", "LLM_NOT_FOUND", "llm CLI not found", false, "Install llm or set --llm-path")
			fr.Error = &ae
			return fr, warnings, &ae
		}
		desc := strings.TrimSuffix(absMedia, filepath.Ext(absMedia)) + "_description.md"
		progress.Step("description.start", fmt.Sprintf("Generating description: %s", filepath.Base(desc)), map[string]any{"file": absMedia, "prompt": promptPath})
		dErr := generateDescription(ctx, llmPath, transcript, promptPath, desc)
		if dErr != nil {
			fr.Error = dErr
			return fr, warnings, dErr
		}
		fr.Description = desc
		progress.Step("description.done", fmt.Sprintf("Description written: %s", desc), map[string]any{"file": absMedia, "description": desc})
	} else {
		warnings = append(warnings, Warning{Code: "PROMPT_NOT_SET", Message: fmt.Sprintf("No prompt found for %s; description generation skipped", absMedia)})
	}

	fr.Status = "succeeded"
	return fr, warnings, nil
}

func convertToTempWAV(ctx context.Context, ffmpegPath, input string, sampleRate int, monoMode string) (string, *Warning, *AppError) {
	tmp, err := os.CreateTemp("", "scriby-*.wav")
	if err != nil {
		ae := newError("filesystem", "TEMP_FILE_ERROR", err.Error(), true, "Ensure temp directory is writable")
		return "", nil, &ae
	}
	tmpPath := tmp.Name()
	_ = tmp.Close()

	channelCount := 1
	warn := (*Warning)(nil)
	if monoMode == "right" || monoMode == "average" {
		if c, err := detectAudioChannels(ctx, ffmpegPath, input); err == nil {
			channelCount = c
		} else {
			warn = &Warning{Code: "CHANNEL_DETECT_FALLBACK", Message: "Could not detect channel count; falling back to left channel when needed"}
		}
	}

	filter := "pan=mono|c0=c0"
	switch monoMode {
	case "right":
		if channelCount > 1 {
			filter = "pan=mono|c0=c1"
		}
	case "average":
		if channelCount > 1 {
			filter = "pan=mono|c0=0.5*c0+0.5*c1"
		}
	case "left":
		filter = "pan=mono|c0=c0"
	default:
		filter = "pan=mono|c0=c0"
	}

	args := []string{
		"-y",
		"-i", input,
		"-af", fmt.Sprintf("%s,aresample=%d:resampler=soxr:precision=28", filter, sampleRate),
		"-c:a", "pcm_s16le",
		tmpPath,
	}
	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	var stderr strings.Builder
	cmd.Stdout = io.Discard
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		_ = os.Remove(tmpPath)
		ae := newError("runtime", "FFMPEG_CONVERT_FAILED", err.Error(), true, trimHint(stderr.String()))
		return "", warn, &ae
	}
	return tmpPath, warn, nil
}

func detectAudioChannels(ctx context.Context, ffmpegPath string, input string) (int, error) {
	ffprobe := filepath.Join(filepath.Dir(ffmpegPath), binaryName("ffprobe"))
	if !fileExists(ffprobe) {
		p, err := exec.LookPath(binaryName("ffprobe"))
		if err != nil {
			return 1, errors.New("ffprobe not found")
		}
		ffprobe = p
	}
	cmd := exec.CommandContext(ctx, ffprobe,
		"-v", "error",
		"-select_streams", "a:0",
		"-show_entries", "stream=channels",
		"-of", "csv=p=0",
		input,
	)
	out, err := cmd.Output()
	if err != nil {
		return 1, err
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return 1, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 {
		return 1, nil
	}
	return n, nil
}

func transcribeAudio(ctx context.Context, cfg RunConfig, transcriberPath string, modelPath string, wavPath string, transcriptPath string, outputMode string, runID string, mediaPath string, progress *ProgressReporter) *AppError {
	switch normalizeEngine(cfg.Engine) {
	case "cohere":
		return transcribeWithCohere(ctx, transcriberPath, cfg.HFModelID, cfg.Language, wavPath, transcriptPath, outputMode, runID, mediaPath, progress)
	default:
		return transcribeWithWhisper(ctx, transcriberPath, modelPath, cfg.Language, wavPath, transcriptPath, cfg.StreamTranscript, cfg.Timestamps, outputMode, runID, mediaPath, progress)
	}
}

func transcribeWithWhisper(ctx context.Context, whisperPath string, modelPath string, language string, wavPath string, transcriptPath string, stream bool, timestamps bool, outputMode string, runID string, mediaPath string, progress *ProgressReporter) *AppError {
	progress.Step("transcribe.start", fmt.Sprintf("Transcribing: %s", filepath.Base(mediaPath)), map[string]any{"file": mediaPath})
	baseArgs := []string{
		"-m", modelPath,
		"-l", language,
		"-f", wavPath,
		"-np",
	}
	if !timestamps {
		baseArgs = append(baseArgs, "-nt")
	}

	if !stream && !timestamps {
		origBase := strings.TrimSuffix(transcriptPath, filepath.Ext(transcriptPath))
		args := append([]string{}, baseArgs...)
		args = append(args, "-otxt", "-of", origBase)
		cmd := exec.CommandContext(ctx, whisperPath, args...)
		var stderr strings.Builder
		cmd.Stdout = io.Discard
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			return ptrError(newError("runtime", "WHISPER_TRANSCRIBE_FAILED", err.Error(), true, trimHint(stderr.String())))
		}
		textPath := origBase + ".txt"
		if !fileExists(textPath) {
			return ptrError(newError("runtime", "WHISPER_OUTPUT_MISSING", "whisper did not emit .txt output", true, "Retry with --stream-transcript true"))
		}
		if fileExists(transcriptPath) {
			_ = os.Remove(transcriptPath)
		}
		if err := os.Rename(textPath, transcriptPath); err != nil {
			return ptrError(newError("filesystem", "TRANSCRIPT_RENAME_FAILED", err.Error(), true, "Check write permissions in media directory"))
		}
		progress.Step("transcribe.done", fmt.Sprintf("Transcript written: %s", transcriptPath), map[string]any{"file": mediaPath, "transcript": transcriptPath})
		return nil
	}

	out, err := os.Create(transcriptPath)
	if err != nil {
		return ptrError(newError("filesystem", "TRANSCRIPT_CREATE_FAILED", err.Error(), true, "Check write permissions in media directory"))
	}
	defer out.Close()

	cmd := exec.CommandContext(ctx, whisperPath, baseArgs...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return ptrError(newError("runtime", "WHISPER_PIPE_FAILED", err.Error(), true, "Retry command"))
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return ptrError(newError("runtime", "WHISPER_START_FAILED", err.Error(), true, "Check whisper runtime binary"))
	}

	scanner := bufio.NewScanner(stdout)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)
	segmentCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		_, _ = out.WriteString(line)
		_, _ = out.WriteString("\n")
		segmentCount++
		if outputMode == "jsonl" {
			emitJSONLEvent("run", runID, "transcript.segment", map[string]any{"file": mediaPath, "text": line})
		} else if segmentCount%20 == 0 {
			progress.Step(
				"transcribe.progress",
				fmt.Sprintf("Transcribing %s: %d segments", filepath.Base(mediaPath), segmentCount),
				map[string]any{"file": mediaPath, "segments": segmentCount},
			)
		}
	}
	if err := scanner.Err(); err != nil {
		return ptrError(newError("runtime", "WHISPER_STREAM_ERROR", err.Error(), true, "Retry with --stream-transcript false"))
	}
	if err := cmd.Wait(); err != nil {
		return ptrError(newError("runtime", "WHISPER_TRANSCRIBE_FAILED", err.Error(), true, trimHint(stderr.String())))
	}
	progress.Step("transcribe.done", fmt.Sprintf("Transcript written: %s", transcriptPath), map[string]any{"file": mediaPath, "transcript": transcriptPath})
	return nil
}

func transcribeWithCohere(ctx context.Context, uvPath string, modelID string, language string, wavPath string, transcriptPath string, outputMode string, runID string, mediaPath string, progress *ProgressReporter) *AppError {
	progress.Step("transcribe.start", fmt.Sprintf("Transcribing: %s", filepath.Base(mediaPath)), map[string]any{"file": mediaPath, "engine": "cohere"})

	outputBase := strings.TrimSuffix(transcriptPath, filepath.Ext(transcriptPath))
	args := []string{
		"run", "--with", "mlx-audio",
		"python", "-m", "mlx_audio.stt.generate",
		"--model", strings.TrimSpace(modelID),
		"--audio", wavPath,
		"--output-path", outputBase,
		"--format", "txt",
		"--language", normalizeCohereLanguage(language),
	}
	cmd := exec.CommandContext(ctx, uvPath, args...)
	var stdout strings.Builder
	var stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		hint := trimHint(stderr.String())
		if hint == "" {
			hint = trimHint(stdout.String())
		}
		return ptrError(newError("runtime", "COHERE_TRANSCRIBE_FAILED", err.Error(), true, hint))
	}

	mlxOutput := outputBase + ".txt"
	if !fileExists(mlxOutput) {
		return ptrError(newError("runtime", "COHERE_OUTPUT_MISSING", "mlx-audio did not write transcript output", true, "Check stderr output and Hugging Face access"))
	}
	if mlxOutput != transcriptPath {
		if fileExists(transcriptPath) {
			_ = os.Remove(transcriptPath)
		}
		if err := os.Rename(mlxOutput, transcriptPath); err != nil {
			return ptrError(newError("filesystem", "TRANSCRIPT_RENAME_FAILED", err.Error(), true, "Check write permissions in media directory"))
		}
	}

	if outputMode == "jsonl" {
		emitJSONLEvent("run", runID, "transcript.complete", map[string]any{"file": mediaPath, "transcript": transcriptPath, "engine": "cohere"})
	}
	progress.Step("transcribe.done", fmt.Sprintf("Transcript written: %s", transcriptPath), map[string]any{"file": mediaPath, "transcript": transcriptPath, "engine": "cohere"})
	return nil
}

func generateDescription(ctx context.Context, llmPath string, transcriptPath string, promptPath string, outputPath string) *AppError {
	promptBytes, err := os.ReadFile(promptPath)
	if err != nil {
		return ptrError(newError("input", "PROMPT_READ_FAILED", err.Error(), false, "Use a readable prompt file"))
	}

	in, err := os.Open(transcriptPath)
	if err != nil {
		return ptrError(newError("filesystem", "TRANSCRIPT_READ_FAILED", err.Error(), true, "Ensure transcript exists and is readable"))
	}
	defer in.Close()

	out, err := os.Create(outputPath)
	if err != nil {
		return ptrError(newError("filesystem", "DESCRIPTION_CREATE_FAILED", err.Error(), true, "Check write permissions in media directory"))
	}
	defer out.Close()

	cmd := exec.CommandContext(ctx, llmPath, "-s", string(promptBytes))
	cmd.Stdin = in
	cmd.Stdout = out
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return ptrError(newError("runtime", "LLM_DESCRIPTION_FAILED", err.Error(), true, trimHint(stderr.String())))
	}
	return nil
}

func ensureFFmpegPath(explicit string) (string, *AppError) {
	if explicit != "" {
		if fileExists(explicit) {
			return explicit, nil
		}
		ae := newError("dependency", "FFMPEG_NOT_FOUND", fmt.Sprintf("ffmpeg not found at %s", explicit), false, "Pass a valid --ffmpeg-path or install ffmpeg")
		return "", &ae
	}
	p, err := exec.LookPath(binaryName("ffmpeg"))
	if err != nil {
		ae := newError("dependency", "FFMPEG_NOT_FOUND", "ffmpeg binary not found in PATH", false, "Install ffmpeg or pass --ffmpeg-path")
		return "", &ae
	}
	return p, nil
}

func ensureWhisperPath(ctx context.Context, stateDir string, explicitPath string, whisperURL string, runtimeManifestURL string, maxRetries int, progress *ProgressReporter) (string, *AppError) {
	if explicitPath != "" {
		if fileExists(explicitPath) {
			progress.Step("runtime.ready", fmt.Sprintf("Using whisper runtime: %s", explicitPath), map[string]any{"path": explicitPath, "source": "flag"})
			return explicitPath, nil
		}
		ae := newError("dependency", "WHISPER_NOT_FOUND", fmt.Sprintf("whisper binary not found at %s", explicitPath), false, "Pass a valid --whisper-path")
		return "", &ae
	}

	runtimePath := filepath.Join(stateDir, "runtime", binaryName("whisper-cli"))
	if fileExists(runtimePath) {
		progress.Step("runtime.ready", fmt.Sprintf("Using cached whisper runtime: %s", runtimePath), map[string]any{"path": runtimePath, "source": "cache"})
		return runtimePath, nil
	}

	if p, err := exec.LookPath(binaryName("whisper-cli")); err == nil {
		progress.Step("runtime.ready", fmt.Sprintf("Using whisper runtime from PATH: %s", p), map[string]any{"path": p, "source": "path"})
		return p, nil
	}

	if err := os.MkdirAll(filepath.Dir(runtimePath), 0o755); err != nil {
		ae := newError("filesystem", "RUNTIME_DIR_CREATE_FAILED", err.Error(), true, "Set --state-dir to a writable directory")
		return "", &ae
	}

	url := strings.TrimSpace(whisperURL)
	checksum := ""
	if url == "" {
		manifestURL := strings.TrimSpace(runtimeManifestURL)
		if manifestURL == "" {
			manifestURL = defaultRuntimeManifestURL
		}
		progress.Step("runtime.manifest", fmt.Sprintf("Resolving runtime asset from manifest: %s", manifestURL), map[string]any{"manifest_url": manifestURL})
		manifest, err := loadRuntimeManifest(ctx, manifestURL, maxRetries, progress)
		if err != nil {
			ae := newError("network", "RUNTIME_MANIFEST_FETCH_FAILED", err.Error(), true, "Set --runtime-manifest-url to a reachable runtime manifest")
			return "", &ae
		}
		asset, err := selectRuntimeAsset(manifest, runtime.GOOS, runtime.GOARCH)
		if err != nil {
			ae := newError("dependency", "RUNTIME_ASSET_UNAVAILABLE", err.Error(), false, "Publish matching whisper-cli runtime asset for this OS/arch")
			return "", &ae
		}
		url = asset.URL
		checksum = asset.SHA256
	}

	progress.Step("runtime.download", fmt.Sprintf("Installing whisper runtime into %s", runtimePath), map[string]any{"url": url, "path": runtimePath})
	if err := downloadAndInstallWhisper(ctx, url, checksum, runtimePath, maxRetries, progress); err != nil {
		ae := newError("dependency", "WHISPER_BOOTSTRAP_FAILED", err.Error(), true, "Set --whisper-url or --runtime-manifest-url to valid runtime sources")
		return "", &ae
	}
	progress.Step("runtime.ready", fmt.Sprintf("Whisper runtime installed at %s", runtimePath), map[string]any{"path": runtimePath, "source": "download"})
	return runtimePath, nil
}

func ensureMLXAudioRuntime() (string, *AppError) {
	uvPath, err := exec.LookPath("uv")
	if err != nil {
		ae := newError("dependency", "UV_NOT_FOUND", "uv not found in PATH", false, "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
		return "", &ae
	}
	return uvPath, nil
}

func loadRuntimeManifest(ctx context.Context, manifestURL string, maxRetries int, progress *ProgressReporter) (RuntimeManifest, error) {
	tmp, err := os.CreateTemp("", "scriby-runtime-manifest-*.json")
	if err != nil {
		return RuntimeManifest{}, err
	}
	tmpPath := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(tmpPath)

	if err := downloadFile(ctx, manifestURL, tmpPath, maxRetries, progress, "runtime-manifest.json"); err != nil {
		return RuntimeManifest{}, err
	}

	b, err := os.ReadFile(tmpPath)
	if err != nil {
		return RuntimeManifest{}, err
	}

	var manifest RuntimeManifest
	if err := json.Unmarshal(b, &manifest); err != nil {
		return RuntimeManifest{}, err
	}
	if len(manifest.Assets) == 0 {
		return RuntimeManifest{}, errors.New("runtime manifest has no assets")
	}
	return manifest, nil
}

func selectRuntimeAsset(manifest RuntimeManifest, goos string, goarch string) (RuntimeAsset, error) {
	wantedOS := strings.ToLower(strings.TrimSpace(goos))
	wantedArch := normalizeArch(goarch)
	for _, asset := range manifest.Assets {
		assetOS := strings.ToLower(strings.TrimSpace(asset.OS))
		assetArch := normalizeArch(asset.Arch)
		assetName := strings.ToLower(strings.TrimSpace(asset.Name))
		if assetOS != wantedOS || assetArch != wantedArch {
			continue
		}
		if assetName != "" && assetName != "whisper-cli" {
			continue
		}
		if strings.TrimSpace(asset.URL) == "" {
			continue
		}
		return asset, nil
	}
	return RuntimeAsset{}, fmt.Errorf("no runtime asset found for %s/%s", wantedOS, wantedArch)
}

func normalizeArch(arch string) string {
	a := strings.ToLower(strings.TrimSpace(arch))
	switch a {
	case "x86_64":
		return "amd64"
	case "aarch64":
		return "arm64"
	default:
		return a
	}
}

func ensureModel(ctx context.Context, stateDir string, modelName string, modelURL string, maxRetries int, progress *ProgressReporter) (string, *AppError) {
	if modelName == "" {
		modelName = defaultModelName
	}
	modelsDir := filepath.Join(stateDir, "models")
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		ae := newError("filesystem", "MODELS_DIR_CREATE_FAILED", err.Error(), true, "Set --state-dir to a writable directory")
		return "", &ae
	}

	target := filepath.Join(modelsDir, modelFilename(modelName))
	if fileExists(target) {
		progress.Step("model.ready", fmt.Sprintf("Using cached model: %s", target), map[string]any{"model": modelFilename(modelName), "path": target, "source": "cache"})
		return target, nil
	}

	url := strings.TrimSpace(modelURL)
	if url == "" {
		if u, ok := knownModelURLs[modelName]; ok {
			url = u
		} else {
			ae := newError("input", "UNKNOWN_MODEL", fmt.Sprintf("unknown model '%s'", modelName), false, "Use --model-url for custom model or choose tiny|base|small|medium|large-v3")
			return "", &ae
		}
	}

	tmp := target + ".tmp"
	progress.Step(
		"model.download.notice",
		modelDownloadNotice(modelFilename(modelName), target),
		map[string]any{
			"model":        modelFilename(modelName),
			"path":         target,
			"size_hint":    modelSizeHint(modelName),
			"cached_after": true,
		},
	)
	progress.Step("model.download", fmt.Sprintf("Downloading model %s", modelFilename(modelName)), map[string]any{"model": modelFilename(modelName), "url": url})
	if err := downloadFile(ctx, url, tmp, maxRetries, progress, modelFilename(modelName)); err != nil {
		ae := newError("network", "MODEL_DOWNLOAD_FAILED", err.Error(), true, "Check network, model URL, or use --model-url")
		return "", &ae
	}
	if err := os.Rename(tmp, target); err != nil {
		_ = os.Remove(tmp)
		ae := newError("filesystem", "MODEL_INSTALL_FAILED", err.Error(), true, "Ensure models directory is writable")
		return "", &ae
	}
	if err := os.Chmod(target, 0o644); err != nil {
		// Non-fatal on some filesystems.
	}
	progress.Step("model.ready", fmt.Sprintf("Model ready: %s", target), map[string]any{"model": modelFilename(modelName), "path": target, "source": "download"})
	return target, nil
}

func modelDownloadNotice(model string, target string) string {
	sizeHint := modelSizeHint(model)
	if sizeHint != "" {
		return fmt.Sprintf("First-time model download for %s (%s). This can take several minutes; Scriby caches it at %s for future runs.", model, sizeHint, target)
	}
	return fmt.Sprintf("First-time model download for %s. This can take several minutes; Scriby caches it at %s for future runs.", model, target)
}

func modelSizeHint(model string) string {
	switch strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(model), "ggml-"), ".bin") {
	case "tiny":
		return "about 75 MiB"
	case "base":
		return "about 140 MiB"
	case "small":
		return "about 465 MiB"
	case "medium":
		return "about 1.4 GiB"
	case "large-v3":
		return "about 3 GiB"
	default:
		return ""
	}
}

func downloadAndInstallWhisper(ctx context.Context, url string, expectedSHA256 string, destBinaryPath string, maxRetries int, progress *ProgressReporter) error {
	tmp, err := os.CreateTemp("", "scriby-whisper-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(tmpPath)

	if err := downloadFile(ctx, url, tmpPath, maxRetries, progress, "whisper-runtime"); err != nil {
		return err
	}

	if strings.TrimSpace(expectedSHA256) != "" {
		if err := verifyFileSHA256(tmpPath, expectedSHA256); err != nil {
			return err
		}
	}

	lowerURL := strings.ToLower(url)
	binary := binaryName("whisper-cli")
	if strings.HasSuffix(lowerURL, ".zip") {
		if err := extractZipBinary(tmpPath, binary, destBinaryPath); err != nil {
			return err
		}
	} else if strings.HasSuffix(lowerURL, ".tar.gz") || strings.HasSuffix(lowerURL, ".tgz") {
		if err := extractTarGzBinary(tmpPath, binary, destBinaryPath); err != nil {
			return err
		}
	} else {
		if err := copyFile(tmpPath, destBinaryPath); err != nil {
			return err
		}
	}

	if runtime.GOOS != "windows" {
		if err := os.Chmod(destBinaryPath, 0o755); err != nil {
			return err
		}
	}
	return nil
}

func downloadFile(ctx context.Context, url string, dest string, maxRetries int, progress *ProgressReporter, artifactLabel string) error {
	if maxRetries < 0 {
		maxRetries = 0
	}

	label := strings.TrimSpace(artifactLabel)
	if label == "" {
		label = downloadLabel(url)
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(attempt*250) * time.Millisecond
			progress.Step(
				"download.retry",
				fmt.Sprintf("Retrying download for %s (attempt %d/%d)", label, attempt+1, maxRetries+1),
				map[string]any{"artifact": label, "attempt": attempt + 1, "max_attempts": maxRetries + 1},
			)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		req.Header.Set("User-Agent", "scriby-cli/1.0")
		progress.Step("download.start", fmt.Sprintf("Downloading %s", label), map[string]any{"artifact": label, "url": url, "attempt": attempt + 1, "max_attempts": maxRetries + 1})

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("download failed: %s", resp.Status)
			continue
		}

		f, err := os.Create(dest)
		if err != nil {
			_ = resp.Body.Close()
			return err
		}

		total := resp.ContentLength
		var downloaded int64
		lastProgressBucket := int64(-1)
		buffer := make([]byte, 64*1024)
		copyErr := error(nil)
		for {
			n, readErr := resp.Body.Read(buffer)
			if n > 0 {
				if _, writeErr := f.Write(buffer[:n]); writeErr != nil {
					copyErr = writeErr
					break
				}
				downloaded += int64(n)

				if total > 0 {
					percent := (downloaded * 100) / total
					if percent > 100 {
						percent = 100
					}
					bucket := percent / 10
					if bucket != lastProgressBucket {
						lastProgressBucket = bucket
						progress.Step(
							"download.progress",
							fmt.Sprintf("Downloading %s: %d%% (%s/%s)", label, percent, humanBytes(downloaded), humanBytes(total)),
							map[string]any{"artifact": label, "url": url, "downloaded_bytes": downloaded, "total_bytes": total, "percent": percent},
						)
					}
				} else if downloaded%(5*1024*1024) < int64(n) {
					progress.Step(
						"download.progress",
						fmt.Sprintf("Downloading %s: %s", label, humanBytes(downloaded)),
						map[string]any{"artifact": label, "url": url, "downloaded_bytes": downloaded},
					)
				}
			}

			if readErr != nil {
				if errors.Is(readErr, io.EOF) {
					break
				}
				copyErr = readErr
				break
			}
		}
		closeErr := resp.Body.Close()
		_ = f.Close()
		if copyErr != nil {
			lastErr = copyErr
			continue
		}
		if closeErr != nil {
			lastErr = closeErr
			continue
		}
		progress.Step(
			"download.done",
			fmt.Sprintf("Downloaded %s (%s)", label, humanBytes(downloaded)),
			map[string]any{"artifact": label, "url": url, "downloaded_bytes": downloaded},
		)
		return nil
	}
	if lastErr == nil {
		lastErr = errors.New("unknown download error")
	}
	return lastErr
}

func extractTarGzBinary(archivePath string, binaryName string, destPath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if hdr.FileInfo().IsDir() {
			continue
		}
		if filepath.Base(hdr.Name) != binaryName {
			continue
		}
		out, err := os.Create(destPath)
		if err != nil {
			return err
		}
		_, err = io.Copy(out, tr)
		_ = out.Close()
		if err != nil {
			return err
		}
		return nil
	}
	return fmt.Errorf("binary %s not found in archive", binaryName)
}

func extractZipBinary(archivePath string, binaryName string, destPath string) error {
	zr, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer zr.Close()

	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}
		if filepath.Base(f.Name) != binaryName {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(destPath)
		if err != nil {
			_ = rc.Close()
			return err
		}
		_, err = io.Copy(out, rc)
		_ = out.Close()
		_ = rc.Close()
		if err != nil {
			return err
		}
		return nil
	}
	return fmt.Errorf("binary %s not found in archive", binaryName)
}

func listInputFiles(input string) ([]string, *AppError) {
	if fileExists(input) {
		if !isSupportedMediaFile(input) {
			ae := newError("input", "UNSUPPORTED_FILE_TYPE", "input file is not supported media (mp4,m4a,mp3,mov,wav)", false, "Provide a supported media file")
			return nil, &ae
		}
		return []string{input}, nil
	}
	if !dirExists(input) {
		ae := newError("input", "INPUT_NOT_FOUND", fmt.Sprintf("input path not found: %s", input), false, "Provide an existing file or directory")
		return nil, &ae
	}

	entries, err := os.ReadDir(input)
	if err != nil {
		ae := newError("filesystem", "READ_DIR_FAILED", err.Error(), true, "Check permissions for input directory")
		return nil, &ae
	}
	files := []string{}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		p := filepath.Join(input, entry.Name())
		if isSupportedMediaFile(p) {
			files = append(files, p)
		}
	}
	sort.Strings(files)
	if len(files) == 0 {
		ae := newError("input", "NO_MEDIA_FILES", "no supported media files found in directory", false, "Directory mode scans top-level .mp4, .m4a, .mp3, .mov, .wav files")
		return nil, &ae
	}
	return files, nil
}

func isSupportedMediaFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".mp4", ".m4a", ".mp3", ".mov", ".wav":
		return true
	default:
		return false
	}
}

func ensureStateDir(explicit string) (string, error) {
	return state.Ensure(explicit)
}

func ensureSession(stateDir string, policy string, sessionID string) (string, *AppError) {
	switch policy {
	case "", "ephemeral":
		return "", nil
	case "sticky", "resume":
		sid := sessionID
		if sid == "" {
			sid = "default"
		}
		sessionsDir := filepath.Join(stateDir, "sessions")
		if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
			ae := newError("filesystem", "SESSION_DIR_CREATE_FAILED", err.Error(), true, "Set --state-dir to writable directory")
			return "", &ae
		}
		sessionFile := filepath.Join(sessionsDir, sanitizeFileName(sid)+".json")
		payload := map[string]any{"session_id": sid, "policy": policy, "updated_at": time.Now().UTC().Format(time.RFC3339)}
		b, _ := json.Marshal(payload)
		if err := os.WriteFile(sessionFile, b, 0o644); err != nil {
			ae := newError("filesystem", "SESSION_WRITE_FAILED", err.Error(), true, "Check state directory permissions")
			return "", &ae
		}
		return sid, nil
	default:
		ae := newError("input", "INVALID_SESSION_POLICY", "--session-policy must be ephemeral|sticky|resume", false, "Set --session-policy ephemeral, sticky, or resume")
		return "", &ae
	}
}

func validateRunInputs(cfg RunConfig) ([]Warning, *AppError) {
	warnings := []Warning{}
	engine := normalizeEngine(cfg.Engine)

	if cfg.SampleRate <= 0 {
		ae := newError("input", "INVALID_SAMPLE_RATE", "--sample-rate must be a positive integer", false, "Set --sample-rate 16000")
		return warnings, &ae
	}

	switch engine {
	case "whisper", "cohere":
	default:
		ae := newError("input", "INVALID_ENGINE", "--engine must be one of whisper|cohere", false, "Use --engine whisper or --engine cohere")
		return warnings, &ae
	}

	switch cfg.MonoMode {
	case "left", "right", "average", "avg", "blend":
	default:
		ae := newError("input", "INVALID_MONO_MODE", "--mono-mode must be one of left|right|average", false, "Use --mono-mode left, right, or average")
		return warnings, &ae
	}

	if !(fileExists(cfg.Input) || dirExists(cfg.Input)) {
		ae := newError("input", "INPUT_NOT_FOUND", fmt.Sprintf("input not found: %s", cfg.Input), false, "Provide a valid file or directory")
		return warnings, &ae
	}

	switch normalizeClipboardMode(cfg.Clipboard) {
	case "never", "ask", "always":
	default:
		ae := newError("input", "INVALID_CLIPBOARD_MODE", "--clipboard must be one of never|ask|always", false, "Use --clipboard never, ask, or always")
		return warnings, &ae
	}

	if cfg.Prompt != "" && !fileExists(cfg.Prompt) {
		warnings = append(warnings, Warning{Code: "PROMPT_NOT_FOUND", Message: fmt.Sprintf("Prompt file not found: %s. Run will fall back to prompt.md.", cfg.Prompt)})
	}

	if engine == "whisper" && cfg.ModelName == "" {
		warnings = append(warnings, Warning{Code: "MODEL_DEFAULTED", Message: "No model specified. Falling back to medium."})
	}

	if engine == "cohere" {
		if !isAppleSilicon() {
			ae := newError("input", "COHERE_REQUIRES_APPLE_SILICON", "the cohere engine requires Apple Silicon (darwin/arm64)", false, "Use --engine whisper on this platform")
			return warnings, &ae
		}
		if cfg.Timestamps {
			ae := newError("input", "COHERE_TIMESTAMPS_UNSUPPORTED", "--timestamps is not supported by the cohere engine", false, "Run without --timestamps or use --engine whisper")
			return warnings, &ae
		}
		normalizedLang := normalizeCohereLanguage(cfg.Language)
		if normalizedLang == "" {
			ae := newError("input", "MISSING_LANGUAGE", "--language must be set for the cohere engine", false, "Use --language en or another supported language")
			return warnings, &ae
		}
		if _, ok := cohereSupportedLanguages[normalizedLang]; !ok {
			ae := newError("input", "UNSUPPORTED_COHERE_LANGUAGE", fmt.Sprintf("unsupported cohere language: %s", cfg.Language), false, "Use one of ar|de|el|en|es|fr|it|ja|ko|nl|pl|pt|vi|zh")
			return warnings, &ae
		}
	}

	return warnings, nil
}

func normalizeClipboardMode(mode string) string {
	return strings.ToLower(strings.TrimSpace(mode))
}

func isAppleSilicon() bool {
	return runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
}

func normalizeEngine(engine string) string {
	e := strings.ToLower(strings.TrimSpace(engine))
	if e == "" {
		return defaultEngineName
	}
	return e
}

func normalizeCohereLanguage(language string) string {
	normalized := strings.ToLower(strings.TrimSpace(language))
	if mapped, ok := cohereSupportedLanguages[normalized]; ok {
		return mapped
	}
	return normalized
}

func maybeHandleClipboard(global GlobalOptions, cfg RunConfig, files []FileResult, progress *ProgressReporter) []Warning {
	mode := normalizeClipboardMode(cfg.Clipboard)
	if mode == "never" {
		return nil
	}

	transcriptPath, warn := clipboardTranscriptPath(files)
	if warn != nil {
		return []Warning{*warn}
	}
	if transcriptPath == "" {
		return nil
	}

	if mode == "ask" {
		if !canPrompt(global) {
			return []Warning{{
				Code:    "CLIPBOARD_PROMPT_SKIPPED",
				Message: "Clipboard prompt skipped because interactive prompts are disabled or no terminal is attached",
			}}
		}
		approved, err := promptYesNo(os.Stdin, os.Stderr, "Copy transcript to clipboard? [Y/n] ")
		if err != nil {
			return []Warning{{
				Code:    "CLIPBOARD_PROMPT_FAILED",
				Message: fmt.Sprintf("Clipboard prompt failed: %v", err),
			}}
		}
		if !approved {
			progress.Step("clipboard.skip", "Clipboard copy skipped", map[string]any{"transcript": transcriptPath})
			return nil
		}
	}

	if err := clipboard.CopyFile(transcriptPath); err != nil {
		return []Warning{{
			Code:    "CLIPBOARD_COPY_FAILED",
			Message: fmt.Sprintf("Failed to copy transcript to clipboard: %v", err),
		}}
	}
	progress.Step("clipboard.done", fmt.Sprintf("Copied transcript to clipboard: %s", transcriptPath), map[string]any{"transcript": transcriptPath})
	return nil
}

func clipboardTranscriptPath(files []FileResult) (string, *Warning) {
	transcripts := []string{}
	for _, file := range files {
		if file.Status == "succeeded" && strings.TrimSpace(file.Transcript) != "" {
			transcripts = append(transcripts, file.Transcript)
		}
	}
	if len(transcripts) == 0 {
		return "", nil
	}
	if len(transcripts) > 1 {
		warn := Warning{
			Code:    "CLIPBOARD_SKIPPED_MULTIPLE_TRANSCRIPTS",
			Message: fmt.Sprintf("Clipboard copy skipped because %d transcripts were produced; clipboard mode currently supports exactly 1 transcript", len(transcripts)),
		}
		return "", &warn
	}
	return transcripts[0], nil
}

func canPrompt(global GlobalOptions) bool {
	if global.NonInteractive {
		return false
	}
	return isCharDevice(os.Stdin) && isCharDevice(os.Stderr)
}

func isCharDevice(f *os.File) bool {
	if f == nil {
		return false
	}
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

func promptYesNo(r io.Reader, w io.Writer, prompt string) (bool, error) {
	if strings.TrimSpace(prompt) != "" {
		if _, err := fmt.Fprint(w, prompt); err != nil {
			return false, err
		}
	}
	line, err := bufio.NewReader(r).ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return false, err
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "", "y", "yes":
		return true, nil
	case "n", "no":
		return false, nil
	default:
		return false, nil
	}
}

func commandContext(timeoutMS int) (context.Context, context.CancelFunc) {
	if timeoutMS <= 0 {
		return context.WithCancel(context.Background())
	}
	return context.WithTimeout(context.Background(), time.Duration(timeoutMS)*time.Millisecond)
}

func saveRunRecord(stateDir string, env Envelope) error {
	runsDir := filepath.Join(stateDir, "runs")
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(runsDir, env.RunID+".json")
	b, err := json.MarshalIndent(env, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func saveIdempotencyRecord(stateDir string, command string, key string, env Envelope) error {
	dir := filepath.Join(stateDir, "idempotency")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(dir, command+"-"+sanitizeFileName(key)+".json")
	b, err := json.MarshalIndent(env, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func loadIdempotencyRecord(stateDir string, command string, key string) (Envelope, bool) {
	path := filepath.Join(stateDir, "idempotency", command+"-"+sanitizeFileName(key)+".json")
	b, err := os.ReadFile(path)
	if err != nil {
		return Envelope{}, false
	}
	var env Envelope
	if err := json.Unmarshal(b, &env); err != nil {
		return Envelope{}, false
	}
	return env, true
}

func loadStoredRunRecord(stateDir string, runID string) (StoredRunEnvelope, error) {
	path := filepath.Join(stateDir, "runs", runID+".json")
	b, err := os.ReadFile(path)
	if err != nil {
		db, dbErr := history.Open(stateDir)
		if dbErr != nil {
			return StoredRunEnvelope{}, err
		}
		defer db.Close()
		if scanErr := db.QueryRow(`SELECT envelope_json FROM runs WHERE run_id = ?`, runID).Scan(&b); scanErr != nil {
			return StoredRunEnvelope{}, err
		}
	}
	var stored StoredRunEnvelope
	if err := json.Unmarshal(b, &stored); err != nil {
		return StoredRunEnvelope{}, err
	}
	if stored.Command != "run" {
		return StoredRunEnvelope{}, fmt.Errorf("run_id %s is command %s, not run", runID, stored.Command)
	}
	return stored, nil
}

func failedRunInputs(files []FileResult) []string {
	inputs := []string{}
	seen := map[string]bool{}
	for _, file := range files {
		if file.Status == "succeeded" || strings.TrimSpace(file.File) == "" {
			continue
		}
		if !seen[file.File] {
			inputs = append(inputs, file.File)
			seen[file.File] = true
		}
	}
	sort.Strings(inputs)
	return inputs
}

func retryRunArgs(global GlobalOptions, data RunData, input string) []string {
	args := []string{
		"--output", global.Output,
		"--state-dir", global.StateDir,
		"--max-retries", strconv.Itoa(global.MaxRetries),
	}
	if global.Agent {
		args = append(args, "--agent")
	}
	if global.Strict {
		args = append(args, "--strict")
	}
	if global.NonInteractive {
		args = append(args, "--non-interactive")
	}
	if global.Yes {
		args = append(args, "--yes")
	}
	if global.TimeoutMS > 0 {
		args = append(args, "--timeout-ms", strconv.Itoa(global.TimeoutMS))
	}
	if data.Prompt != "" {
		args = append(args, "--prompt", data.Prompt)
	}
	if data.Engine != "" {
		args = append(args, "--engine", data.Engine)
	}
	if data.Language != "" {
		args = append(args, "--language", data.Language)
	}
	clipboard := data.Clipboard
	if global.Agent {
		clipboard = "never"
	}
	if clipboard != "" {
		args = append(args, "--clipboard", clipboard)
	}
	if data.MonoMode != "" {
		args = append(args, "--mono-mode", data.MonoMode)
	}
	if data.SampleRate > 0 {
		args = append(args, "--sample-rate", strconv.Itoa(data.SampleRate))
	}
	if data.Timestamps {
		args = append(args, "--timestamps")
	}
	args = append(args, "--stream-transcript="+strconv.FormatBool(data.StreamTranscript && !global.Agent))
	if data.ModelName != "" {
		args = append(args, "--model", data.ModelName)
	} else if data.ModelRef != "" {
		args = append(args, "--model", strings.TrimSuffix(strings.TrimPrefix(data.ModelRef, "ggml-"), ".bin"))
	}
	if data.HFModelID != "" {
		args = append(args, "--hf-model-id", data.HFModelID)
	}
	if data.FFmpegPath != "" {
		args = append(args, "--ffmpeg-path", data.FFmpegPath)
	}
	if input == "" {
		input = data.Input
	}
	args = append(args, input)
	return args
}

func openHistoryCommandDB(global GlobalOptions) (*sql.DB, string, *AppError) {
	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		ae := newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")
		return nil, "", &ae
	}
	db, err := history.Open(stateDir)
	if err != nil {
		ae := newError("filesystem", "HISTORY_DB_ERROR", err.Error(), false, "Check the SQLite database path or set --state-dir")
		return nil, stateDir, &ae
	}
	return db, stateDir, nil
}

func openHistoryReadOnlyCommandDB(global GlobalOptions) (*sql.DB, string, *AppError) {
	stateDir, err := ensureStateDir(global.StateDir)
	if err != nil {
		ae := newError("filesystem", "STATE_DIR_ERROR", err.Error(), false, "Set --state-dir to a writable directory")
		return nil, "", &ae
	}
	if db, err := history.Open(stateDir); err == nil {
		_ = db.Close()
	}
	db, err := history.OpenReadOnly(stateDir)
	if err != nil {
		ae := newError("filesystem", "HISTORY_DB_ERROR", err.Error(), false, "Check the SQLite database path or set --state-dir")
		return nil, stateDir, &ae
	}
	return db, stateDir, nil
}

func saveHistoryRecord(stateDir string, env Envelope) error {
	runData, ok := env.Data.(RunData)
	if !ok {
		return nil
	}
	envelopeJSON, err := history.MarshalEnvelope(env)
	if err != nil {
		return err
	}
	rec := history.Record{
		RunID:          env.RunID,
		CreatedAt:      history.RunCreatedAt(env.RunID),
		Command:        env.Command,
		Status:         env.Status,
		SessionID:      env.SessionID,
		Input:          runData.Input,
		Engine:         runData.Engine,
		ModelRef:       runData.ModelRef,
		EnvelopeJSON:   envelopeJSON,
		DurationMS:     asInt64(env.Metrics["duration_ms"]),
		FilesTotal:     asInt64(env.Metrics["files_total"]),
		FilesSucceeded: asInt64(env.Metrics["files_succeeded"]),
		FilesFailed:    asInt64(env.Metrics["files_failed"]),
		Files:          make([]history.FileRecord, 0, len(runData.Files)),
	}
	for _, file := range runData.Files {
		transcriptText := ""
		if file.Transcript != "" {
			transcriptText, _ = readOptionalTextFile(file.Transcript)
		}
		descriptionText := ""
		if file.Description != "" {
			descriptionText, _ = readOptionalTextFile(file.Description)
		}
		errorCode := ""
		if file.Error != nil {
			errorCode = file.Error.Code
		}
		rec.Files = append(rec.Files, history.FileRecord{
			File:            file.File,
			TranscriptPath:  file.Transcript,
			DescriptionPath: file.Description,
			Status:          file.Status,
			Transcript:      transcriptText,
			Description:     descriptionText,
			ErrorCode:       errorCode,
		})
	}
	return history.Save(stateDir, rec)
}

func readOptionalTextFile(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func printEnvelope(env Envelope, mode string) error {
	switch mode {
	case "text":
		return printTextEnvelope(env)
	case "json", "jsonl", "":
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		return enc.Encode(env)
	default:
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		return enc.Encode(env)
	}
}

func printCommandResult(env Envelope, args []string) error {
	if help, ok := helpText(env); ok && shouldPrintRawHelp(args) {
		_, err := fmt.Fprint(os.Stdout, help)
		return err
	}
	if ver, ok := versionText(env); ok && shouldPrintRawHelp(args) {
		_, err := fmt.Fprintln(os.Stdout, ver)
		return err
	}
	return printEnvelope(env, guessOutput(args))
}

func printTextEnvelope(env Envelope) error {
	if printHistoryTextEnvelope(env) {
		return nil
	}
	fmt.Fprintf(os.Stdout, "status: %s\n", env.Status)
	fmt.Fprintf(os.Stdout, "command: %s\n", env.Command)
	fmt.Fprintf(os.Stdout, "run_id: %s\n", env.RunID)
	if env.SessionID != "" {
		fmt.Fprintf(os.Stdout, "session_id: %s\n", env.SessionID)
	}
	if len(env.Errors) > 0 {
		for _, e := range env.Errors {
			fmt.Fprintf(os.Stdout, "error[%s/%s]: %s\n", e.Class, e.Code, e.Message)
			if e.Hint != "" {
				fmt.Fprintf(os.Stdout, "hint: %s\n", e.Hint)
			}
		}
	}
	if len(env.Warnings) > 0 {
		for _, w := range env.Warnings {
			fmt.Fprintf(os.Stdout, "warning[%s]: %s\n", w.Code, w.Message)
		}
	}
	if env.Data != nil {
		b, _ := json.MarshalIndent(env.Data, "", "  ")
		fmt.Fprintf(os.Stdout, "data:\n%s\n", string(b))
	}
	if len(env.Metrics) > 0 {
		b, _ := json.MarshalIndent(env.Metrics, "", "  ")
		fmt.Fprintf(os.Stdout, "metrics:\n%s\n", string(b))
	}
	return nil
}

func printHistoryTextEnvelope(env Envelope) bool {
	if !strings.HasPrefix(env.Command, "history.") || env.Status != "succeeded" {
		return false
	}
	data, ok := env.Data.(map[string]any)
	if !ok {
		return false
	}
	switch env.Command {
	case "history.path":
		fmt.Fprintf(os.Stdout, "State directory: %s\n", textValue(data["state_dir"]))
		fmt.Fprintf(os.Stdout, "Database: %s\n", textValue(data["database"]))
	case "history.list":
		runs, _ := data["runs"].([]history.Run)
		fmt.Fprintf(os.Stdout, "History runs: %d\n", len(runs))
		for _, run := range runs {
			printHistoryRunSummary(run)
		}
	case "history.latest", "history.show":
		if transcripts, ok := data["transcripts"].([]map[string]string); ok {
			for i, rec := range transcripts {
				if i > 0 {
					fmt.Fprintln(os.Stdout)
				}
				if rec["file"] != "" {
					fmt.Fprintf(os.Stdout, "## %s\n\n", rec["file"])
				}
				fmt.Fprintln(os.Stdout, strings.TrimSpace(rec["transcript"]))
			}
			return true
		}
		run, _ := data["run"].(history.Run)
		files, _ := data["files"].([]history.Transcription)
		printHistoryRunDetail(run, files)
	case "history.search":
		query := textValue(data["query"])
		matches, _ := data["matches"].([]history.Transcription)
		fmt.Fprintf(os.Stdout, "Search: %q\nMatches: %d\n", query, len(matches))
		for _, match := range matches {
			fmt.Fprintln(os.Stdout)
			fmt.Fprintf(os.Stdout, "%s  %s  %s\n", match.RunID, match.Status, match.File)
			if match.TranscriptPath != "" {
				fmt.Fprintf(os.Stdout, "transcript: %s\n", match.TranscriptPath)
			}
			if preview := textPreview(firstNonEmpty(match.TranscriptPreview, match.DescriptionPreview, match.Transcript, match.Description), 260); preview != "" {
				fmt.Fprintf(os.Stdout, "%s\n", preview)
			}
		}
	case "history.export":
		if content := textValue(data["content"]); content != "" {
			fmt.Fprint(os.Stdout, content)
			if !strings.HasSuffix(content, "\n") {
				fmt.Fprintln(os.Stdout)
			}
			return true
		}
		return false
	case "history.schema":
		schema, _ := data["schema"].(map[string]any)
		fmt.Fprintln(os.Stdout, "History schema")
		for _, table := range []string{"runs", "transcriptions"} {
			fmt.Fprintf(os.Stdout, "\n%s\n", table)
			cols, _ := schema[table].([]map[string]any)
			for _, col := range cols {
				flags := []string{}
				if b, _ := col["primary"].(bool); b {
					flags = append(flags, "primary")
				}
				if b, _ := col["not_null"].(bool); b {
					flags = append(flags, "not null")
				}
				suffix := ""
				if len(flags) > 0 {
					suffix = " (" + strings.Join(flags, ", ") + ")"
				}
				fmt.Fprintf(os.Stdout, "  %-18s %s%s\n", textValue(col["name"]), textValue(col["type"]), suffix)
			}
		}
	case "history.sql":
		rows, _ := data["rows"].([]map[string]any)
		printRowsTable(rows)
	default:
		return false
	}
	return true
}

func printHistoryRunSummary(run history.Run) {
	fmt.Fprintf(os.Stdout, "\n%s  %s  %s\n", run.RunID, run.Status, run.CreatedAt)
	if run.Input != "" {
		fmt.Fprintf(os.Stdout, "input: %s\n", run.Input)
	}
	if run.Engine != "" || run.ModelRef != "" {
		fmt.Fprintf(os.Stdout, "engine: %s  model: %s\n", run.Engine, run.ModelRef)
	}
	fmt.Fprintf(os.Stdout, "files: %d total, %d succeeded, %d failed  duration: %s\n",
		run.FilesTotal, run.FilesSucceeded, run.FilesFailed, formatMillis(run.DurationMS))
}

func printHistoryRunDetail(run history.Run, files []history.Transcription) {
	printHistoryRunSummary(run)
	for _, file := range files {
		fmt.Fprintln(os.Stdout)
		fmt.Fprintf(os.Stdout, "- %s  [%s]\n", file.File, file.Status)
		if file.TranscriptPath != "" {
			fmt.Fprintf(os.Stdout, "  transcript: %s\n", file.TranscriptPath)
		}
		if file.DescriptionPath != "" {
			fmt.Fprintf(os.Stdout, "  description: %s\n", file.DescriptionPath)
		}
		if file.ErrorCode != "" {
			fmt.Fprintf(os.Stdout, "  error: %s\n", file.ErrorCode)
		}
		transcriptChars := file.TranscriptChars
		if transcriptChars == 0 {
			transcriptChars = int64(len(strings.TrimSpace(file.Transcript)))
		}
		if transcriptChars > 0 {
			fmt.Fprintf(os.Stdout, "  transcript_chars: %d\n", transcriptChars)
			if preview := textPreview(firstNonEmpty(file.TranscriptPreview, file.Transcript), 320); preview != "" {
				fmt.Fprintf(os.Stdout, "  preview: %s\n", preview)
			}
		}
		descriptionChars := file.DescriptionChars
		if descriptionChars == 0 {
			descriptionChars = int64(len(strings.TrimSpace(file.Description)))
		}
		if descriptionChars > 0 {
			fmt.Fprintf(os.Stdout, "  description_chars: %d\n", descriptionChars)
		}
	}
}

func printRowsTable(rows []map[string]any) {
	if len(rows) == 0 {
		fmt.Fprintln(os.Stdout, "No rows.")
		return
	}
	cols := []string{}
	for col := range rows[0] {
		cols = append(cols, col)
	}
	sort.Strings(cols)
	widths := make([]int, len(cols))
	for i, col := range cols {
		widths[i] = len(col)
	}
	table := make([][]string, len(rows))
	for r, row := range rows {
		table[r] = make([]string, len(cols))
		for c, col := range cols {
			value := textPreview(textValue(row[col]), 80)
			table[r][c] = value
			if len(value) > widths[c] {
				widths[c] = len(value)
			}
		}
	}
	for i, col := range cols {
		if i > 0 {
			fmt.Fprint(os.Stdout, "  ")
		}
		fmt.Fprintf(os.Stdout, "%-*s", widths[i], col)
	}
	fmt.Fprintln(os.Stdout)
	for i, w := range widths {
		if i > 0 {
			fmt.Fprint(os.Stdout, "  ")
		}
		fmt.Fprint(os.Stdout, strings.Repeat("-", w))
	}
	fmt.Fprintln(os.Stdout)
	for _, row := range table {
		for i, value := range row {
			if i > 0 {
				fmt.Fprint(os.Stdout, "  ")
			}
			fmt.Fprintf(os.Stdout, "%-*s", widths[i], value)
		}
		fmt.Fprintln(os.Stdout)
	}
}

func textValue(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case fmt.Stringer:
		return t.String()
	default:
		return fmt.Sprint(t)
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func textPreview(text string, max int) string {
	s := strings.Join(strings.Fields(text), " ")
	if max <= 0 || len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return strings.TrimSpace(s[:max-3]) + "..."
}

func formatMillis(ms int64) string {
	if ms < 1000 {
		return fmt.Sprintf("%dms", ms)
	}
	d := time.Duration(ms) * time.Millisecond
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return d.Round(time.Second).String()
}

func emitJSONLEvent(command string, runID string, event string, data map[string]any) {
	rec := map[string]any{
		"schema_version": schemaVersion,
		"command":        command,
		"run_id":         runID,
		"event":          event,
		"data":           data,
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return
	}
	fmt.Fprintln(os.Stdout, string(b))
}

func downloadLabel(rawURL string) string {
	trimmed := strings.TrimSpace(rawURL)
	if trimmed == "" {
		return "artifact"
	}
	withoutQuery := strings.SplitN(trimmed, "?", 2)[0]
	base := filepath.Base(withoutQuery)
	if base == "" || base == "." || base == "/" {
		return "artifact"
	}
	return base
}

func humanBytes(n int64) string {
	if n <= 0 {
		return "0 B"
	}
	units := []string{"B", "KiB", "MiB", "GiB", "TiB"}
	size := float64(n)
	unitIdx := 0
	for size >= 1024 && unitIdx < len(units)-1 {
		size /= 1024
		unitIdx++
	}
	if unitIdx == 0 {
		return fmt.Sprintf("%d %s", n, units[unitIdx])
	}
	return fmt.Sprintf("%.1f %s", size, units[unitIdx])
}

func finishEnvelope(env *Envelope, started time.Time, total int64, succeeded int64, failed int64) {
	env.Metrics["duration_ms"] = time.Since(started).Milliseconds()
	if total > 0 {
		env.Metrics["files_total"] = total
		env.Metrics["files_succeeded"] = succeeded
		env.Metrics["files_failed"] = failed
	}
}

func guessOutput(args []string) string {
	mode, _ := outputPreference(args)
	return mode
}

func outputPreference(args []string) (string, string) {
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--agent" || a == "--agent=true" {
			return "json", "flag"
		}
		if strings.HasPrefix(a, "--output=") {
			return strings.TrimSpace(strings.TrimPrefix(a, "--output=")), "flag"
		}
		if a == "--output" && i+1 < len(args) {
			return strings.TrimSpace(args[i+1]), "flag"
		}
	}
	if mode, ok := os.LookupEnv("SCRIBY_OUTPUT"); ok {
		trimmed := strings.TrimSpace(mode)
		if trimmed != "" {
			return trimmed, "env"
		}
	}
	if envBool("SCRIBY_AGENT", false) {
		return "json", "env"
	}
	return "text", "default"
}

func hoistGlobalFlags(args []string) []string {
	if len(args) == 0 {
		return args
	}
	front := []string{}
	rest := []string{}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if isGlobalBoolFlag(arg) || isGlobalValueFlagEquals(arg) {
			front = append(front, arg)
			continue
		}
		if isGlobalValueFlag(arg) && i+1 < len(args) {
			front = append(front, arg, args[i+1])
			i++
			continue
		}
		rest = append(rest, arg)
	}
	if len(front) == 0 {
		return args
	}
	out := make([]string, 0, len(args))
	out = append(out, front...)
	out = append(out, rest...)
	return out
}

func isGlobalBoolFlag(arg string) bool {
	switch arg {
	case "--agent", "--agent=true", "--strict", "--strict=true", "--non-interactive", "--non-interactive=true", "--yes", "--yes=true":
		return true
	default:
		return false
	}
}

func isGlobalValueFlag(arg string) bool {
	switch arg {
	case "--output", "--timeout-ms", "--max-retries", "--idempotency-key", "--session-policy", "--session-id", "--state-dir":
		return true
	default:
		return false
	}
}

func isGlobalValueFlagEquals(arg string) bool {
	for _, prefix := range []string{
		"--output=",
		"--timeout-ms=",
		"--max-retries=",
		"--idempotency-key=",
		"--session-policy=",
		"--session-id=",
		"--state-dir=",
	} {
		if strings.HasPrefix(arg, prefix) {
			return true
		}
	}
	return false
}

func helpText(env Envelope) (string, bool) {
	if env.Status != "succeeded" || len(env.Errors) != 0 {
		return "", false
	}
	text, ok := env.Data.(string)
	if !ok {
		return "", false
	}
	if !strings.HasPrefix(text, "Usage: scriby ") {
		return "", false
	}
	return text, true
}

func versionText(env Envelope) (string, bool) {
	if env.Command != "version" || env.Status != "succeeded" {
		return "", false
	}
	text, ok := env.Data.(string)
	if !ok {
		return "", false
	}
	return text, true
}

func shouldPrintRawHelp(args []string) bool {
	mode, source := outputPreference(args)
	switch source {
	case "flag", "env":
		return mode == "text"
	default:
		return true
	}
}

func splitRootArgs(args []string) ([]string, []string, bool, error) {
	leading := []string{}
	help := false

	for i := 0; i < len(args); {
		arg := args[i]
		if arg == "--" {
			return leading, args[i+1:], help, nil
		}
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			return leading, args[i:], help, nil
		}
		if arg == "-V" || arg == "--version" {
			return leading, args[i:], help, nil
		}

		switch {
		case arg == "-h" || arg == "--help":
			help = true
			leading = append(leading, arg)
			i++
		case arg == "--agent" || arg == "--strict" || arg == "--non-interactive" || arg == "--yes":
			leading = append(leading, arg)
			i++
		case strings.HasPrefix(arg, "--output="),
			strings.HasPrefix(arg, "--timeout-ms="),
			strings.HasPrefix(arg, "--max-retries="),
			strings.HasPrefix(arg, "--idempotency-key="),
			strings.HasPrefix(arg, "--session-policy="),
			strings.HasPrefix(arg, "--session-id="),
			strings.HasPrefix(arg, "--state-dir="):
			leading = append(leading, arg)
			i++
		case arg == "--output",
			arg == "--timeout-ms",
			arg == "--max-retries",
			arg == "--idempotency-key",
			arg == "--session-policy",
			arg == "--session-id",
			arg == "--state-dir":
			if i+1 >= len(args) {
				return nil, nil, false, fmt.Errorf("flag needs an argument: %s", arg)
			}
			leading = append(leading, arg, args[i+1])
			i += 2
		default:
			name := strings.TrimLeft(arg, "-")
			if name == "" {
				name = arg
			}
			return nil, nil, false, fmt.Errorf("flag provided but not defined: -%s", name)
		}
	}

	return leading, nil, help, nil
}

func isValidOutputMode(mode string) bool {
	switch mode {
	case "json", "jsonl", "text":
		return true
	default:
		return false
	}
}

func binaryName(name string) string {
	if runtime.GOOS == "windows" {
		return name + ".exe"
	}
	return name
}

func runtimeAssetName() string {
	if runtime.GOOS == "windows" {
		return fmt.Sprintf("whisper-cli-%s-%s.zip", runtime.GOOS, runtime.GOARCH)
	}
	return fmt.Sprintf("whisper-cli-%s-%s.tar.gz", runtime.GOOS, runtime.GOARCH)
}

func modelFilename(name string) string {
	n := strings.TrimSpace(name)
	if n == "" {
		n = defaultModelName
	}
	n = strings.TrimPrefix(n, "ggml-")
	n = strings.TrimSuffix(n, ".bin")
	return "ggml-" + n + ".bin"
}

func newError(class string, code string, msg string, retryable bool, hint string) AppError {
	return AppError{Class: class, Code: code, Message: msg, Retryable: retryable, Hint: strings.TrimSpace(hint)}
}

func newRunID() string {
	raw := make([]byte, 4)
	_, _ = rand.Read(raw)
	return time.Now().UTC().Format("20060102-150405") + "-" + hex.EncodeToString(raw)
}

func exitFromStatus(status string) int {
	switch status {
	case "succeeded":
		return exitOK
	case "partial":
		return exitPartial
	case "failed":
		return exitRuntime
	default:
		return exitInternal
	}
}

func trimHint(s string) string {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return ""
	}
	if len(trimmed) > 220 {
		return trimmed[:220]
	}
	return trimmed
}

func envOr(key string, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	switch strings.ToLower(v) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func envInt(key string, fallback int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func sanitizeFileName(s string) string {
	if s == "" {
		return "default"
	}
	s = strings.ReplaceAll(s, string(os.PathSeparator), "_")
	s = strings.ReplaceAll(s, "..", "_")
	s = strings.ReplaceAll(s, " ", "_")
	return s
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func dirExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}

func ptrError(ae AppError) *AppError {
	return &ae
}

func asInt(v any) int {
	switch t := v.(type) {
	case int:
		return t
	case int64:
		return int(t)
	case float64:
		return int(t)
	default:
		return 0
	}
}

func asInt64(v any) int64 {
	switch t := v.(type) {
	case int:
		return int64(t)
	case int64:
		return t
	case float64:
		return int64(t)
	default:
		return 0
	}
}

func copyFile(src string, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Sync()
}

func verifyFileSHA256(path string, expected string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	actual := hex.EncodeToString(h.Sum(nil))
	want := strings.ToLower(strings.TrimSpace(expected))
	if want == "" {
		return nil
	}
	if actual != want {
		return fmt.Errorf("sha256 mismatch: expected %s, got %s", want, actual)
	}
	return nil
}

func runHelp() string {
	return `Usage: scriby run [flags] <file-or-directory> [prompt_file]

Args:
  <file-or-directory>  One media file or a directory scanned for top-level .mp4/.m4a/.mp3/.mov/.wav
  [prompt_file]        Optional prompt file for description generation

Run Flags:
  --prompt <path>           Prompt file override
  --clipboard <mode>        never|ask|always (default: ask)
  --mono-mode <mode>        left|right|average (default: average)
  --sample-rate <hz>        Sample rate for conversion (default: 16000)
  --timestamps              Include timestamps in transcript (whisper engine only)
  --engine <name>           whisper|cohere (default: whisper; cohere requires Apple Silicon + uv)
  --language <code>         Language code for the active ASR engine (default: en)
  --stream-transcript       Stream whisper stdout into transcript (default: true; whisper only)
  --model <name>            tiny|base|small|medium|large-v3 (whisper only)
  --model-url <url>         Override Whisper model download URL
  --hf-model-id <repo>      Hugging Face model ID for the cohere engine (default: CohereLabs/cohere-transcribe-03-2026)
  --python-path <path>      Python executable (unused by cohere engine, which uses uv)
  --whisper-path <path>     Use an existing whisper binary
  --whisper-url <url>       Bootstrap whisper runtime from explicit URL/archive
  --runtime-manifest-url <url>
                             Deterministic manifest URL for runtime asset resolution
  --ffmpeg-path <path>      Use an explicit ffmpeg path
  --llm-path <path>         llm CLI path (description only)
  --keep-temp               Keep intermediate WAV files

Global Flags:
  --output text|json|jsonl  (default: text; jsonl streams progress/events)
  --agent                   JSON output, non-interactive prompts, clipboard disabled for run
  --strict
  --non-interactive         Disable interactive prompts (default: false)
  --yes
  --timeout-ms <ms>
  --max-retries <n>
  --idempotency-key <key>
  --session-policy ephemeral|sticky|resume
  --session-id <id>
  --state-dir <path>

Exit Codes:
  0 success, 2 input, 3 dependency/setup, 4 runtime failure, 5 partial, 10 internal
`
}

func validateHelp() string {
	return `Usage: scriby validate [flags] <file-or-directory> [prompt_file]

Purpose:
  Validate arguments, input paths, and runtime readiness without transcribing.

Flags:
  Same flags as 'scriby run'.

Exit Codes:
  0 success, 2 input, 3 dependency/setup, 4 runtime failure, 10 internal
`
}

func doctorHelp() string {
	return `Usage: scriby doctor [flags]

Purpose:
  Perform deterministic health checks for ffmpeg, whisper runtime, whisper model availability,
  uv/mlx-audio readiness, llm optional dependency, and state directory writability.

Exit Codes:
  0 success, 2 input, 3 dependency/setup, 4 runtime failure, 10 internal
`
}

func replayHelp() string {
	return `Usage: scriby replay [flags] <run_id>

Purpose:
  Load and emit a previously saved run envelope by run_id.

Exit Codes:
  0 success, 2 input, 3 dependency/setup, 4 runtime failure, 10 internal
`
}

func retryHelp() string {
	return `Usage: scriby retry [flags] <run_id>

Purpose:
  Retry a saved transcription run. By default retries the original input; use --failed-only
  to retry only failed files from the saved run.

Flags:
  --failed-only            Retry only failed files
  --agent                  JSON output, non-interactive prompts, clipboard disabled
  --state-dir <path>       State directory containing saved run history

Exit Codes:
  0 success, 2 input, 3 dependency/setup, 4 runtime failure, 5 partial, 10 internal
`
}

func modelsHelp() string {
	return `Usage: scriby models <pull|list|prune> [flags]

Subcommands:
  pull   Download/install a whisper model into state dir
  list   List known whisper model names and installed model files
  prune  Remove one whisper model (--name) or all models (requires --yes)

Exit Codes:
  0 success, 2 input, 3 dependency/setup, 4 runtime failure, 10 internal
`
}

func historyHelp() string {
	return `Usage: scriby history <path|list|latest|show|search|export|schema|sql> [flags]

Purpose:
  Explore durable local run/transcription history stored in ~/.scriby/scriby.db.

Subcommands:
  path                 Print state directory and SQLite database path
  list                 List recent transcription runs (--limit, --status, --since)
  latest               Show the latest run summary, optionally --include-transcript or --transcript-only
  show <run_id>        Show one run summary, optionally --include-transcript or --transcript-only
  search <query>       Search transcript, description, and file path text (--since, --include-transcript)
  export <run_id>      Export one run as markdown or json
  schema               Print SQLite table/column schema
  sql <statement>      Run a read-only SELECT/WITH/PRAGMA query

Examples:
  scriby history path
  scriby history list --limit 10
  scriby history latest --transcript-only
  scriby history show 20260224-abc123
  scriby history search "customer discovery" --since 7d
  scriby history export --latest --format markdown
  scriby history schema
  scriby history sql "select run_id, status, input from runs order by created_at desc limit 5"

Exit Codes:
  0 success, 2 input, 3 dependency/setup, 4 runtime failure, 10 internal
`
}
