package history

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type Run struct {
	RunID          string `json:"run_id"`
	CreatedAt      string `json:"created_at"`
	Status         string `json:"status"`
	Input          string `json:"input,omitempty"`
	Engine         string `json:"engine,omitempty"`
	ModelRef       string `json:"model_ref,omitempty"`
	FilesTotal     int64  `json:"files_total"`
	FilesSucceeded int64  `json:"files_succeeded"`
	FilesFailed    int64  `json:"files_failed"`
	DurationMS     int64  `json:"duration_ms"`
}

type Transcription struct {
	RunID           string `json:"run_id"`
	CreatedAt       string `json:"created_at"`
	File            string `json:"file"`
	TranscriptPath  string `json:"transcript_path,omitempty"`
	DescriptionPath string `json:"description_path,omitempty"`
	Status          string `json:"status"`
	Transcript      string `json:"transcript,omitempty"`
	Description     string `json:"description,omitempty"`
	ErrorCode       string `json:"error_code,omitempty"`
}

type FileRecord struct {
	File            string
	TranscriptPath  string
	DescriptionPath string
	Status          string
	Transcript      string
	Description     string
	ErrorCode       string
}

type Record struct {
	RunID          string
	CreatedAt      string
	Command        string
	Status         string
	SessionID      string
	Input          string
	Engine         string
	ModelRef       string
	EnvelopeJSON   string
	DurationMS     int64
	FilesTotal     int64
	FilesSucceeded int64
	FilesFailed    int64
	Files          []FileRecord
}

func DBPath(stateDir string) string {
	return filepath.Join(stateDir, "scriby.db")
}

func Open(stateDir string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", DBPath(stateDir))
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec(`PRAGMA foreign_keys = ON`); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := EnsureSchema(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func OpenReadOnly(stateDir string) (*sql.DB, error) {
	if _, err := os.Stat(DBPath(stateDir)); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", readOnlyDSN(DBPath(stateDir)))
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	if _, err := db.Exec(`PRAGMA query_only = ON`); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func readOnlyDSN(path string) string {
	u := url.URL{
		Scheme: "file",
		Path:   path,
	}
	q := u.Query()
	q.Set("mode", "ro")
	u.RawQuery = q.Encode()
	return u.String()
}

func EnsureSchema(db *sql.DB) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS runs (
			run_id TEXT PRIMARY KEY,
			created_at TEXT NOT NULL,
			command TEXT NOT NULL,
			status TEXT NOT NULL,
			session_id TEXT,
			input TEXT,
			engine TEXT,
			model_ref TEXT,
			envelope_json TEXT NOT NULL,
			duration_ms INTEGER NOT NULL DEFAULT 0,
			files_total INTEGER NOT NULL DEFAULT 0,
			files_succeeded INTEGER NOT NULL DEFAULT 0,
			files_failed INTEGER NOT NULL DEFAULT 0
		)`,
		`CREATE TABLE IF NOT EXISTS transcriptions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
			created_at TEXT NOT NULL,
			file_path TEXT NOT NULL,
			transcript_path TEXT,
			description_path TEXT,
			status TEXT NOT NULL,
			transcript_text TEXT,
			description_text TEXT,
			error_code TEXT,
			UNIQUE(run_id, file_path)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)`,
		`CREATE INDEX IF NOT EXISTS idx_transcriptions_run_id ON transcriptions(run_id)`,
		`CREATE INDEX IF NOT EXISTS idx_transcriptions_file_path ON transcriptions(file_path)`,
	}
	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return err
		}
	}
	return nil
}

func Save(stateDir string, rec Record) error {
	db, err := Open(stateDir)
	if err != nil {
		return err
	}
	defer db.Close()

	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if rec.CreatedAt == "" {
		rec.CreatedAt = RunCreatedAt(rec.RunID)
	}
	if _, err := tx.Exec(
		`INSERT INTO runs (
			run_id, created_at, command, status, session_id, input, engine, model_ref,
			envelope_json, duration_ms, files_total, files_succeeded, files_failed
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(run_id) DO UPDATE SET
			created_at=excluded.created_at,
			command=excluded.command,
			status=excluded.status,
			session_id=excluded.session_id,
			input=excluded.input,
			engine=excluded.engine,
			model_ref=excluded.model_ref,
			envelope_json=excluded.envelope_json,
			duration_ms=excluded.duration_ms,
			files_total=excluded.files_total,
			files_succeeded=excluded.files_succeeded,
			files_failed=excluded.files_failed`,
		rec.RunID,
		rec.CreatedAt,
		rec.Command,
		rec.Status,
		rec.SessionID,
		rec.Input,
		rec.Engine,
		rec.ModelRef,
		rec.EnvelopeJSON,
		rec.DurationMS,
		rec.FilesTotal,
		rec.FilesSucceeded,
		rec.FilesFailed,
	); err != nil {
		return err
	}

	for _, file := range rec.Files {
		if _, err := tx.Exec(
			`INSERT INTO transcriptions (
				run_id, created_at, file_path, transcript_path, description_path, status,
				transcript_text, description_text, error_code
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(run_id, file_path) DO UPDATE SET
				created_at=excluded.created_at,
				transcript_path=excluded.transcript_path,
				description_path=excluded.description_path,
				status=excluded.status,
				transcript_text=excluded.transcript_text,
				description_text=excluded.description_text,
				error_code=excluded.error_code`,
			rec.RunID,
			rec.CreatedAt,
			file.File,
			file.TranscriptPath,
			file.DescriptionPath,
			file.Status,
			file.Transcript,
			file.Description,
			file.ErrorCode,
		); err != nil {
			return err
		}
	}

	return tx.Commit()
}

func EnvelopeJSON(db *sql.DB, runID string) (string, error) {
	var s string
	err := db.QueryRow(`SELECT envelope_json FROM runs WHERE run_id = ?`, runID).Scan(&s)
	return s, err
}

func ListRuns(db *sql.DB, limit int, status string, since *time.Time) ([]Run, error) {
	query := `SELECT run_id, created_at, status, COALESCE(input, ''), COALESCE(engine, ''),
		COALESCE(model_ref, ''), files_total, files_succeeded, files_failed, duration_ms
		FROM runs`
	args := []any{}
	clauses := []string{}
	if status != "" {
		clauses = append(clauses, `status = ?`)
		args = append(args, status)
	}
	if since != nil {
		clauses = append(clauses, `created_at >= ?`)
		args = append(args, since.UTC().Format(time.RFC3339))
	}
	if len(clauses) > 0 {
		query += ` WHERE ` + strings.Join(clauses, ` AND `)
	}
	query += ` ORDER BY created_at DESC LIMIT ?`
	args = append(args, limit)

	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanRuns(rows)
}

func LatestRunID(db *sql.DB) (string, error) {
	var runID string
	err := db.QueryRow(`SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1`).Scan(&runID)
	return runID, err
}

func GetRun(db *sql.DB, runID string, includeTranscript bool) (Run, []Transcription, string, error) {
	row := db.QueryRow(`SELECT run_id, created_at, status, COALESCE(input, ''), COALESCE(engine, ''),
		COALESCE(model_ref, ''), files_total, files_succeeded, files_failed, duration_ms, envelope_json
		FROM runs WHERE run_id = ?`, runID)
	var run Run
	var envelopeJSON string
	if err := row.Scan(&run.RunID, &run.CreatedAt, &run.Status, &run.Input, &run.Engine, &run.ModelRef, &run.FilesTotal, &run.FilesSucceeded, &run.FilesFailed, &run.DurationMS, &envelopeJSON); err != nil {
		return Run{}, nil, "", err
	}

	selectText := "'' AS transcript_text, '' AS description_text"
	if includeTranscript {
		selectText = "COALESCE(transcript_text, '') AS transcript_text, COALESCE(description_text, '') AS description_text"
	}
	rows, err := db.Query(`SELECT run_id, created_at, file_path, COALESCE(transcript_path, ''),
		COALESCE(description_path, ''), status, `+selectText+`, COALESCE(error_code, '')
		FROM transcriptions WHERE run_id = ? ORDER BY id ASC`, runID)
	if err != nil {
		return Run{}, nil, "", err
	}
	defer rows.Close()
	files, err := scanTranscriptions(rows)
	if err != nil {
		return Run{}, nil, "", err
	}
	return run, files, envelopeJSON, nil
}

func Search(db *sql.DB, query string, limit int, since *time.Time) ([]Transcription, error) {
	like := "%" + escapeLike(query) + "%"
	where := `WHERE (transcript_text LIKE ? ESCAPE '\' OR description_text LIKE ? ESCAPE '\' OR file_path LIKE ? ESCAPE '\')`
	args := []any{like, like, like}
	if since != nil {
		where += ` AND created_at >= ?`
		args = append(args, since.UTC().Format(time.RFC3339))
	}
	args = append(args, limit)
	rows, err := db.Query(`SELECT run_id, created_at, file_path, COALESCE(transcript_path, ''),
		COALESCE(description_path, ''), status, COALESCE(transcript_text, ''), COALESCE(description_text, ''),
		COALESCE(error_code, '')
		FROM transcriptions
		`+where+`
		ORDER BY created_at DESC
		LIMIT ?`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanTranscriptions(rows)
}

func Schema(db *sql.DB) (map[string]any, error) {
	tables := []string{"runs", "transcriptions"}
	out := map[string]any{}
	for _, table := range tables {
		rows, err := db.Query(`PRAGMA table_info(` + table + `)`)
		if err != nil {
			return nil, err
		}
		cols := []map[string]any{}
		for rows.Next() {
			var cid int
			var name string
			var typ string
			var notNull int
			var defaultValue any
			var pk int
			if err := rows.Scan(&cid, &name, &typ, &notNull, &defaultValue, &pk); err != nil {
				rows.Close()
				return nil, err
			}
			cols = append(cols, map[string]any{
				"name":     name,
				"type":     typ,
				"not_null": notNull == 1,
				"primary":  pk > 0,
			})
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
		out[table] = cols
	}
	return out, nil
}

func TranscriptsOnly(files []Transcription) []map[string]string {
	out := []map[string]string{}
	for _, file := range files {
		if strings.TrimSpace(file.Transcript) == "" {
			continue
		}
		out = append(out, map[string]string{
			"file":       file.File,
			"transcript": file.Transcript,
		})
	}
	return out
}

func ExportMarkdown(run Run, files []Transcription) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Scriby Run %s\n\n", run.RunID)
	fmt.Fprintf(&b, "- status: %s\n", run.Status)
	if run.Input != "" {
		fmt.Fprintf(&b, "- input: %s\n", run.Input)
	}
	if run.Engine != "" {
		fmt.Fprintf(&b, "- engine: %s\n", run.Engine)
	}
	if run.CreatedAt != "" {
		fmt.Fprintf(&b, "- created_at: %s\n", run.CreatedAt)
	}
	for _, file := range files {
		fmt.Fprintf(&b, "\n## %s\n\n", file.File)
		if file.Status != "" {
			fmt.Fprintf(&b, "- status: %s\n", file.Status)
		}
		if file.TranscriptPath != "" {
			fmt.Fprintf(&b, "- transcript_path: %s\n", file.TranscriptPath)
		}
		if strings.TrimSpace(file.Transcript) != "" {
			fmt.Fprintf(&b, "\n### Transcript\n\n%s\n", strings.TrimSpace(file.Transcript))
		}
		if strings.TrimSpace(file.Description) != "" {
			fmt.Fprintf(&b, "\n### Description\n\n%s\n", strings.TrimSpace(file.Description))
		}
	}
	return b.String()
}

func ParseSince(raw string, now time.Time) (*time.Time, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return nil, nil
	}
	if strings.HasSuffix(s, "d") {
		n, err := strconv.Atoi(strings.TrimSuffix(s, "d"))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("invalid day duration: %s", raw)
		}
		t := now.Add(-time.Duration(n) * 24 * time.Hour).UTC()
		return &t, nil
	}
	if d, err := time.ParseDuration(s); err == nil && d > 0 {
		t := now.Add(-d).UTC()
		return &t, nil
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		utc := t.UTC()
		return &utc, nil
	}
	if t, err := time.Parse("2006-01-02", s); err == nil {
		utc := t.UTC()
		return &utc, nil
	}
	return nil, fmt.Errorf("invalid since value: %s", raw)
}

func QuerySQL(db *sql.DB, query string) ([]map[string]any, error) {
	rows, err := db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return nil, err
	}
	results := []map[string]any{}
	for rows.Next() {
		values := make([]any, len(cols))
		ptrs := make([]any, len(cols))
		for i := range values {
			ptrs[i] = &values[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, err
		}
		row := map[string]any{}
		for i, col := range cols {
			switch v := values[i].(type) {
			case []byte:
				row[col] = string(v)
			default:
				row[col] = v
			}
		}
		results = append(results, row)
	}
	return results, rows.Err()
}

func LooksReadOnlySQL(query string) bool {
	q := strings.TrimSpace(query)
	q = strings.TrimSuffix(q, ";")
	if strings.Contains(q, ";") {
		return false
	}
	lower := strings.ToLower(strings.TrimSpace(q))
	return strings.HasPrefix(lower, "select ") || strings.HasPrefix(lower, "with ") || strings.HasPrefix(lower, "pragma ")
}

func RunCreatedAt(runID string) string {
	if len(runID) >= len("20060102-150405") {
		if t, err := time.Parse("20060102-150405", runID[:15]); err == nil {
			return t.UTC().Format(time.RFC3339)
		}
	}
	return time.Now().UTC().Format(time.RFC3339)
}

func scanRuns(rows *sql.Rows) ([]Run, error) {
	runs := []Run{}
	for rows.Next() {
		var run Run
		if err := rows.Scan(&run.RunID, &run.CreatedAt, &run.Status, &run.Input, &run.Engine, &run.ModelRef, &run.FilesTotal, &run.FilesSucceeded, &run.FilesFailed, &run.DurationMS); err != nil {
			return nil, err
		}
		runs = append(runs, run)
	}
	return runs, rows.Err()
}

func scanTranscriptions(rows *sql.Rows) ([]Transcription, error) {
	files := []Transcription{}
	for rows.Next() {
		var file Transcription
		if err := rows.Scan(&file.RunID, &file.CreatedAt, &file.File, &file.TranscriptPath, &file.DescriptionPath, &file.Status, &file.Transcript, &file.Description, &file.ErrorCode); err != nil {
			return nil, err
		}
		files = append(files, file)
	}
	return files, rows.Err()
}

func escapeLike(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

func MarshalEnvelope(env any) (string, error) {
	b, err := json.Marshal(env)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
