package state

import (
	"errors"
	"io"
	"os"
	"path/filepath"
)

func Ensure(explicit string) (string, error) {
	if explicit != "" {
		if err := os.MkdirAll(explicit, 0o755); err != nil {
			return "", err
		}
		ap, err := filepath.Abs(explicit)
		if err != nil {
			return explicit, nil
		}
		return ap, nil
	}

	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return "", errors.New("unable to determine state dir")
	}
	stateDir := filepath.Join(home, ".scriby")
	if legacy, ok := LegacyDir(); ok && dirExists(legacy) {
		if err := copyDirContents(legacy, stateDir); err != nil {
			return "", err
		}
	}
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		return "", err
	}
	return stateDir, nil
}

func LegacyDir() (string, bool) {
	base, err := os.UserCacheDir()
	if err != nil || base == "" {
		return "", false
	}
	return filepath.Join(base, "scriby"), true
}

func dirExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func copyDirContents(src string, dst string) error {
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return os.MkdirAll(dst, 0o755)
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		if fileExists(target) {
			return nil
		}
		return copyFile(path, target)
	})
}

func copyFile(src string, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
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
