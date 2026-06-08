package clipboard

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

type Command struct {
	Path string
	Args []string
}

func CopyFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	return CopyText(string(content))
}

func CopyText(text string) error {
	cmdInfo, err := CommandFor(runtime.GOOS, os.Getenv, exec.LookPath)
	if err != nil {
		return err
	}
	cmd := exec.Command(cmdInfo.Path, cmdInfo.Args...)
	cmd.Stdin = strings.NewReader(text)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if hint := trimHint(stderr.String()); hint != "" {
			return fmt.Errorf("%w: %s", err, hint)
		}
		return err
	}
	return nil
}

func CommandFor(goos string, getenv func(string) string, lookPath func(string) (string, error)) (Command, error) {
	switch goos {
	case "darwin":
		if path, err := lookPath("pbcopy"); err == nil {
			return Command{Path: path}, nil
		}
		return Command{}, fmt.Errorf("clipboard unavailable: pbcopy not found")
	case "windows":
		if path, err := lookPath("clip"); err == nil {
			return Command{Path: path}, nil
		}
		if path, err := lookPath("clip.exe"); err == nil {
			return Command{Path: path}, nil
		}
		if path, err := lookPath("powershell"); err == nil {
			return Command{Path: path, Args: []string{"-NoProfile", "-Command", "$text = [Console]::In.ReadToEnd(); Set-Clipboard -Value $text"}}, nil
		}
		if path, err := lookPath("powershell.exe"); err == nil {
			return Command{Path: path, Args: []string{"-NoProfile", "-Command", "$text = [Console]::In.ReadToEnd(); Set-Clipboard -Value $text"}}, nil
		}
		return Command{}, fmt.Errorf("clipboard unavailable: clip.exe or powershell not found")
	default:
		if getenv("WAYLAND_DISPLAY") != "" || strings.EqualFold(getenv("XDG_SESSION_TYPE"), "wayland") {
			if path, err := lookPath("wl-copy"); err == nil {
				return Command{Path: path}, nil
			}
		}
		if path, err := lookPath("xclip"); err == nil {
			return Command{Path: path, Args: []string{"-selection", "clipboard"}}, nil
		}
		if path, err := lookPath("xsel"); err == nil {
			return Command{Path: path, Args: []string{"--clipboard", "--input"}}, nil
		}
		if path, err := lookPath("wl-copy"); err == nil {
			return Command{Path: path}, nil
		}
		return Command{}, fmt.Errorf("clipboard unavailable: install wl-copy, xclip, or xsel")
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
