// Package session holds the pythia-tui settings file, project-root detection and
// the optional launch of the project's own API server.
//
// The server is started exactly the way src/pythia/services.py:320 starts it --
// `<python> -m pythia serve --host H --port P`, from the project root, with
// PYTHIA_CONFIG pointing at the resolved pythia.yaml -- so a server started by
// the Go TUI is the same server the Python TUI would have started. Nothing about
// the service lifecycle is reimplemented.
package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Settings are the values the TUI remembers between runs.
type Settings struct {
	BaseURL      string `json:"base_url"`
	Host         string `json:"host"`
	Port         int    `json:"port"`
	Model        string `json:"model"`
	ConfigPath   string `json:"config_path"`
	LaunchServer bool   `json:"launch_server"`
}

// Defaults match pythia.yaml and src/pythia/config.py.
const (
	DefaultHost   = "127.0.0.1"
	DefaultPort   = 8900
	DefaultConfig = "pythia.yaml"
)

// EnvPython lets a user name the interpreter that has pythia installed.
const EnvPython = "PYTHIA_TUI_PYTHON"

// Default returns the settings used when nothing is saved.
func Default() Settings {
	return Settings{Host: DefaultHost, Port: DefaultPort, ConfigPath: DefaultConfig}
}

// Normalize fills in every unset field and derives BaseURL from Host/Port.
func (s Settings) Normalize() Settings {
	if s.Host == "" {
		s.Host = DefaultHost
	}
	if s.Port == 0 {
		s.Port = DefaultPort
	}
	if strings.TrimSpace(s.ConfigPath) == "" {
		s.ConfigPath = DefaultConfig
	}
	if strings.TrimSpace(s.BaseURL) == "" {
		s.BaseURL = BaseURL(s.Host, s.Port)
	}
	return s
}

// BaseURL normalizes app-facing API URLs so 0.0.0.0 becomes reachable
// localhost, exactly as src/pythia/tui/app.py:_build_api_base does.
func BaseURL(host string, port int) string {
	apiHost := host
	if apiHost == "" || apiHost == "0.0.0.0" {
		apiHost = "127.0.0.1"
	}
	return fmt.Sprintf("http://%s:%d", apiHost, port)
}

// Path returns the settings file location.
func Path() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "pythia-tui", "settings.json"), nil
}

// Load reads the settings file, returning defaults when it does not exist.
func Load() (Settings, error) {
	path, err := Path()
	if err != nil {
		return Default(), err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Default(), nil
		}
		return Default(), err
	}
	loaded := Default()
	if err := json.Unmarshal(data, &loaded); err != nil {
		return Default(), fmt.Errorf("settings file %s is not valid JSON: %w", path, err)
	}
	return loaded.Normalize(), nil
}

// Save writes the settings file with 0600 permissions.
//
// The file can name a config path and a model but never a credential, because no
// credential is stored: the Oracle password stays in pythia.yaml and the
// environment, where the service already reads it.
func Save(s Settings) error {
	path, err := Path()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(s.Normalize(), "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(encoded, '\n'), 0o600)
}

// ValidatePort rejects a port the server could not bind.
func ValidatePort(raw string) error {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return errors.New("port is empty")
	}
	port, err := strconv.Atoi(trimmed)
	if err != nil {
		return fmt.Errorf("port must be a number, got %q", raw)
	}
	if port < 1 || port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535, got %d", port)
	}
	return nil
}

// FindProjectRoot walks up from start looking for a pythia checkout.
//
// Both markers are required because starting the service needs both: the
// pyproject.toml proves it is the project, and pythia.yaml is what the server
// loads.
func FindProjectRoot(start string) string {
	dir := start
	if dir == "" {
		if wd, err := os.Getwd(); err == nil {
			dir = wd
		}
	}
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	for {
		if isProjectRoot(dir) {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func isProjectRoot(dir string) bool {
	if dir == "" {
		return false
	}
	for _, marker := range []string{"pyproject.toml", "pythia.yaml"} {
		if _, err := os.Stat(filepath.Join(dir, marker)); err != nil {
			return false
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "src", "pythia", "__main__.py")); err != nil {
		return false
	}
	return true
}

// ResolveConfigPath returns the absolute pythia.yaml path for a project root.
func ResolveConfigPath(root, configured string) string {
	if strings.TrimSpace(configured) == "" {
		configured = DefaultConfig
	}
	if filepath.IsAbs(configured) {
		return configured
	}
	return filepath.Join(root, configured)
}

// PythonCandidates lists the interpreters to try, most specific first.
//
// The project's own .venv is preferred over a bare "python3" so the TUI starts
// the server from the environment that actually has pythia installed, which is
// the same interpreter `pythia serve` would use.
func PythonCandidates(root string) []string {
	var candidates []string
	if override := strings.TrimSpace(os.Getenv(EnvPython)); override != "" {
		candidates = append(candidates, override)
	}
	if root != "" {
		candidates = append(candidates,
			filepath.Join(root, ".venv", "bin", "python"),
			filepath.Join(root, ".venv", "Scripts", "python.exe"),
		)
	}
	return append(candidates, "python3", "python")
}

// ServerArgs builds the argv for the project's own server, mirroring
// services.py:320.
func ServerArgs(python, host string, port int) []string {
	return []string{python, "-m", "pythia", "serve", "--host", host, "--port", strconv.Itoa(port)}
}

// Server is a running API server started by this process.
type Server struct {
	cmd  *exec.Cmd
	done chan struct{}
}

// LaunchServer starts `python -m pythia serve` for a project root.
func LaunchServer(ctx context.Context, root string, settings Settings, out io.Writer) (*Server, error) {
	if root == "" {
		return nil, errors.New("cannot start the service: the pythia project root was not found " +
			"(looked for pyproject.toml, pythia.yaml and src/pythia/__main__.py)")
	}
	python, err := findPython(root)
	if err != nil {
		return nil, err
	}
	settings = settings.Normalize()
	configPath := ResolveConfigPath(root, settings.ConfigPath)

	arguments := ServerArgs(python, settings.Host, settings.Port)
	cmd := exec.CommandContext(ctx, arguments[0], arguments[1:]...)
	cmd.Dir = root
	// PYTHIA_CONFIG is how services.py:318 points the server at the config, and
	// cli.py's resolve_config_path reads it back.
	cmd.Env = append(environWithout("PYTHIA_CONFIG"), "PYTHIA_CONFIG="+configPath)
	cmd.Stdout = out
	cmd.Stderr = out
	// Own process group, so Stop kills the uvicorn children too.
	cmd.SysProcAttr = processGroupAttr()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %s: %w", strings.Join(arguments, " "), err)
	}
	server := &Server{cmd: cmd, done: make(chan struct{})}
	go func() {
		_ = cmd.Wait()
		close(server.done)
	}()
	return server, nil
}

// Stop terminates the server and everything it spawned.
func (s *Server) Stop() {
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return
	}
	terminateProcessGroup(s.cmd)
	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
	}
}

// WaitForPort polls until something accepts connections on host:port.
func WaitForPort(ctx context.Context, host string, port int, timeout time.Duration) error {
	dialHost := host
	if dialHost == "" || dialHost == "0.0.0.0" {
		dialHost = "127.0.0.1"
	}
	address := net.JoinHostPort(dialHost, strconv.Itoa(port))
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		conn, err := net.DialTimeout("tcp", address, time.Second)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		lastErr = err
		time.Sleep(250 * time.Millisecond)
	}
	return fmt.Errorf("nothing is listening on %s after %s: %w", address, timeout, lastErr)
}

// PortInUse reports whether something already answers on host:port, which means
// the TUI should attach to the running server instead of starting a second one.
func PortInUse(host string, port int) bool {
	dialHost := host
	if dialHost == "" || dialHost == "0.0.0.0" {
		dialHost = "127.0.0.1"
	}
	conn, err := net.DialTimeout("tcp", net.JoinHostPort(dialHost, strconv.Itoa(port)), time.Second)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

func findPython(root string) (string, error) {
	var tried []string
	for _, candidate := range PythonCandidates(root) {
		tried = append(tried, candidate)
		if strings.ContainsRune(candidate, os.PathSeparator) {
			if info, err := os.Stat(candidate); err != nil || info.IsDir() {
				continue
			}
			return candidate, nil
		}
		if resolved, err := exec.LookPath(candidate); err == nil {
			return resolved, nil
		}
	}
	return "", fmt.Errorf("no python interpreter found (tried %s); the project's .venv should "+
		"provide one, or set %s", strings.Join(tried, ", "), EnvPython)
}

func environWithout(key string) []string {
	prefix := key + "="
	source := os.Environ()
	out := make([]string, 0, len(source)+1)
	for _, kv := range source {
		if strings.HasPrefix(kv, prefix) {
			continue
		}
		out = append(out, kv)
	}
	return out
}
