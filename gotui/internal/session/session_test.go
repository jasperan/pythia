package session

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestServerArgsMatchesServicesPy pins the argv against the one the Python TUI
// itself uses to launch the API server (src/pythia/services.py:320). If the Go
// front-end started the server differently it would not be "an additional way to
// run pythia" -- it would be a second, subtly different server.
func TestServerArgsMatchesServicesPy(t *testing.T) {
	got := ServerArgs("/p/.venv/bin/python", "127.0.0.1", 8900)
	want := []string{"/p/.venv/bin/python", "-m", "pythia", "serve",
		"--host", "127.0.0.1", "--port", "8900"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ServerArgs = %v\nwant %v", got, want)
	}
}

// TestBaseURLNormalizesWildcardBind mirrors app.py:_build_api_base: a server
// bound to 0.0.0.0 is not reachable at 0.0.0.0 from a client.
func TestBaseURLNormalizesWildcardBind(t *testing.T) {
	if got := BaseURL("0.0.0.0", 8900); got != "http://127.0.0.1:8900" {
		t.Errorf("BaseURL(0.0.0.0) = %q, want http://127.0.0.1:8900", got)
	}
	if got := BaseURL("", 8900); got != "http://127.0.0.1:8900" {
		t.Errorf("BaseURL(\"\") = %q", got)
	}
	if got := BaseURL("pythia.internal", 9000); got != "http://pythia.internal:9000" {
		t.Errorf("BaseURL(host) = %q", got)
	}
}

func TestNormalizeFillsEveryUnsetField(t *testing.T) {
	got := Settings{}.Normalize()
	if got.Host != DefaultHost || got.Port != DefaultPort || got.ConfigPath != DefaultConfig {
		t.Errorf("Normalize() = %+v, want the documented defaults", got)
	}
	if got.BaseURL != "http://127.0.0.1:8900" {
		t.Errorf("BaseURL = %q, want it derived from host and port", got.BaseURL)
	}
	// An explicit BaseURL must survive normalization: --base-url could point at a
	// server behind a proxy.
	explicit := Settings{BaseURL: "https://pythia.example"}.Normalize()
	if explicit.BaseURL != "https://pythia.example" {
		t.Errorf("an explicit BaseURL was overwritten with %q", explicit.BaseURL)
	}
}

func TestValidatePort(t *testing.T) {
	for _, valid := range []string{"1", "8900", "65535"} {
		if err := ValidatePort(valid); err != nil {
			t.Errorf("ValidatePort(%q) = %v, want nil", valid, err)
		}
	}
	for _, invalid := range []string{"", "  ", "0", "65536", "-1", "abc", "89 00"} {
		if err := ValidatePort(invalid); err == nil {
			t.Errorf("ValidatePort(%q) = nil, want an error", invalid)
		}
	}
}

// TestFindProjectRootRequiresAllMarkers is the guard against attaching to an
// unrelated directory that merely has a pyproject.toml.
func TestFindProjectRootRequiresAllMarkers(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "pyproject.toml"), "[project]\nname='pythia'\n")
	mustWrite(t, filepath.Join(root, "pythia.yaml"), "server: {port: 8900}\n")

	// Missing src/pythia/__main__.py: not yet a usable checkout.
	if got := FindProjectRoot(root); got != "" {
		t.Errorf("FindProjectRoot found %q without src/pythia/__main__.py", got)
	}

	mustWrite(t, filepath.Join(root, "src", "pythia", "__main__.py"), "")

	// Found from a nested directory, which is the normal case: the user may be
	// anywhere inside the checkout.
	nested := filepath.Join(root, "src", "pythia", "server")
	if got := FindProjectRoot(nested); got != root {
		t.Errorf("FindProjectRoot(%q) = %q, want %q", nested, got, root)
	}
	if got := FindProjectRoot(t.TempDir()); got != "" {
		t.Errorf("FindProjectRoot found %q in an unrelated directory", got)
	}
}

func TestResolveConfigPath(t *testing.T) {
	root := "/project"
	if got := ResolveConfigPath(root, ""); got != filepath.Join(root, DefaultConfig) {
		t.Errorf("ResolveConfigPath(empty) = %q", got)
	}
	if got := ResolveConfigPath(root, "custom.yaml"); got != filepath.Join(root, "custom.yaml") {
		t.Errorf("ResolveConfigPath(relative) = %q", got)
	}
	abs := filepath.Join(string(os.PathSeparator), "etc", "pythia.yaml")
	if got := ResolveConfigPath(root, abs); got != abs {
		t.Errorf("ResolveConfigPath(absolute) = %q, want %q", got, abs)
	}
}

// TestPythonCandidatesPrefersTheProjectsVenv pins the resolution order.
func TestPythonCandidatesPrefersTheProjectsVenv(t *testing.T) {
	t.Setenv(EnvPython, "")
	candidates := PythonCandidates("/project")
	if len(candidates) < 3 {
		t.Fatalf("expected the venv candidates and bare fallbacks, got %v", candidates)
	}
	if candidates[0] != filepath.Join("/project", ".venv", "bin", "python") {
		t.Errorf("first candidate = %q, want the project venv", candidates[0])
	}
	if candidates[len(candidates)-2] != "python3" || candidates[len(candidates)-1] != "python" {
		t.Errorf("candidates must end with python3 then python, got %v", candidates)
	}

	t.Setenv(EnvPython, "/custom/python")
	overridden := PythonCandidates("/project")
	if overridden[0] != "/custom/python" {
		t.Errorf("override was ignored: %v", overridden)
	}
}

// TestSaveLoadRoundTrip also checks the file is private: settings name paths and
// a model, so they are not world-readable.
func TestSaveLoadRoundTrip(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	if err := Save(Settings{Host: "0.0.0.0", Port: 9100, Model: "qwen3.5:9b"}); err != nil {
		t.Fatalf("Save returned %v", err)
	}
	path, err := Path()
	if err != nil {
		t.Fatalf("Path returned %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("settings file missing: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("settings file mode = %o, want 600", perm)
	}

	loaded, err := Load()
	if err != nil {
		t.Fatalf("Load returned %v", err)
	}
	if loaded.Host != "0.0.0.0" || loaded.Port != 9100 || loaded.Model != "qwen3.5:9b" {
		t.Errorf("round-trip lost values: %+v", loaded)
	}
}

func TestLoadReturnsDefaultsWhenMissing(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	loaded, err := Load()
	if err != nil {
		t.Fatalf("Load on a fresh machine returned %v, want defaults", err)
	}
	if loaded.Port != DefaultPort {
		t.Errorf("port = %d, want %d", loaded.Port, DefaultPort)
	}
}

// TestLoadRejectsCorruptSettingsFile ensures a damaged file is reported rather
// than silently replaced, so a user does not lose a config path without notice.
func TestLoadRejectsCorruptSettingsFile(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	path, err := Path()
	if err != nil {
		t.Fatal(err)
	}
	mustWrite(t, path, "{not json")
	if _, err := Load(); err == nil {
		t.Error("Load accepted a corrupt settings file")
	} else if !strings.Contains(err.Error(), path) {
		t.Errorf("error %v does not name the offending file", err)
	}
}

func TestPortInUseAndWaitForPort(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port

	if !PortInUse("127.0.0.1", port) {
		t.Error("PortInUse said a bound port was free")
	}
	if err := WaitForPort(context.Background(), "127.0.0.1", port, time.Second); err != nil {
		t.Errorf("WaitForPort on a bound port = %v, want nil", err)
	}

	// An unused port: both must report absence, and WaitForPort must respect its
	// deadline instead of hanging.
	free := freePort(t)
	if PortInUse("127.0.0.1", free) {
		t.Error("PortInUse said a free port was bound")
	}
	start := time.Now()
	if err := WaitForPort(context.Background(), "127.0.0.1", free, 300*time.Millisecond); err == nil {
		t.Error("WaitForPort succeeded on a port nothing is listening on")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("WaitForPort took %s; it must honour its timeout", elapsed)
	}
}

// TestWaitForPortNormalizesWildcardHost mirrors BaseURL: a server started on
// 0.0.0.0 must be reachable at 127.0.0.1 for the wait to ever succeed.
func TestWaitForPortNormalizesWildcardHost(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port

	if err := WaitForPort(context.Background(), "0.0.0.0", port, 2*time.Second); err != nil {
		t.Errorf("WaitForPort(0.0.0.0) = %v, want nil", err)
	}
}

// TestWaitForPortHonoursContextCancel keeps Ctrl+C responsive while attaching.
func TestWaitForPortHonoursContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := WaitForPort(ctx, "127.0.0.1", freePort(t), 10*time.Second); err == nil {
		t.Error("WaitForPort ignored a cancelled context")
	}
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	return port
}
