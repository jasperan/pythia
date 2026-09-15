package tui

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jasperan/pythia/gotui/internal/session"
)

// This file covers the accessible (screen-reader) path and the validation
// helpers that path depends on.
//
// Why it matters here: huh's accessible prompt runs a field's validator on the
// raw line and only afterwards substitutes the field's default, and it never
// prints that default. A pre-filled field whose validator rejects "" therefore
// re-prompts on every bare Enter, so a screen-reader user cannot accept a value
// they cannot see -- which for this tool means they cannot get past the
// connection form at all, because the host, port and config path are all seeded.

var errBlank = errors.New("blank answer rejected")

// TestValidateDefaultedAcceptsBlank is the unit half of the fix.
func TestValidateDefaultedAcceptsBlank(t *testing.T) {
	inner := func(s string) error {
		if strings.TrimSpace(s) == "" {
			return errBlank
		}
		if s == "bad" {
			return errors.New("not usable")
		}
		return nil
	}
	wrapped := ValidateDefaulted(inner)

	for _, input := range []string{"", "   ", "\t"} {
		if err := wrapped(input); err != nil {
			t.Errorf("ValidateDefaulted(inner)(%q) = %v, want nil", input, err)
		}
	}
	if err := wrapped("bad"); err == nil {
		t.Error(`ValidateDefaulted(inner)("bad") = nil, want the inner validator's error`)
	}
	if err := wrapped("127.0.0.1"); err != nil {
		t.Errorf(`ValidateDefaulted(inner)("127.0.0.1") = %v, want nil`, err)
	}
}

// TestValidateDefaultedValueOnlyRelaxesWhenSomethingIsKept is the guard: the
// host and config path are seeded from settings and pythia.yaml, and either may
// be absent.
func TestValidateDefaultedValueOnlyRelaxesWhenSomethingIsKept(t *testing.T) {
	inner := func(s string) error {
		if strings.TrimSpace(s) == "" {
			return errBlank
		}
		return nil
	}

	if err := ValidateDefaultedValue("", inner)(""); err == nil {
		t.Error("an empty seed accepted a blank answer; the required field was weakened")
	}
	if err := ValidateDefaultedValue("   ", inner)(""); err == nil {
		t.Error("a whitespace-only seed accepted a blank answer")
	}
	if err := ValidateDefaultedValue("127.0.0.1", inner)(""); err != nil {
		t.Errorf("a seeded host rejected a blank answer: %v", err)
	}
}

// TestValidateDefaultedPortStillRejectsGarbage: relaxing blanks must not disable
// the real validation, or a bad port would reach the server as a 422.
func TestValidateDefaultedPortStillRejectsGarbage(t *testing.T) {
	validator := ValidateDefaultedValue("8900", session.ValidatePort)
	if err := validator(""); err != nil {
		t.Errorf("a blank port should keep the seeded 8900, got %v", err)
	}
	for _, bad := range []string{"0", "70000", "abc"} {
		if err := validator(bad); err == nil {
			t.Errorf("port %q was accepted", bad)
		}
	}
}

// runConnectionFormAccessible drives the real connection form through huh's
// accessible path with scripted input.
func runConnectionFormAccessible(t *testing.T, answers *ConnectionAnswers, input string) string {
	t.Helper()
	var out bytes.Buffer
	form := ConnectionForm(answers).
		WithAccessible(true).
		WithInput(strings.NewReader(input)).
		WithOutput(&out)

	done := make(chan error, 1)
	go func() { done <- form.Run() }()

	select {
	case err := <-done:
		if err != nil {
			t.Logf("form.Run returned %v (a field may need a tty; huh reports this too)", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatalf("the accessible form did not finish within 15s; output so far:\n%s", out.String())
	}
	return out.String()
}

// TestAccessibleBlankAnswerKeepsTheSeededHost is the regression test proper.
// With the documented default seed, a bare Enter must keep it instead of looping
// on "input cannot be empty".
func TestAccessibleBlankAnswerKeepsTheSeededHost(t *testing.T) {
	answers := ConnectionDefaults(session.Default(), "qwen3.5:9b")
	seededHost := answers.Host
	if seededHost == "" {
		t.Fatal("the host is not seeded; the test would prove nothing")
	}

	out := runConnectionFormAccessible(t, &answers, strings.Repeat("\n", 8))

	if strings.Contains(out, "cannot be empty") {
		t.Errorf("a blank answer was rejected, so a screen-reader user cannot keep the seeded "+
			"host.\noutput:\n%s", out)
	}
	if answers.Host != seededHost {
		t.Errorf("host = %q after a blank answer, want the seeded %q", answers.Host, seededHost)
	}
}

// TestAccessibleStillRejectsBlankWithoutASeed is the negative half: with nothing
// to keep, a required field must still refuse a blank answer.
func TestAccessibleStillRejectsBlankWithoutASeed(t *testing.T) {
	answers := ConnectionAnswers{}
	out := runConnectionFormAccessible(t, &answers, strings.Repeat("\n", 4))

	if !strings.Contains(out, "cannot be empty") {
		t.Errorf("an unseeded required field accepted a blank answer, so the blank-pass "+
			"wrapper leaked beyond seeded fields.\noutput:\n%s", out)
	}
}

// TestConnectionAnswersRoundTrip checks the form answers become the settings the
// client and the service launcher actually consume.
func TestConnectionAnswersRoundTrip(t *testing.T) {
	answers := ConnectionAnswers{
		Host: "0.0.0.0", Port: "9100", Model: " qwen3.5:9b ",
		ConfigPath: "custom.yaml", Launch: true,
	}
	settings := answers.Export()

	if settings.Port != 9100 {
		t.Errorf("port = %d, want 9100", settings.Port)
	}
	if settings.Model != "qwen3.5:9b" {
		t.Errorf("model = %q, want it trimmed", settings.Model)
	}
	// A server bound to 0.0.0.0 is reached at 127.0.0.1, as app.py does.
	if settings.BaseURL != "http://127.0.0.1:9100" {
		t.Errorf("base URL = %q, want the wildcard normalized", settings.BaseURL)
	}
	if !settings.LaunchServer {
		t.Error("the launch answer was dropped")
	}
}

// TestConnectionAnswersSurviveGarbagePort: a hand-edited settings file or a bad
// answer must fall back to the documented default rather than port 0.
func TestConnectionAnswersSurviveGarbagePort(t *testing.T) {
	settings := ConnectionAnswers{Host: "127.0.0.1", Port: "not-a-port"}.Export()
	if settings.Port != session.DefaultPort {
		t.Errorf("port = %d, want the default %d", settings.Port, session.DefaultPort)
	}
}
