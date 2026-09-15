package main

import (
	"testing"

	"github.com/jasperan/pythia/gotui/internal/session"
	"github.com/jasperan/pythia/gotui/internal/tui"
)

// TestApplyOverridesBaseURLKeepsHostAndPortCoherent is the regression guard.
//
// The connection form seeds its host and port fields from settings, and its
// answers are the settings that get saved. If --base-url only overwrote
// BaseURL, the form would show the old host/port, the saved config would point
// somewhere else, and --base-url would be silently discarded.
func TestApplyOverridesBaseURLKeepsHostAndPortCoherent(t *testing.T) {
	settings := session.Default()
	if err := applyOverrides(&settings, "http://127.0.0.1:8931", "", 0, "", "", false); err != nil {
		t.Fatalf("applyOverrides returned %v", err)
	}

	if settings.BaseURL != "http://127.0.0.1:8931" {
		t.Errorf("BaseURL = %q", settings.BaseURL)
	}
	if settings.Host != "127.0.0.1" {
		t.Errorf("Host = %q, want it derived from the URL", settings.Host)
	}
	if settings.Port != 8931 {
		t.Errorf("Port = %d, want it derived from the URL", settings.Port)
	}

	// The form round-trip must preserve the address.
	answers := tui.ConnectionDefaults(settings, settings.Model)
	if answers.Port != "8931" {
		t.Errorf("the form would seed port %q, want 8931", answers.Port)
	}
	if got := answers.Export().BaseURL; got != "http://127.0.0.1:8931" {
		t.Errorf("saving the form would store %q, losing --base-url", got)
	}
}

// TestApplyOverridesBaseURLWithoutPort keeps the default port when the URL omits
// one, rather than zeroing it.
func TestApplyOverridesBaseURLWithoutPort(t *testing.T) {
	settings := session.Default()
	if err := applyOverrides(&settings, "https://pythia.example", "", 0, "", "", false); err != nil {
		t.Fatalf("applyOverrides returned %v", err)
	}
	if settings.Host != "pythia.example" {
		t.Errorf("Host = %q, want pythia.example", settings.Host)
	}
	if settings.Port != session.DefaultPort {
		t.Errorf("Port = %d, want the default %d", settings.Port, session.DefaultPort)
	}
	if settings.BaseURL != "https://pythia.example" {
		t.Errorf("an explicit URL was rewritten to %q", settings.BaseURL)
	}
}

// TestApplyOverridesRejectsBadInput covers the validation that stops a typo from
// reaching the HTTP client as a confusing dial error.
func TestApplyOverridesRejectsBadInput(t *testing.T) {
	cases := []struct {
		name    string
		baseURL string
		port    int
	}{
		{name: "bad scheme", baseURL: "ftp://host"},
		{name: "no scheme", baseURL: "127.0.0.1:8900"},
		{name: "port too high", port: 70000},
		{name: "port negative", port: -1},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			settings := session.Default()
			if err := applyOverrides(&settings, testCase.baseURL, "", testCase.port, "", "", false); err == nil {
				t.Error("invalid input was accepted")
			}
		})
	}
}

// TestApplyOverridesHostAndPortRecomputeBaseURL keeps the derived URL in step
// when host/port are given directly instead of as a URL.
func TestApplyOverridesHostAndPortRecomputeBaseURL(t *testing.T) {
	settings := session.Default()
	if err := applyOverrides(&settings, "", "0.0.0.0", 9100, "", "", false); err != nil {
		t.Fatalf("applyOverrides returned %v", err)
	}
	if settings.BaseURL != "http://127.0.0.1:9100" {
		t.Errorf("BaseURL = %q, want the wildcard bind normalized", settings.BaseURL)
	}
}

// TestApplyOverridesFlagOnlyValuesFallBack to pythia.yaml-derived defaults.
func TestApplyOverridesFlagOnlyValuesLeaveDefaults(t *testing.T) {
	settings := session.Default()
	if err := applyOverrides(&settings, "", "", 0, "", "", false); err != nil {
		t.Fatalf("applyOverrides returned %v", err)
	}
	if settings.Host != session.DefaultHost || settings.Port != session.DefaultPort {
		t.Errorf("defaults were changed: %+v", settings)
	}
	if settings.BaseURL != "http://127.0.0.1:8900" {
		t.Errorf("BaseURL = %q, want it derived from the defaults", settings.BaseURL)
	}
}

// TestApplyOverridesStartServiceSetsTheLaunchFlag pins --start-service, which is
// what makes the TUI start the project's own server.
func TestApplyOverridesStartServiceSetsTheLaunchFlag(t *testing.T) {
	settings := session.Default()
	if err := applyOverrides(&settings, "", "", 0, "", "", true); err != nil {
		t.Fatalf("applyOverrides returned %v", err)
	}
	if !settings.LaunchServer {
		t.Error("--start-service did not set LaunchServer")
	}
}
