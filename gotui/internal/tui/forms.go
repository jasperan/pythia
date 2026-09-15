package tui

import (
	"fmt"
	"io"
	"strings"

	"charm.land/huh/v2"

	"github.com/jasperan/pythia/gotui/internal/huhstyle"
	"github.com/jasperan/pythia/gotui/internal/session"
)

// ConnectionAnswers backs the connection form.
//
// It is a pointer type throughout: huh writes through the pointer, and binding a
// field to a field of a value-typed struct would silently keep the default.
type ConnectionAnswers struct {
	Host       string
	Port       string
	Model      string
	ConfigPath string
	Launch     bool
}

// ConnectionDefaults seeds the form from saved settings and pythia.yaml.
func ConnectionDefaults(settings session.Settings, model string) ConnectionAnswers {
	settings = settings.Normalize()
	return ConnectionAnswers{
		Host:       settings.Host,
		Port:       fmt.Sprintf("%d", settings.Port),
		Model:      model,
		ConfigPath: settings.ConfigPath,
		Launch:     settings.LaunchServer,
	}
}

// Export converts the form answers back into settings.
func (a ConnectionAnswers) Export() session.Settings {
	port := 0
	if _, err := fmt.Sscanf(strings.TrimSpace(a.Port), "%d", &port); err != nil {
		port = session.DefaultPort
	}
	settings := session.Settings{
		Host:         strings.TrimSpace(a.Host),
		Port:         port,
		Model:        strings.TrimSpace(a.Model),
		ConfigPath:   strings.TrimSpace(a.ConfigPath),
		LaunchServer: a.Launch,
	}
	settings.BaseURL = session.BaseURL(settings.Host, settings.Port)
	return settings.Normalize()
}

// Themed connects a form to the project theme, keymap and accessibility flag.
//
// The keymap matters: a bare field has none, so keystrokes would be silently
// ignored.
func Themed(form *huh.Form) *huh.Form {
	return form.
		WithTheme(huh.ThemeFunc(huhstyle.Theme)).
		WithAccessible(huhstyle.Accessible()).
		WithKeyMap(huh.NewDefaultKeyMap())
}

// ValidateDefaulted accepts an empty answer as "keep the value already there".
//
// It exists for accessible mode. huh's screen-reader path runs a field's
// validator on the raw line and only afterwards substitutes the field's default,
// and it never prints that default. A pre-filled field whose validator rejects
// "" therefore re-prompts on every bare Enter, so a screen-reader user cannot
// accept a value they cannot see -- which here would mean being unable to get
// past the connection form at all.
func ValidateDefaulted(inner func(string) error) func(string) error {
	return func(s string) error {
		if strings.TrimSpace(s) == "" {
			return nil
		}
		return inner(s)
	}
}

// ValidateDefaultedValue is ValidateDefaulted for a field whose pre-filled value
// may itself be empty.
//
// The host and model are both seeded from settings and pythia.yaml, and either
// may be absent, so a blank answer only counts as "keep the default" when there
// is in fact a default to keep.
func ValidateDefaultedValue(prefilled string, inner func(string) error) func(string) error {
	if strings.TrimSpace(prefilled) == "" {
		return inner
	}
	return ValidateDefaulted(inner)
}

// ConnectionForm builds the connection settings form.
//
// It is split into two groups, which huh renders as pages: five bordered fields
// do not fit a 24-row terminal and huh clips whatever overflows.
func ConnectionForm(answers *ConnectionAnswers) *huh.Form {
	return Themed(huh.NewForm(
		huh.NewGroup(
			huh.NewInput().
				Title("API host").
				Description("Use the project root's pythia.yaml server.host").
				Placeholder(session.DefaultHost).
				Value(&answers.Host).
				Validate(ValidateDefaultedValue(answers.Host, huh.ValidateNotEmpty())),
			huh.NewInput().
				Title("API port").
				Description("1-65535; the default is 8900").
				Placeholder(fmt.Sprintf("%d", session.DefaultPort)).
				Value(&answers.Port).
				Validate(ValidateDefaultedValue(answers.Port, session.ValidatePort)),
		).Title("Connection").Description("the pythia FastAPI service the TUI attaches to"),
		huh.NewGroup(
			huh.NewInput().
				Title("Ollama model").
				Description("Sent as the search request's model; empty uses the server default").
				Value(&answers.Model),
			huh.NewInput().
				Title("Config path").
				Description("pythia.yaml, used when starting the service").
				Placeholder(session.DefaultConfig).
				Value(&answers.ConfigPath).
				Validate(ValidateDefaultedValue(answers.ConfigPath, huh.ValidateNotEmpty())),
			huh.NewConfirm().
				Title("Start the API server if it is not running?").
				Description("Runs the project's own `python -m pythia serve`, the same command "+
					"the Python TUI uses. Backends (Oracle, SearXNG, Ollama) are not started.").
				Affirmative("Start it").
				Negative("Attach only").
				Value(&answers.Launch),
		).Title("Service"),
	))
}

// RunConnectionPrompts runs the connection form standalone for the accessible
// and non-TTY paths.
//
// It runs as ONE form over one reader, deliberately. huh's accessible Form.Run
// wraps the reader in its own scanner, which buffers ahead; a second form built
// over the same stdin would therefore start at EOF and return empty answers for
// every field.
func RunConnectionPrompts(in io.Reader, out io.Writer, defaults ConnectionAnswers) (ConnectionAnswers, error) {
	answers := defaults
	form := ConnectionForm(&answers).WithInput(in).WithOutput(out)
	if err := form.Run(); err != nil {
		return defaults, err
	}
	return answers, nil
}
