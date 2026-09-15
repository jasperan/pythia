// Command pythia-tui is an additional way to run Pythia: a Go front-end in the
// charm v2 + huh stack that talks to the same FastAPI service the Python Textual
// TUI and the `pythia query`/`pythia research` CLI commands already use.
//
// It never reimplements search, research, grounding or caching. Every answer,
// source, finding and statistic is fetched from src/pythia/server, so a Go user
// and a Python user get identical results, including the same semantic-cache
// hits and similarity scores.
//
// Backends are not managed: Oracle, SearXNG and Ollama are reported by the
// dashboard's health row and started with `docker compose up -d`, which is the
// same division of labour pythia.yaml describes.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
	"github.com/jasperan/pythia/gotui/internal/huhstyle"
	"github.com/jasperan/pythia/gotui/internal/session"
	"github.com/jasperan/pythia/gotui/internal/tui"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "pythia-tui: "+err.Error())
		os.Exit(1)
	}
}

func run() error {
	var (
		baseURL     = flag.String("base-url", "", "pythia API URL (default: derived from --host/--port, then saved settings)")
		host        = flag.String("host", "", "override the API host (default: pythia.yaml server.host, or 127.0.0.1)")
		port        = flag.Int("port", 0, "override the API port (default: pythia.yaml server.port, or 8900)")
		model       = flag.String("model", "", "Ollama model to request (default: settings, else the server's own)")
		configPath  = flag.String("config", "", "path to pythia.yaml (default: the project root's pythia.yaml)")
		startSvc    = flag.Bool("start-service", false, "start the project's own API server with `python -m pythia serve` before attaching")
		projectRoot = flag.String("project-root", "", "the pythia checkout (default: discovered from the working directory)")
		setupFlag   = flag.Bool("setup", false, "run the connection form, save the answers, and exit")

		query     = flag.String("query", "", "ask one question, print the answer, then exit")
		research  = flag.String("research", "", "run one deep-research question, print the report, then exit")
		history   = flag.Bool("history", false, "print recent searches, then exit")
		statsFlag = flag.Bool("stats", false, "print cache statistics, then exit")
		health    = flag.Bool("health", false, "print backing-service health, then exit")
		skills    = flag.Bool("skills", false, "list research skills, then exit")
		clear     = flag.Bool("clear-cache", false, "delete every cached answer (requires --yes)")
		yes       = flag.Bool("yes", false, "confirm a destructive --clear-cache without prompting")

		deep      = flag.Bool("deep", false, "scrape the top URLs for full content on --query")
		maxRounds = flag.Int("max-rounds", 0, "override the research round cap for --research")
		limit     = flag.Int("limit", 20, "rows for --history")
		jsonFlag  = flag.Bool("json", false, "emit machine-readable JSON for scripted actions")
		noInput   = flag.Bool("no-input", false, "never prompt; fail instead if input is required")
	)

	flag.Parse()

	settings, err := session.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "note: "+err.Error())
	}
	settings = settings.Normalize()

	if err := applyOverrides(&settings, *baseURL, *host, *port, *model, *configPath, *startSvc); err != nil {
		return err
	}

	root := *projectRoot
	if root == "" {
		root = defaultProjectRoot()
	}

	// --- scripted path ---------------------------------------------------------
	// This runs before any prompt is considered, so a pipeline never blocks on a
	// question.
	action, statusAction := tui.ParseActionFlags(*query, *research, *history,
		*statsFlag, *health, *skills, *clear)
	if statusAction {
		return runScripted(settings, root, tui.ActionRequest{
			Action:    action,
			Query:     firstNonEmpty(*research, *query),
			Model:     settings.Model,
			Deep:      *deep,
			MaxRounds: *maxRounds,
			Limit:     *limit,
			Confirmed: *yes,
			JSON:      *jsonFlag,
		})
	}

	// --- setup-only path -------------------------------------------------------
	if *setupFlag {
		return runSetup(settings, root, *noInput)
	}

	// --- screen-reader / piped path --------------------------------------------
	// huh's accessible rendering only exists in its standalone Run path, so the
	// embedded full-screen UI is skipped entirely here.
	if huhstyle.Accessible() {
		fmt.Fprintln(os.Stdout, tui.AccessibleNotice)
		return runAccessible(settings, root, *noInput)
	}
	if !huhstyle.Interactive() || *noInput {
		return errors.New("no terminal on stdin: pass an action flag such as --query \"...\", " +
			"--research \"...\", --history, --stats, --health or --skills (or set ACCESSIBLE " +
			"for plain prompts)")
	}

	// --- full-screen path ------------------------------------------------------
	client := api.NewClient(settings.BaseURL)
	server, err := startServiceIfRequested(settings, root)
	if err != nil {
		return err
	}
	if server != nil {
		defer server.Stop()
	}

	app := tui.New(tui.Options{
		ProjectRoot: root,
		Settings:    settings,
		Client:      client,
		Model:       settings.Model,
	})
	defer app.Close()

	program := tea.NewProgram(app)
	if _, err := program.Run(); err != nil {
		return err
	}
	return nil
}

// applyOverrides folds flags into settings and validates what it can.
func applyOverrides(settings *session.Settings, baseURL, host string, port int,
	model, configPath string, startService bool) error {

	if strings.TrimSpace(baseURL) != "" {
		if err := api.ValidateBaseURL(baseURL); err != nil {
			return fmt.Errorf("--base-url: %w", err)
		}
		settings.BaseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
		// Keep host/port coherent with the URL. The connection form seeds its
		// host and port fields from settings and its answers are the settings
		// that get saved, so leaving them behind would silently discard
		// --base-url and save a config pointing somewhere else.
		if parsed, parseErr := url.Parse(settings.BaseURL); parseErr == nil {
			if parsed.Hostname() != "" {
				settings.Host = parsed.Hostname()
			}
			if parsed.Port() != "" {
				if parsedPort, convErr := strconv.Atoi(parsed.Port()); convErr == nil {
					settings.Port = parsedPort
				}
			}
		}
	}
	if strings.TrimSpace(host) != "" {
		settings.Host = strings.TrimSpace(host)
	}
	if port != 0 {
		if err := session.ValidatePort(fmt.Sprint(port)); err != nil {
			return fmt.Errorf("--port: %w", err)
		}
		settings.Port = port
	}
	if strings.TrimSpace(model) != "" {
		settings.Model = strings.TrimSpace(model)
	}
	if strings.TrimSpace(configPath) != "" {
		settings.ConfigPath = strings.TrimSpace(configPath)
	}
	if startService {
		settings.LaunchServer = true
	}
	// A base URL given explicitly wins; otherwise keep it in step with host/port.
	if strings.TrimSpace(baseURL) == "" {
		settings.BaseURL = session.BaseURL(settings.Host, settings.Port)
	}
	return nil
}

// runScripted attaches (starting the service when asked) and runs one action.
func runScripted(settings session.Settings, root string, req tui.ActionRequest) error {
	ctx := context.Background()

	// Reads do not need the service started for them; only do it when asked.
	if settings.LaunchServer {
		server, err := startServiceIfRequested(settings, root)
		if err != nil {
			return err
		}
		if server != nil {
			defer server.Stop()
		}
	}

	client := api.NewClient(settings.BaseURL)
	return tui.RunAction(ctx, client, req, os.Stdout)
}

// runSetup runs the connection form and persists the answers.
func runSetup(settings session.Settings, root string, noInput bool) error {
	if noInput {
		return errors.New("--setup needs a terminal to prompt on")
	}
	answers, err := tui.RunConnectionPrompts(os.Stdin, os.Stdout,
		tui.ConnectionDefaults(settings, settings.Model))
	if err != nil {
		return err
	}
	chosen := answers.Export()
	if err := session.Save(chosen); err != nil {
		return err
	}
	path, _ := session.Path()
	fmt.Fprintf(os.Stdout, "saved %s\n", path)
	return nil
}

// runAccessible drives the same connection form through plain prompts.
func runAccessible(settings session.Settings, root string, noInput bool) error {
	if noInput {
		return errors.New("ACCESSIBLE plain prompts cannot be combined with --no-input")
	}
	answers, err := tui.RunConnectionPrompts(os.Stdin, os.Stdout,
		tui.ConnectionDefaults(settings, settings.Model))
	if err != nil {
		return err
	}
	chosen := answers.Export()
	if err := session.Save(chosen); err != nil {
		fmt.Fprintln(os.Stderr, "note: "+err.Error())
	}

	if chosen.LaunchServer {
		server, err := startServiceIfRequested(chosen, root)
		if err != nil {
			return err
		}
		if server != nil {
			defer server.Stop()
		}
	}

	fmt.Fprintf(os.Stdout, "\nConnection saved: %s\n", chosen.BaseURL)
	fmt.Fprintln(os.Stdout, "Ask a question with: pythia-tui --query \"your question\"")
	fmt.Fprintln(os.Stdout, "Run deep research with: pythia-tui --research \"your topic\"")
	return nil
}

// startServiceIfRequested launches the project's own API server when the answer
// or the flag asked for it, and waits until it accepts connections.
func startServiceIfRequested(settings session.Settings, root string) (*session.Server, error) {
	settings = settings.Normalize()
	if !settings.LaunchServer {
		return nil, nil
	}
	// A server already answering means the user started one themselves; attach
	// rather than fail on a busy port.
	if session.PortInUse(settings.Host, settings.Port) {
		fmt.Fprintf(os.Stderr, "note: a service already answers on %s; attaching to it\n",
			settings.BaseURL)
		return nil, nil
	}
	if root == "" {
		return nil, errors.New("cannot start the service: no pythia project root found " +
			"(pass --project-root)")
	}
	configPath := session.ResolveConfigPath(root, settings.ConfigPath)
	if _, err := os.Stat(configPath); err != nil {
		return nil, fmt.Errorf("cannot start the service: %s not found at %s", session.DefaultConfig, configPath)
	}

	fmt.Fprintf(os.Stderr, "Starting the pythia API on %s...\n", settings.BaseURL)
	// No cancel handle is kept: Stop() terminates the process group, which is
	// what actually ends the server, and WaitForPort bounds its own wait.
	ctx := context.Background()
	server, err := session.LaunchServer(ctx, root, settings, os.Stderr)
	if err != nil {
		return nil, err
	}
	if err := session.WaitForPort(ctx, settings.Host, settings.Port, 60*time.Second); err != nil {
		server.Stop()
		return nil, err
	}
	return server, nil
}

// defaultProjectRoot walks up from the executable and then the working
// directory, because the binary may be built anywhere.
func defaultProjectRoot() string {
	if root := session.FindProjectRoot(""); root != "" {
		return root
	}
	if executable, err := os.Executable(); err == nil {
		return session.FindProjectRoot(filepath.Dir(executable))
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
