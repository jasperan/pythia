// Package tui is the Bubble Tea front-end for Pythia.
//
// It is an ADDITIONAL way to run the project: a Go terminal UI in the charm v2 +
// huh stack that talks to the same FastAPI service the Python Textual TUI uses.
//
// It never reimplements search, research, grounding, caching or scoring. Every
// answer, source, finding and statistic comes from src/pythia/server over HTTP,
// so a Go user and a Python user get identical results -- including the same
// cache hits and the same semantic-cache similarity scores.
package tui

import (
	"context"
	"errors"
	"strconv"
	"strings"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
	"github.com/jasperan/pythia/gotui/internal/session"
)

// Screen identifies the active tab.
type Screen int

const (
	// ScreenSearch is the single-shot search and streaming answer screen.
	ScreenSearch Screen = iota
	// ScreenResearch is the multi-round deep-research screen.
	ScreenResearch
	// ScreenHistory lists past searches from Oracle.
	ScreenHistory
	// ScreenDashboard shows service health, cache statistics and skills.
	ScreenDashboard
)

var screenNames = []string{"Search", "Research", "History", "Dashboard"}

// errNoClient is returned when the model was built without an API client. It is
// reported to the user instead of panicking on a nil dereference.
var errNoClient = errors.New("no API client is configured")

// Options configures a Model.
type Options struct {
	ProjectRoot string
	Settings    session.Settings
	Client      *api.Client
	// Model is the Ollama model the server should use, from pythia.yaml.
	Model string
}

// streamMsg is one item from a streaming goroutine.
type streamMsg struct {
	event api.Event
	err   error
	done  bool
}

// historyMsg carries a finished GET /history.
type historyMsg struct {
	items []api.HistoryItem
	err   error
}

// overviewMsg carries a finished dashboard refresh.
type overviewMsg struct {
	stats  api.Stats
	health api.Health
	skills []api.Skill
	err    error
}

// clearCacheMsg carries the result of DELETE /cache.
type clearCacheMsg struct {
	deleted int
	err     error
}

// searchState is everything the search screen accumulates.
type searchState struct {
	// answer is a POINTER on purpose. Bubble Tea's Update receives the model by
	// value, so a strings.Builder held directly here would be copied on every
	// message and panic ("illegal use of non-zero Builder copied by value") the
	// next time a token arrived.
	answer      *strings.Builder
	Sources     []api.Source
	Suggestions []string
	Grounding   *api.GroundingData
	Status      string
	Done        bool
	DoneData    api.SearchDoneData
	// Conversation is the multi-turn context the Python TUI sends back as
	// conversation_history (search.py:182). It is capped the same way.
	Conversation []api.Message
}

// browseState backs the history and dashboard tabs.
type browseState struct {
	history []api.HistoryItem
	stats   api.Stats
	health  api.Health
	skills  []api.Skill
	loaded  bool
}

// Model is the root Bubble Tea model.
type Model struct {
	opts   Options
	width  int
	height int
	active Screen

	query  textinput.Model
	typing bool
	deep   bool
	busy   bool

	search   searchState
	research ResearchState
	browse   browseState

	events    chan streamMsg
	cancel    context.CancelFunc
	spinner   spinner.Model
	help      help.Model
	keys      keyMap
	status    string
	err       error
	notice    string
	scrollTop int
}

// New builds the root model.
func New(opts Options) Model {
	input := textinput.New()
	input.Placeholder = "Ask a question…  (Enter to send)"
	input.Prompt = "› "
	input.CharLimit = 2000

	sp := spinner.New(spinner.WithSpinner(spinner.Dot), spinner.WithStyle(progressStyle))

	model := Model{
		opts:    opts,
		active:  ScreenSearch,
		query:   input,
		width:   80,
		height:  24,
		spinner: sp,
		help:    help.New(),
		keys:    defaultKeys(),
	}
	// Start with the input focused: the first thing a user does is type.
	model.query.Focus()
	model.typing = true
	return model
}

// keyMap implements help.KeyMap so the footer comes from bubbles/help rather
// than a hand-rolled string.
type keyMap struct {
	Send    key.Binding
	NextTab key.Binding
	Deep    key.Binding
	Refresh key.Binding
	Clear   key.Binding
	Quit    key.Binding
	Cancel  key.Binding
	Help    key.Binding
}

func defaultKeys() keyMap {
	return keyMap{
		Send: key.NewBinding(
			key.WithKeys("enter"),
			key.WithHelp("enter", "send"),
		),
		NextTab: key.NewBinding(
			key.WithKeys("tab"),
			key.WithHelp("tab", "switch"),
		),
		Deep: key.NewBinding(
			key.WithKeys("ctrl+d"),
			key.WithHelp("ctrl+d", "deep"),
		),
		Refresh: key.NewBinding(
			key.WithKeys("ctrl+r"),
			key.WithHelp("ctrl+r", "refresh"),
		),
		Clear: key.NewBinding(
			key.WithKeys("ctrl+l"),
			key.WithHelp("ctrl+l", "clear"),
		),
		Cancel: key.NewBinding(
			key.WithKeys("esc"),
			key.WithHelp("esc", "stop/cancel"),
		),
		Help: key.NewBinding(
			key.WithKeys("ctrl+h"),
			key.WithHelp("ctrl+h", "help"),
		),
		Quit: key.NewBinding(
			key.WithKeys("ctrl+c", "q"),
			key.WithHelp("ctrl+c", "quit"),
		),
	}
}

// ShortHelp is the one-line footer.
func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Send, k.NextTab, k.Deep, k.Cancel, k.Quit}
}

// FullHelp is the expanded footer.
func (k keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Send, k.NextTab, k.Deep},
		{k.Cancel, k.Refresh, k.Clear},
		{k.Help, k.Quit},
	}
}

// Init starts the spinner and loads the dashboard in the background.
func (m Model) Init() tea.Cmd {
	return tea.Batch(m.spinner.Tick, m.refreshCmd(), m.loadHistoryCmd())
}

// Close releases the streaming context. main calls it after Run returns.
func (m *Model) Close() {
	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}
}

// Update is the Elm update function.
//
// Only tea.KeyPressMsg is matched, never the tea.KeyMsg interface: in v2
// KeyMsg is an interface satisfied by BOTH KeyPressMsg and KeyReleaseMsg, so a
// handler written against it runs twice per keystroke and toggles cancel
// themselves out. keyrelease_test.go pins this.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = clampWidth(msg.Width)
		m.height = clampHeight(msg.Height)
		m.help.SetWidth(m.width)
		m.query.SetWidth(clampPositive(m.width - 6))
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case streamMsg:
		return m.handleStream(msg)

	case historyMsg:
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.browse.history = msg.items
		}
		return m, nil

	case overviewMsg:
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.browse.stats = msg.stats
			m.browse.health = msg.health
			m.browse.skills = msg.skills
			m.browse.loaded = true
		}
		return m, nil

	case clearCacheMsg:
		if msg.err != nil {
			m.err = msg.err
		} else {
			m.notice = "cache cleared: " + strconv.Itoa(msg.deleted) + " entries"
		}
		return m, nil

	case tea.KeyPressMsg:
		return m.handleKey(msg)
	}
	return m, nil
}

func (m Model) handleStream(msg streamMsg) (tea.Model, tea.Cmd) {
	if msg.done {
		m.busy = false
		m.status = ""
		return m, nil
	}
	if msg.err != nil {
		m.busy = false
		m.status = ""
		m.err = msg.err
		return m, nil
	}
	switch m.active {
	case ScreenSearch:
		m.search.apply(msg.event)
	case ScreenResearch:
		if err := m.research.ApplyEvent(msg.event); err != nil {
			m.err = err
		}
	}
	// Keep reading the stream.
	return m, waitForStream(m.events)
}

// handleKey is the single key entry point.
func (m Model) handleKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		return m, tea.Quit
	case "q":
		// "q" only quits when it is not being typed into the query box.
		if !m.typing {
			return m, tea.Quit
		}
	case "tab":
		return m.switchScreen((m.active + 1) % Screen(len(screenNames))), nil
	case "shift+tab":
		next := (int(m.active) - 1 + len(screenNames)) % len(screenNames)
		return m.switchScreen(Screen(next)), nil
	case "esc":
		if m.busy {
			// Stop the stream but stay where we are: cancelling the context
			// closes the SSE response, which ends the goroutine cleanly.
			if m.cancel != nil {
				m.cancel()
				m.cancel = nil
			}
			m.busy = false
			m.status = ""
			m.notice = "cancelled"
			return m, nil
		}
		m.typing = false
		m.query.Blur()
		return m, nil
	case "ctrl+d":
		m.deep = !m.deep
		if m.deep {
			m.notice = "deep scrape: ON"
		} else {
			m.notice = "deep scrape: OFF"
		}
		return m, nil
	case "ctrl+r":
		return m, tea.Batch(m.refreshCmd(), m.loadHistoryCmd())
	case "ctrl+l":
		m.search = searchState{}
		m.research = ResearchState{}
		m.err = nil
		m.notice = "cleared"
		return m, nil
	case "ctrl+h":
		m.help.ShowAll = !m.help.ShowAll
		return m, nil
	case "enter":
		if !m.typing {
			// Re-focus the prompt instead of sending an empty query.
			m.typing = true
			m.query.Focus()
			return m, nil
		}
		return m.submit()
	}

	// Number keys jump between tabs, but only when the user is not typing --
	// the same guard src/pythia/tui/app.py:on_key applies.
	if !m.typing && len(msg.String()) == 1 && msg.String() >= "1" && msg.String() <= "4" {
		index := int(msg.String()[0] - '1')
		if index < len(screenNames) {
			return m.switchScreen(Screen(index)), nil
		}
	}

	if m.typing {
		var cmd tea.Cmd
		m.query, cmd = m.query.Update(msg)
		return m, cmd
	}

	// Not typing: page the active pane.
	return m.scrollActive(msg), nil
}

// submit dispatches the typed query to the active screen's endpoint.
func (m Model) submit() (tea.Model, tea.Cmd) {
	text := strings.TrimSpace(m.query.Value())
	if text == "" {
		return m, nil
	}
	if m.opts.Client == nil {
		m.err = errNoClient
		return m, nil
	}
	m.err = nil
	m.notice = ""
	m.query.SetValue("")
	m.typing = false
	m.query.Blur()
	m.busy = true
	m.scrollTop = 0

	switch m.active {
	case ScreenResearch:
		m.research.Reset(text)
		m.status = "starting research…"
		return m.startStream(m.researchRequest(text))
	case ScreenHistory, ScreenDashboard:
		// Those tabs have no query endpoint; send it as a search instead and
		// move the user there so the result is not invisible.
		m.active = ScreenSearch
	}

	m.search = searchState{Conversation: m.search.Conversation}
	m.search.beginTurn(text)
	m.status = "searching…"
	return m.startStream(m.searchRequest(text))
}

// startStream runs the request in a goroutine and forwards its events.
func (m Model) startStream(req streamRequest) (Model, tea.Cmd) {
	ctx, cancel := context.WithCancel(context.Background())
	if m.cancel != nil {
		m.cancel()
	}
	m.cancel = cancel

	channel := make(chan streamMsg, 128)
	m.events = channel

	client := m.opts.Client
	go func() {
		defer close(channel)
		err := req.run(ctx, client, func(event api.Event) error {
			select {
			case channel <- streamMsg{event: event}:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		// A cancelled stream is a user action, not a failure.
		if err != nil && ctx.Err() == nil {
			select {
			case channel <- streamMsg{err: err}:
			case <-ctx.Done():
			}
		}
	}()

	return m, tea.Batch(waitForStream(channel), m.spinner.Tick)
}

// streamRequest is a small indirection so startStream serves both endpoints.
type streamRequest struct {
	search   *api.SearchRequest
	research *api.ResearchRequest
}

func (r streamRequest) run(ctx context.Context, client *api.Client, yield func(api.Event) error) error {
	switch {
	case r.research != nil:
		return client.StreamResearch(ctx, *r.research, yield)
	case r.search != nil:
		return client.StreamSearch(ctx, *r.search, yield)
	}
	return errNoClient
}

func (m Model) searchRequest(query string) streamRequest {
	req := &api.SearchRequest{Query: query, Model: m.opts.Model, Deep: m.deep}
	// Send prior turns so the server can use multi-turn context, exactly as
	// src/pythia/tui/screens/search.py:186 does.
	if len(m.search.Conversation) > 0 {
		req.History = append([]api.Message(nil), m.search.Conversation...)
	}
	return streamRequest{search: req}
}

func (m Model) researchRequest(query string) streamRequest {
	return streamRequest{research: &api.ResearchRequest{Query: query, Model: m.opts.Model}}
}

// switchScreen changes tabs, cancelling an in-flight stream.
func (m Model) switchScreen(screen Screen) Model {
	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}
	m.busy = false
	m.active = screen
	m.status = ""
	// The query box is only meaningful on the two query screens.
	if screen == ScreenSearch || screen == ScreenResearch {
		if !m.typing {
			m.typing = true
			m.query.Focus()
		}
	} else {
		m.typing = false
		m.query.Blur()
	}
	return m
}

// scrollActive pages the active screen's scrollable pane.
func (m Model) scrollActive(msg tea.KeyPressMsg) Model {
	switch msg.String() {
	case "up", "k":
		m.scrollTop--
	case "down", "j":
		m.scrollTop++
	case "pgup":
		m.scrollTop -= m.bodyHeight()
	case "pgdown":
		m.scrollTop += m.bodyHeight()
	case "home", "g":
		m.scrollTop = 0
	case "end", "G":
		m.scrollTop = 1 << 20
	}
	if m.scrollTop < 0 {
		m.scrollTop = 0
	}
	return m
}

// refreshCmd loads health, stats and skills for the dashboard.
func (m Model) refreshCmd() tea.Cmd {
	client := m.opts.Client
	if client == nil {
		return nil
	}
	return func() tea.Msg {
		ctx := context.Background()
		var out overviewMsg
		out.health, out.err = client.Health(ctx)
		if out.err != nil {
			return out
		}
		out.stats, out.err = client.Stats(ctx)
		if out.err != nil {
			return out
		}
		out.skills, out.err = client.Skills(ctx)
		return out
	}
}

// loadHistoryCmd loads the recent-query list.
func (m Model) loadHistoryCmd() tea.Cmd {
	client := m.opts.Client
	if client == nil {
		return nil
	}
	return func() tea.Msg {
		items, err := client.History(context.Background(), api.MaxHistoryLimit)
		return historyMsg{items: items, err: err}
	}
}

// waitForStream blocks until the next event arrives.
func waitForStream(channel <-chan streamMsg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-channel
		if !ok {
			return streamMsg{done: true}
		}
		return msg
	}
}

// View renders the whole application.
func (m Model) View() tea.View {
	var out strings.Builder

	out.WriteString(m.renderHeader())

	body := m.renderBody()
	if body != "" {
		out.WriteString("\n")
		out.WriteString(body)
	}

	out.WriteString("\n")
	out.WriteString(m.renderFooter())

	view := tea.NewView(out.String())
	view.AltScreen = true
	view.WindowTitle = "Pythia"
	return view
}

func (m Model) renderHeader() string {
	title := titleStyle.Render("PYTHIA") + " " + mutedStyle.Render("self-hosted AI search")
	tabs := tabBar(screenNames, int(m.active), m.width)
	// The bar is padded horizontally, so the text budget is the terminal width
	// minus that padding; truncating to the full width would wrap the header.
	inner := clampPositive(m.width - 2)
	return headerBarStyle.Width(inner).Render(truncate(title+"  "+tabs, inner))
}

// bodyHeight is the number of rows available to the active screen.
func (m Model) bodyHeight() int {
	// Header, query line, status line, help line and the blank separators.
	return clampPositive(m.height - 5)
}

func (m Model) renderBody() string {
	height := m.bodyHeight()
	var body string
	switch m.active {
	case ScreenSearch:
		body = m.renderSearch(height)
	case ScreenResearch:
		body = m.renderResearch(height)
	case ScreenHistory:
		body = renderHistory(m.browse, height, clampPositive(m.width-2))
	case ScreenDashboard:
		body = renderDashboard(m.browse, height, clampPositive(m.width-2))
	}
	return body
}

func (m Model) renderFooter() string {
	var lines []string

	// If the active screen owns a prompt, show it. Otherwise show a hint, so a
	// user always knows how to get back to typing.
	if m.active == ScreenSearch || m.active == ScreenResearch {
		line := m.query.View()
		if !m.typing {
			line = dimStyle.Render("› press enter to type a question")
		}
		lines = append(lines, truncate(line, m.width))
	}

	switch {
	case m.err != nil:
		lines = append(lines, truncate(errorStyle.Render("✗ "+m.err.Error()), m.width))
	case m.busy:
		lines = append(lines, truncate("  "+m.spinner.View()+" "+m.status, m.width))
	case m.notice != "":
		lines = append(lines, truncate(dimStyle.Render(m.notice), m.width))
	default:
		lines = append(lines, "")
	}

	lines = append(lines, footerBarStyle.Width(clampPositive(m.width-2)).Render(
		truncate(m.help.View(m.keys), clampPositive(m.width-2))))
	return strings.Join(lines, "\n")
}
