package tui

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"charm.land/bubbles/v2/spinner"
	tea "charm.land/bubbletea/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
	"github.com/jasperan/pythia/gotui/internal/session"
)

// fakeService is a stand-in for the pythia FastAPI app. It serves the same paths
// with the same framing so the TUI is exercised through the real api.Client.
type fakeService struct {
	searchBody   string
	researchBody string
	health       string
	stats        string
	history      string
	skills       string
}

func (f fakeService) handler() http.Handler {
	mux := http.NewServeMux()
	sse := func(body string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(body))
		}
	}
	jsonHandler := func(body string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
		}
	}
	mux.HandleFunc("/search", sse(f.searchBody))
	mux.HandleFunc("/research", sse(f.researchBody))
	mux.HandleFunc("/health", jsonHandler(f.health))
	mux.HandleFunc("/stats", jsonHandler(f.stats))
	mux.HandleFunc("/history", jsonHandler(f.history))
	mux.HandleFunc("/skills", jsonHandler(f.skills))
	return mux
}

// newViewOnlyModel builds a model with NO HTTP server behind it.
//
// Rendering tests must use this. newTestModel starts an httptest server, and a
// width sweep calls it thousands of times; that turns a millisecond render check
// into minutes of socket churn. A nil client is fine for pure layout work, and
// the "no API client" path is itself asserted by a separate test.
func newViewOnlyModel() Model {
	model := New(Options{Settings: session.Settings{BaseURL: "http://127.0.0.1:1"}})
	model.width, model.height = 80, 24
	return model
}

// newTestModel builds a model wired to a fake service with sane payloads.
func newTestModel(t *testing.T) Model {
	t.Helper()
	service := fakeService{
		searchBody: "event: status\ndata: {\"message\": \"Searching...\"}\n\n" +
			"event: source\ndata: {\"index\": 1, \"title\": \"Result\", \"url\": \"https://example.test\", \"snippet\": \"s\"}\n\n" +
			"event: token\ndata: {\"content\": \"Hello \"}\n\n" +
			"event: token\ndata: {\"content\": \"world\"}\n\n" +
			"event: grounding\ndata: {\"score\": 0.8, \"label\": \"strong\", \"total_claims\": 4, \"grounded_claims\": 3}\n\n" +
			"event: suggestions\ndata: {\"suggestions\": [\"more?\"]}\n\n" +
			"event: done\ndata: {\"cache_hit\": false, \"response_time_ms\": 120, \"sources_count\": 1}\n\n",
		researchBody: "event: status\ndata: {\"message\": \"Planning research strategy...\"}\n\n" +
			"event: plan\ndata: {\"sub_queries\": [\"q1\"], \"slug\": \"slug\"}\n\n" +
			"event: round_start\ndata: {\"round\": 1, \"max_rounds\": 3}\n\n" +
			"event: finding\ndata: {\"sub_query\": \"q1\", \"num_sources\": 2}\n\n" +
			"event: token\ndata: {\"content\": \"Report body\"}\n\n" +
			"event: done\ndata: {\"rounds_used\": 1, \"total_findings\": 1, \"total_sources\": 2, \"elapsed_ms\": 900}\n\n",
		health:  `{"oracle": true, "searxng": true, "llm": false, "cache_size": 12}`,
		stats:   `{"total_searches": 30, "cache_hits": 12, "cache_hit_rate": 0.4, "avg_response_ms": 850, "active_days": 5}`,
		history: `[{"query": "what is rag", "cache_hit": true, "response_time_ms": 40, "model_used": "qwen3.5:9b", "created_at": "2026-09-14T10:00:00"}]`,
		skills:  `[{"name": "deep-dive", "description": "Long-form investigation", "triggers": ["deep"]}]`,
	}
	server := httptest.NewServer(service.handler())
	t.Cleanup(server.Close)

	model := New(Options{
		Settings: session.Settings{BaseURL: server.URL},
		Client:   api.NewClient(server.URL),
		Model:    "qwen3.5:9b",
	})
	model.width, model.height = 100, 30
	return model
}

// drain runs a command tree to completion.
//
// It deliberately does NOT follow spinner ticks: a tick re-arms itself forever,
// so following one would make every test hang instead of asserting anything.
func drain(t *testing.T, model tea.Model, cmd tea.Cmd, depth int) tea.Model {
	t.Helper()
	if cmd == nil || depth > 64 {
		return model
	}
	msg := cmd()
	if msg == nil {
		return model
	}
	if batch, ok := msg.(tea.BatchMsg); ok {
		for _, child := range batch {
			model = drain(t, model, child, depth+1)
		}
		return model
	}
	if _, isTick := msg.(spinner.TickMsg); isTick {
		return model
	}
	next, nextCmd := model.Update(msg)
	return drain(t, next, nextCmd, depth+1)
}

// typeQuery feeds a string into the focused query box one rune at a time, which
// is what a user does and what the real key path sees.
func typeQuery(t *testing.T, model Model, text string) Model {
	t.Helper()
	for _, r := range text {
		msg := tea.KeyPressMsg{Code: r, Text: string(r)}
		next, _ := model.Update(msg)
		model = next.(Model)
	}
	return model
}

// TestSubmitStreamsTheAnswer drives the whole search flow end to end through the
// HTTP client, and asserts the screen shows what the server sent.
func TestSubmitStreamsTheAnswer(t *testing.T) {
	model := newTestModel(t)
	model = typeQuery(t, model, "what is rag")

	sent, cmd := model.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	model = drain(t, sent.(Model), cmd, 0).(Model)

	if !model.search.Done {
		t.Fatalf("search never completed; err=%v", model.err)
	}
	if got := model.search.AnswerText(); got != "Hello world" {
		t.Errorf("answer = %q, want %q", got, "Hello world")
	}
	if len(model.search.Sources) != 1 {
		t.Fatalf("sources = %d, want 1", len(model.search.Sources))
	}
	if model.search.Grounding == nil || model.search.Grounding.GroundedClaims != 3 {
		t.Errorf("grounding not recorded: %+v", model.search.Grounding)
	}
	if model.busy {
		t.Error("model is still busy after the stream finished")
	}

	view := model.View().Content
	for _, want := range []string{"Hello world", "Result", "grounding"} {
		if !strings.Contains(view, want) {
			t.Errorf("view is missing %q:\n%s", want, view)
		}
	}
}

// TestResearchStreamBuildsTheTree drives the research flow and asserts the tree
// pane, which is the feature this front-end exists for.
func TestResearchStreamBuildsTheTree(t *testing.T) {
	model := newTestModel(t)
	model = model.switchScreen(ScreenResearch)
	model = typeQuery(t, model, "how does rag scale")

	sent, cmd := model.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	model = drain(t, sent.(Model), cmd, 0).(Model)

	if !model.research.Done {
		t.Fatalf("research never completed; err=%v", model.err)
	}
	if len(model.research.Rounds) != 1 {
		t.Fatalf("rounds = %d, want 1", len(model.research.Rounds))
	}
	if model.research.Rounds[0].Queries[0].State != NodeComplete {
		t.Error("the sub-query was not marked complete")
	}

	view := model.View().Content
	for _, want := range []string{"Round 1/3", "q1", "Report body", "2 sources"} {
		if !strings.Contains(view, want) {
			t.Errorf("research view is missing %q:\n%s", want, view)
		}
	}
}

// TestTabKeySwitchesScreens: tab is always a screen switch, even while typing.
func TestTabKeySwitchesScreens(t *testing.T) {
	model := newTestModel(t)
	for _, want := range []Screen{ScreenResearch, ScreenHistory, ScreenDashboard, ScreenSearch} {
		next, _ := model.Update(tea.KeyPressMsg{Code: tea.KeyTab})
		model = next.(Model)
		if model.active != want {
			t.Fatalf("after tab, active = %v, want %v", model.active, want)
		}
	}
}

// TestNumberKeysAreTypedNotSwitched is the guard src/pythia/tui/app.py:on_key
// applies: pressing 2 while composing a query must type "2", not jump tabs.
func TestNumberKeysAreTypedNotSwitched(t *testing.T) {
	model := newTestModel(t)
	model = typeQuery(t, model, "2")
	if model.active != ScreenSearch {
		t.Errorf("a number key switched to %v while typing", model.active)
	}
	if got := model.query.Value(); got != "2" {
		t.Errorf("query = %q, want the typed digit", got)
	}

	// Once typing stops, the same key is a screen switch.
	model.typing = false
	model.query.Blur()
	next, _ := model.Update(tea.KeyPressMsg{Code: '2', Text: "2"})
	if got := next.(Model).active; got != ScreenResearch {
		t.Errorf("number key did not switch screens: active = %v", got)
	}
}

// TestEscCancelsAnInFlightStream keeps esc responsive during a long run.
func TestEscCancelsAnInFlightStream(t *testing.T) {
	// A server that never finishes the stream.
	//
	// Defer order is load-bearing. httptest.Server.Close() blocks until every
	// in-flight request handler returns, and this handler is parked on <-blocked,
	// so the channel MUST be released first. Defers run LIFO, which means
	// server.Close() has to be registered first and close(blocked) second --
	// registering them the other way deadlocks the test binary.
	blocked := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		select {
		case <-blocked:
		case <-r.Context().Done():
			// The client cancelled (esc). Release the handler so Close() can
			// finish even if the test never closes the channel.
		}
	}))
	defer server.Close()
	defer close(blocked)

	model := New(Options{Client: api.NewClient(server.URL)})
	model.width, model.height = 100, 30
	model = typeQuery(t, model, "slow")

	sent, cmd := model.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	model = sent.(Model)
	_ = cmd
	if !model.busy {
		t.Fatal("the model is not busy after submitting")
	}
	if model.cancel == nil {
		t.Fatal("no cancellable context was created; esc could not stop the stream")
	}

	next, _ := model.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
	model = next.(Model)
	if model.busy {
		t.Error("esc did not clear the busy state")
	}
	if model.cancel != nil {
		t.Error("esc left the stream context live")
	}
}

// TestEscNeverLeavesADeadEnd is the regression guard for a bug class that has
// shipped twice elsewhere in this workspace.
//
// In charm.land/huh/v2 the default keymap binds Quit to ctrl+c ONLY
// (keymap.go: Quit: key.NewBinding(key.WithKeys("ctrl+c"))), and form.go reaches
// StateAborted solely through key.Matches(msg, keymap.Quit). Escape therefore
// never aborts a huh form. A TUI that delegates "esc" to an embedded form leaves
// that form installed, and it then silently swallows every later keystroke while
// the user believes they cancelled -- a dead end with no advertised way out, and
// navigation behind it stops working.
//
// This repo embeds no form (esc is dispatched by handleKey directly -- see
// TestTheTUIOwnsEscape), so the structural preconditions are absent. What is
// still worth pinning is the observable guarantee: whatever esc does, it must
// never strand the user in a state that eats keys.
func TestEscNeverLeavesADeadEnd(t *testing.T) {
	t.Run("while a stream is running", func(t *testing.T) {
		model := newViewOnlyModel()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		// The streaming state esc has to clear. The submit-driven path that
		// populates it is covered by TestEscCancelsAnInFlightStream.
		_ = ctx
		model.busy = true
		model.cancel = cancel

		before := model.active
		next, _ := model.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
		model = next.(Model)

		if model.busy {
			t.Error("esc did not clear the busy state")
		}
		if model.cancel != nil {
			t.Error("esc left the stream context live")
		}
		if model.notice != "cancelled" {
			t.Errorf("notice = %q, want %q so the user is told why the stream stopped",
				model.notice, "cancelled")
		}
		if model.active != before {
			t.Errorf("esc moved the screen %v -> %v; cancelling must not navigate", before, model.active)
		}
		assertNavigationStillWorks(t, model)
	})

	t.Run("while typing", func(t *testing.T) {
		model := newViewOnlyModel()
		// New focuses the query box, so the model starts in typing mode.
		if !model.typing {
			t.Fatal("the model did not start in typing mode; this case would assert nothing")
		}
		before := model.active

		next, _ := model.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
		model = next.(Model)

		if model.typing {
			t.Error("esc did not leave typing mode")
		}
		if model.active != before {
			t.Errorf("esc moved the screen %v -> %v; dropping focus must not navigate", before, model.active)
		}
		assertNavigationStillWorks(t, model)
	})

	t.Run("on every screen", func(t *testing.T) {
		for screen := Screen(0); screen < Screen(len(screenNames)); screen++ {
			model := newViewOnlyModel()
			model = model.switchScreen(screen)

			next, _ := model.Update(tea.KeyPressMsg{Code: tea.KeyEscape})
			after := next.(Model)
			if after.active != screen {
				t.Errorf("%s: esc moved the screen to %v; escaping must not navigate",
					screenNames[screen], screenNames[after.active])
			}
			assertNavigationStillWorks(t, after)
		}
	})
}

// assertNavigationStillWorks proves nothing is still swallowing keys: after esc
// the user must still be able to reach the other screens.
func assertNavigationStillWorks(t *testing.T, model Model) {
	t.Helper()
	before := model.active
	next, _ := model.Update(tea.KeyPressMsg{Code: tea.KeyTab})
	after := next.(Model)
	if after.active == before {
		t.Fatalf("tab did not change screens after esc (stuck on %v); keys are being swallowed",
			screenNames[before])
	}
}

// TestTheTUIOwnsEscape pins the structural reason this repo cannot hit the
// esc-swallowing bug: the Bubble Tea model holds no huh form, so esc is always
// dispatched by handleKey and is never delegated to a form that ignores it.
//
// This is a tripwire, not a style rule. If a form is ever embedded, esc has to be
// intercepted before it is delegated (huh binds Quit to ctrl+c only), and this
// failure is the reminder to wire that up.
func TestTheTUIOwnsEscape(t *testing.T) {
	typ := reflect.TypeOf(Model{})
	for i := range typ.NumField() {
		field := typ.Field(i)
		if name := field.Type.String(); strings.Contains(name, "huh.Form") {
			t.Errorf("Model.%s is %s: an embedded huh form ignores esc "+
				"(huh binds Quit to ctrl+c only), so esc must be intercepted before delegation",
				field.Name, name)
		}
	}
}

// TestEmptyQueryIsNotSubmitted stops a stray Enter from posting an empty search.
func TestEmptyQueryIsNotSubmitted(t *testing.T) {
	model := newTestModel(t)
	next, cmd := model.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	model = next.(Model)
	if model.busy {
		t.Error("an empty query started a request")
	}
	if cmd != nil {
		t.Error("an empty query produced a command")
	}
}

// TestDashboardAndHistoryRenderLoadedData covers the two read-only tabs.
func TestDashboardAndHistoryRenderLoadedData(t *testing.T) {
	model := newTestModel(t)
	model = drain(t, model, model.loadHistoryCmd(), 0).(Model)
	model = drain(t, model, model.refreshCmd(), 0).(Model)

	history := renderHistory(model.browse, 20, 100)
	if !strings.Contains(history, "what is rag") {
		t.Errorf("history view is missing the recorded query:\n%s", history)
	}
	dashboard := renderDashboard(model.browse, 20, 100)
	for _, want := range []string{"oracle", "searxng", "ollama", "30", "deep-dive"} {
		if !strings.Contains(dashboard, want) {
			t.Errorf("dashboard is missing %q:\n%s", want, dashboard)
		}
	}
	// ollama reports false in the fixture: a degraded service must be visible,
	// not silently rendered as healthy.
	if !strings.Contains(dashboard, "degraded") {
		t.Errorf("a down service was not surfaced:\n%s", dashboard)
	}
}

// TestClearResetsScreens checks ctrl+l empties the panes.
func TestClearResetsScreens(t *testing.T) {
	model := newTestModel(t)
	model.search.appendAnswer("stale")
	model.research.Rounds = []*Round{{Number: 1, Queries: []*TreeNode{{SubQuery: "q"}}}}
	next, _ := model.Update(tea.KeyPressMsg{Code: 'l', Mod: tea.ModCtrl})
	model = next.(Model)
	if model.search.AnswerText() != "" {
		t.Errorf("ctrl+l left answer text: %q", model.search.AnswerText())
	}
	if len(model.research.Rounds) != 0 {
		t.Errorf("ctrl+l left %d research rounds", len(model.research.Rounds))
	}
}

// TestDeepToggleSwitchesTheRequestFlag pins ctrl+d, which the Python TUI also
// binds.
func TestDeepToggleSwitchesTheRequestFlag(t *testing.T) {
	model := newTestModel(t)
	next, _ := model.Update(tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl})
	model = next.(Model)
	if !model.deep {
		t.Fatal("ctrl+d did not enable deep mode")
	}
	request := model.searchRequest("q")
	if request.search == nil || !request.search.Deep {
		t.Error("deep mode was not sent on the search request")
	}
}

// TestModelWithoutClientReportsAnError rather than dereferencing nil.
func TestModelWithoutClientReportsAnError(t *testing.T) {
	model := New(Options{})
	model.width, model.height = 80, 24
	model = typeQuery(t, model, "q")
	next, _ := model.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	model = next.(Model)
	if model.err == nil {
		t.Fatal("submitting with no client produced no error")
	}
	if got := model.View().Content; !strings.Contains(got, "no API client") {
		t.Errorf("the error is not shown to the user:\n%s", got)
	}
}

// TestSpinnerTicksDoNotLoopForever documents the reason drain() must ignore
// ticks: a tick re-arms itself, so following one never terminates.
func TestSpinnerTicksDoNotLoopForever(t *testing.T) {
	model := newTestModel(t)
	if cmd := model.Init(); cmd == nil {
		t.Fatal("Init produced no commands")
	}
	next, cmd := model.Update(spinner.TickMsg{})
	if cmd == nil {
		t.Fatal("the spinner stopped re-arming; the fixture is wrong")
	}
	if next.(Model).width != model.width {
		t.Error("a spinner tick changed the layout")
	}
}
