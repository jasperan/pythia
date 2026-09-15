package tui

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// fakeClient builds a client against a fake pythia service.
func fakeClient(t *testing.T) *api.Client {
	t.Helper()
	service := fakeService{
		searchBody: "event: source\ndata: {\"index\": 1, \"title\": \"Result\", \"url\": \"https://example.test\"}\n\n" +
			"event: token\ndata: {\"content\": \"Answer text\"}\n\n" +
			"event: grounding\ndata: {\"score\": 0.75, \"label\": \"partial\", \"total_claims\": 4, \"grounded_claims\": 3}\n\n" +
			"event: done\ndata: {\"cache_hit\": true, \"similarity\": 0.93, \"response_time_ms\": 33, \"sources_count\": 1}\n\n",
		researchBody: "event: plan\ndata: {\"sub_queries\": [\"q1\"], \"slug\": \"s\"}\n\n" +
			"event: round_start\ndata: {\"round\": 1, \"max_rounds\": 3}\n\n" +
			"event: finding\ndata: {\"sub_query\": \"q1\", \"num_sources\": 2}\n\n" +
			"event: token\ndata: {\"content\": \"Report\"}\n\n" +
			"event: done\ndata: {\"rounds_used\": 1, \"total_findings\": 1, \"total_sources\": 2, \"elapsed_ms\": 500}\n\n",
		health:  `{"oracle": true, "searxng": false, "llm": true, "cache_size": 9}`,
		stats:   `{"total_searches": 10, "cache_hits": 4, "cache_hit_rate": 0.4, "avg_response_ms": 700, "active_days": 3}`,
		history: `[{"query": "q", "cache_hit": true, "response_time_ms": 12, "model_used": "m", "created_at": "x"}]`,
		skills:  `[{"name": "sk", "description": "d", "triggers": ["t"]}]`,
	}
	server := httptest.NewServer(service.handler())
	t.Cleanup(server.Close)
	return api.NewClient(server.URL)
}

// TestParseActionFlagsOrdering pins the precedence.
//
// Order matters: the most specific or destructive flag wins, so
// "--stats --clear-cache" can never silently degrade into a harmless read.
func TestParseActionFlagsOrdering(t *testing.T) {
	cases := []struct {
		name                   string
		query, research        string
		history, stats, health bool
		skills, clearCache     bool
		wantAction             string
	}{
		{name: "nothing", wantAction: ""},
		{name: "query", query: "q", wantAction: ActionQuery},
		{name: "research", research: "r", wantAction: ActionResearch},
		{name: "history", history: true, wantAction: ActionHistory},
		{name: "stats", stats: true, wantAction: ActionStats},
		{name: "health", health: true, wantAction: ActionHealth},
		{name: "skills", skills: true, wantAction: ActionSkills},
		{name: "clear beats everything", query: "q", stats: true, clearCache: true, wantAction: ActionClearCache},
		{name: "research beats query", query: "q", research: "r", wantAction: ActionResearch},
		{name: "query beats stats", query: "q", stats: true, wantAction: ActionQuery},
		{name: "whitespace query is not an action", query: "   ", stats: true, wantAction: ActionStats},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			action, ok := ParseActionFlags(testCase.query, testCase.research, testCase.history,
				testCase.stats, testCase.health, testCase.skills, testCase.clearCache)
			if action != testCase.wantAction {
				t.Errorf("action = %q, want %q", action, testCase.wantAction)
			}
			if ok != (testCase.wantAction != "") {
				t.Errorf("ok = %v, want %v", ok, testCase.wantAction != "")
			}
		})
	}
}

// TestRunQueryActionPrintsTheAnswerAndOrigin covers the headless search path.
func TestRunQueryActionPrintsTheAnswerAndOrigin(t *testing.T) {
	var out bytes.Buffer
	err := RunAction(context.Background(), fakeClient(t), ActionRequest{
		Action: ActionQuery, Query: "what is rag",
	}, &out)
	if err != nil {
		t.Fatalf("RunAction returned %v", err)
	}
	for _, want := range []string{"Answer text", "cache", "0.93", "grounding", "Result"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output is missing %q:\n%s", want, out.String())
		}
	}
}

// TestRunQueryActionJSONIsMachineReadable: --json must emit one parseable
// document, since that is the whole point of the scripted path.
func TestRunQueryActionJSONIsMachineReadable(t *testing.T) {
	var out bytes.Buffer
	err := RunAction(context.Background(), fakeClient(t), ActionRequest{
		Action: ActionQuery, Query: "q", JSON: true,
	}, &out)
	if err != nil {
		t.Fatalf("RunAction returned %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(out.Bytes(), &decoded); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out.String())
	}
	if decoded["answer"] != "Answer text" {
		t.Errorf("answer field = %v", decoded["answer"])
	}
	if decoded["cache_hit"] != true {
		t.Errorf("cache_hit = %v, want true", decoded["cache_hit"])
	}
}

// TestRunResearchActionPrintsPlanAndReport checks the headless research path
// shows the same plan the tree view does.
func TestRunResearchActionPrintsPlanAndReport(t *testing.T) {
	var out bytes.Buffer
	err := RunAction(context.Background(), fakeClient(t), ActionRequest{
		Action: ActionResearch, Query: "topic",
	}, &out)
	if err != nil {
		t.Fatalf("RunAction returned %v", err)
	}
	for _, want := range []string{"q1", "round 1/3", "Report", "1 findings", "2 sources"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output is missing %q:\n%s", want, out.String())
		}
	}
}

// TestClearCacheRefusesWithoutYes is the safety guard: clearing the semantic
// cache is irreversible, so a prompt or a pipe must never be the consent.
func TestClearCacheRefusesWithoutYes(t *testing.T) {
	var out bytes.Buffer
	err := RunAction(context.Background(), fakeClient(t), ActionRequest{
		Action: ActionClearCache, Confirmed: false,
	}, &out)
	if err == nil {
		t.Fatal("clearing the cache without --yes succeeded")
	}
	if !strings.Contains(err.Error(), "--yes") {
		t.Errorf("error = %v, want it to name the flag", err)
	}
	if out.Len() != 0 {
		t.Errorf("a refused clear still wrote output: %q", out.String())
	}
}

// TestClearCacheWithYesDeletes: with the flag it must go through to the server.
func TestClearCacheWithYesDeletes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete || r.URL.Path != "/cache" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"deleted": 7}`))
	}))
	defer server.Close()

	var out bytes.Buffer
	if err := RunAction(context.Background(), api.NewClient(server.URL), ActionRequest{
		Action: ActionClearCache, Confirmed: true,
	}, &out); err != nil {
		t.Fatalf("RunAction returned %v", err)
	}
	if !strings.Contains(out.String(), "7") {
		t.Errorf("output = %q, want the deleted count", out.String())
	}
}

// TestReadOnlyActions covers history, stats, health and skills.
func TestReadOnlyActions(t *testing.T) {
	cases := []struct {
		action string
		wants  []string
	}{
		{ActionHistory, []string{"q", "cache"}},
		{ActionStats, []string{"searches", "10", "hit rate"}},
		{ActionHealth, []string{"oracle", "up", "searxng", "down"}},
		{ActionSkills, []string{"sk", "d", "t"}},
	}
	for _, testCase := range cases {
		t.Run(testCase.action, func(t *testing.T) {
			var out bytes.Buffer
			if err := RunAction(context.Background(), fakeClient(t), ActionRequest{
				Action: testCase.action, Limit: 20,
			}, &out); err != nil {
				t.Fatalf("RunAction(%s) returned %v", testCase.action, err)
			}
			for _, want := range testCase.wants {
				if !strings.Contains(out.String(), want) {
					t.Errorf("%s output is missing %q:\n%s", testCase.action, want, out.String())
				}
			}
		})
	}
}

// TestUnknownActionIsRejected keeps a typo from silently doing nothing.
func TestUnknownActionIsRejected(t *testing.T) {
	if err := RunAction(context.Background(), fakeClient(t), ActionRequest{Action: "nope"},
		&bytes.Buffer{}); err == nil {
		t.Error("an unknown action was accepted")
	}
}
