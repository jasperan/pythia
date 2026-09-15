package tui

import (
	"encoding/json"
	"strings"
	"testing"

	"charm.land/lipgloss/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// event builds an api.Event from a payload, exactly as the wire decoder would.
func event(t *testing.T, eventType string, payload any) api.Event {
	t.Helper()
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal %s payload: %v", eventType, err)
	}
	return api.Event{Type: eventType, Data: encoded}
}

func applyAll(t *testing.T, state *ResearchState, events ...api.Event) {
	t.Helper()
	for _, e := range events {
		if err := state.ApplyEvent(e); err != nil {
			t.Fatalf("ApplyEvent(%s) returned %v", e.Type, err)
		}
	}
}

// TestResearchStateFoldsARealisticStream drives the reducer with the exact event
// sequence research.py emits for a two-round investigation, and asserts the
// resulting tree. This is the Go equivalent of the Python TUI's research_tree
// widget tests.
func TestResearchStateFoldsARealisticStream(t *testing.T) {
	var state ResearchState
	state.Reset("how does RAG scale")

	applyAll(t, &state,
		event(t, "status", api.StatusData{Message: "Searching knowledge base for related research..."}),
		event(t, "recall", api.RecallData{
			Count: 1,
			Findings: []api.RecallItem{
				{SubQuery: "prior work", Similarity: 0.91, FromQuery: "vector search"},
			},
		}),
		event(t, "status", api.StatusData{Message: "Planning research strategy..."}),
		event(t, "plan", api.PlanData{SubQueries: []string{"q1", "q2"}, Slug: "how-does-rag-scale"}),
		event(t, "round_start", api.RoundStartData{Round: 1, MaxRounds: 3, NumQuery: 2}),
		event(t, "finding", api.FindingData{SubQuery: "q1", NumSources: 3, SummaryPreview: "a", Round: 1}),
		event(t, "finding", api.FindingData{SubQuery: "q2", NumSources: 2, SummaryPreview: "b", Round: 1}),
		event(t, "gap_analysis", api.GapAnalysisData{
			Sufficient: false,
			Gaps:       []string{"q3"},
			Reasoning:  "missing latency data",
		}),
		event(t, "round_start", api.RoundStartData{Round: 2, MaxRounds: 3, NumQuery: 1}),
		event(t, "finding", api.FindingData{SubQuery: "q3", NumSources: 4, Round: 2}),
		event(t, "gap_analysis", api.GapAnalysisData{Sufficient: true}),
		event(t, "evolution", api.EvolutionData{Changes: []api.EvolutionChange{
			{Type: "contradiction", PastFinding: "old claim", NewFinding: "new claim"},
		}}),
		event(t, "token", api.TokenData{Content: "The answer "}),
		event(t, "token", api.TokenData{Content: "is 42."}),
		event(t, "done", api.ResearchDoneData{
			RoundsUsed: 2, TotalFindings: 3, TotalSources: 9, ElapsedMS: 12_500,
			Slug: "how-does-rag-scale", VerificationStatus: "verified",
		}),
	)

	if len(state.Rounds) != 2 {
		t.Fatalf("got %d rounds, want 2: a sufficient gap_analysis must not open a new round", len(state.Rounds))
	}
	if !state.Rounds[0].Complete() || !state.Rounds[1].Complete() {
		t.Error("both rounds should be complete after a done event")
	}
	if state.Rounds[1].Queries[0].SubQuery != "q3" {
		t.Errorf("round 2 query = %q, want q3 (the gap)", state.Rounds[1].Queries[0].SubQuery)
	}
	if state.Rounds[1].Prompt != "missing latency data" {
		t.Errorf("gap reasoning = %q", state.Rounds[1].Prompt)
	}
	if state.Findings != 3 || state.Sources != 9 {
		t.Errorf("findings/sources = %d/%d, want 3/9", state.Findings, state.Sources)
	}
	if state.AnswerText() != "The answer is 42." {
		t.Errorf("streamed answer = %q", state.AnswerText())
	}
	if !state.Done {
		t.Error("state is not marked done")
	}
	if state.Slug != "how-does-rag-scale" {
		t.Errorf("slug = %q", state.Slug)
	}
	if state.VerifyNote != "verified" {
		t.Errorf("verify note = %q, want the done event's verification_status", state.VerifyNote)
	}
	if len(state.Evolution) != 1 || len(state.Recall) != 1 {
		t.Errorf("evolution=%d recall=%d, want 1 and 1", len(state.Evolution), len(state.Recall))
	}
}

// TestCompleteFindingPrefersTheActiveRound is the regression guard for a
// repeated sub-query: q1 appears in round 1 and again as a gap in round 2, and
// the second finding must complete round 2's node.
func TestCompleteFindingPrefersTheActiveRound(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	applyAll(t, &state,
		event(t, "plan", api.PlanData{SubQueries: []string{"q1"}}),
		event(t, "round_start", api.RoundStartData{Round: 1, MaxRounds: 3}),
		event(t, "finding", api.FindingData{SubQuery: "q1", NumSources: 1}),
		event(t, "gap_analysis", api.GapAnalysisData{Sufficient: false, Gaps: []string{"q1"}}),
		event(t, "round_start", api.RoundStartData{Round: 2, MaxRounds: 3}),
		event(t, "finding", api.FindingData{SubQuery: "q1", NumSources: 5}),
	)

	if !state.Rounds[0].Complete() {
		t.Error("round 1's q1 should be complete")
	}
	if !state.Rounds[1].Complete() {
		t.Error("round 2's q1 should be complete, not left pending")
	}
	if state.Rounds[1].Queries[0].Sources != 5 {
		t.Errorf("round 2 source count = %d, want 5 (the later finding)", state.Rounds[1].Queries[0].Sources)
	}
	if state.Findings != 2 {
		t.Errorf("findings = %d, want 2", state.Findings)
	}
}

// TestSufficientGapAnalysisDoesNotOpenARound pins research.py:444.
func TestSufficientGapAnalysisDoesNotOpenARound(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	applyAll(t, &state,
		event(t, "plan", api.PlanData{SubQueries: []string{"q1"}}),
		event(t, "gap_analysis", api.GapAnalysisData{Sufficient: true, Gaps: []string{"ignored"}}),
	)
	if len(state.Rounds) != 1 {
		t.Errorf("got %d rounds, want 1: a sufficient result must not add a round", len(state.Rounds))
	}
}

// TestDoneSettlesUnreportedNodes stops a spinner that would otherwise run
// forever if the server errored before reporting a finding.
func TestDoneSettlesUnreportedNodes(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	applyAll(t, &state,
		event(t, "plan", api.PlanData{SubQueries: []string{"q1"}}),
		event(t, "round_start", api.RoundStartData{Round: 1, MaxRounds: 1}),
		event(t, "done", api.ResearchDoneData{RoundsUsed: 1, TotalFindings: 0}),
	)
	if state.Rounds[0].Queries[0].State != NodeComplete {
		t.Error("a node left searching after done would spin forever")
	}
}

// TestUnknownEventTypeIsIgnored keeps a future server event from breaking the
// stream; the server already sends verify/completeness_check payloads this
// front-end does not render.
func TestUnknownEventTypeIsIgnored(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	if err := state.ApplyEvent(api.Event{Type: "completeness_check", Data: json.RawMessage(`{"attempt":1}`)}); err != nil {
		t.Errorf("unknown event returned %v, want nil", err)
	}
}

// TestMalformedPayloadIsReported: a genuinely malformed payload must surface,
// because silently dropping it would show a tree that disagrees with the server.
func TestMalformedPayloadIsReported(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	err := state.ApplyEvent(api.Event{Type: "plan", Data: json.RawMessage(`{"sub_queries": "not-a-list"}`)})
	if err == nil {
		t.Error("a malformed plan payload was accepted")
	}
}

// TestRenderTreeContainsTheRealStructure checks the rendered tree, not just the
// state, since the rendering is the product surface.
func TestRenderTreeContainsTheRealStructure(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	applyAll(t, &state,
		event(t, "plan", api.PlanData{SubQueries: []string{"alpha query", "beta query"}}),
		event(t, "round_start", api.RoundStartData{Round: 1, MaxRounds: 3}),
		event(t, "finding", api.FindingData{SubQuery: "alpha query", NumSources: 3}),
	)

	rendered := RenderTree(&state, 100)
	for _, want := range []string{"Round 1/3", "alpha query", "beta query", "3 sources", "◉", "◎"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("rendered tree is missing %q:\n%s", want, rendered)
		}
	}
	// The last sub-query must use the closing branch, not a tee.
	if !strings.Contains(rendered, "└─") {
		t.Errorf("rendered tree has no closing branch:\n%s", rendered)
	}
}

// TestRenderTreeRecordsFailures: a failed finding must be visible, because
// research.py reports summary_failed and a silent gap would misrepresent
// coverage.
func TestRenderTreeRecordsFailures(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	applyAll(t, &state,
		event(t, "plan", api.PlanData{SubQueries: []string{"broken"}}),
		event(t, "finding", api.FindingData{SubQuery: "broken", SummaryFailed: true, Error: "timeout"}),
	)
	if rendered := RenderTree(&state, 80); !strings.Contains(rendered, "(failed)") {
		t.Errorf("a failed finding is not shown:\n%s", rendered)
	}
}

// TestRenderTreeNeverExceedsItsWidth is the narrow-terminal guarantee for the
// tree. A pane whose content is wider than the pane wraps and destroys the
// layout, so every line must fit.
func TestRenderTreeNeverExceedsItsWidth(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	applyAll(t, &state,
		event(t, "recall", api.RecallData{Count: 1, Findings: []api.RecallItem{
			{SubQuery: "s", Similarity: 0.8, FromQuery: strings.Repeat("verylongquery", 8)},
		}}),
		event(t, "evolution", api.EvolutionData{Changes: []api.EvolutionChange{
			{Type: "contradiction", PastFinding: strings.Repeat("contradicted", 12)},
		}}),
		event(t, "plan", api.PlanData{SubQueries: []string{strings.Repeat("longsubquery", 10)}}),
		event(t, "round_start", api.RoundStartData{Round: 1, MaxRounds: 3}),
		event(t, "finding", api.FindingData{SubQuery: strings.Repeat("longsubquery", 10), NumSources: 42}),
		event(t, "gap_analysis", api.GapAnalysisData{
			Sufficient: false,
			Gaps:       []string{"gap"},
			Reasoning:  strings.Repeat("reasoning ", 20),
		}),
	)

	for _, width := range []int{20, 21, 25, 30, 40, 80} {
		rendered := RenderTree(&state, width)
		for _, line := range strings.Split(rendered, "\n") {
			if got := lipgloss.Width(line); got > width {
				t.Errorf("width %d: line is %d cells wide: %q", width, got, line)
			}
		}
	}
}

// TestRenderProgressReportsTheRun pins the progress summary, including the
// "failed" count and verification note it must not drop.
func TestRenderProgressReportsTheRun(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	state.Findings = 4
	state.Sources = 11
	state.ElapsedMS = 12_500
	state.Failed = 1
	state.Done = true
	state.VerifyNote = "verified"

	rendered := RenderProgress(&state, 100)
	for _, want := range []string{"4 findings", "11 sources", "1 failed", "verified", "12.5s"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("progress is missing %q: %s", want, rendered)
		}
	}

	state.Done = false
	state.Status = "Planning research strategy..."
	if rendered := RenderProgress(&state, 100); !strings.Contains(rendered, "Planning research strategy") {
		t.Errorf("in-flight progress does not show the status: %s", rendered)
	}
}

func TestFormatDuration(t *testing.T) {
	cases := map[int]string{
		0:       "0s",
		250:     "250ms",
		1500:    "1.5s",
		12_500:  "12.5s",
		61_000:  "1m01s",
		120_000: "2m00s",
	}
	for input, want := range cases {
		if got := formatDuration(input); got != want {
			t.Errorf("formatDuration(%d) = %q, want %q", input, got, want)
		}
	}
}
