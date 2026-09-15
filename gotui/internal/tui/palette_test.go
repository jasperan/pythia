package tui

import (
	"strings"
	"testing"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// This file pins the design-system mapping in CODE, not just in the palette.
//
// scripts/tui-shot/check_palette.py proves no off-token hex exists in the source.
// It cannot prove a token is used for the right MEANING: swapping the spinner
// from primary to error would pass the gate and be wrong. These tests assert the
// semantic assignments, expressed as 24-bit ANSI sequences so they contain no hex
// literal of their own.
//
// Approved tokens as 24-bit SGR payloads, from docs/tui-design-tokens.md.
//
// These are matched as SUBSTRINGS, not whole escape sequences: lipgloss merges
// attributes into a single SGR, so a bold style emits "1;38;2;R;G;B" rather than
// a bold escape followed by a colour escape. Matching the full
// "ESC[38;2;R;G;Bm" form would fail on every bold style and pass only by luck on
// plain ones.
//
// The names say which channel and the comments give the token, so a reader does
// not have to decode RGB triples by hand.
var (
	fgBG      = "38;2;30;30;46"    // bg       foreground
	fgHighest = "38;2;69;71;90"    // highest  foreground
	fgText    = "38;2;205;214;244" // text     foreground
	fgMuted   = "38;2;108;112;134" // muted    foreground
	fgDim     = "38;2;88;91;112"   // dim      foreground
	fgPrimary = "38;2;137;180;250" // primary  foreground
	fgInfo    = "38;2;137;220;235" // info     foreground
	fgSuccess = "38;2;166;227;161" // success  foreground
	fgWarning = "38;2;249;226;175" // warning  foreground
	fgError   = "38;2;243;139;168" // error    foreground

	bgElevated = "48;2;49;50;68"    // elevated background
	bgSurface  = "48;2;24;24;37"    // surface  background
	bgPrimary  = "48;2;137;180;250" // primary  background
	bgSuccess  = "48;2;166;227;161" // success  background
	bgInfo     = "48;2;137;220;235" // info     background
)

// TestBarsAndBordersUseTheSurfaceTokens checks the layered chrome: the header
// takes elevated, the footer takes surface, and an unfocused panel border takes
// highest (rule 2 says unfocused borders are highest/dim, never primary).
func TestBarsAndBordersUseTheSurfaceTokens(t *testing.T) {
	model := newViewOnlyModel()
	model.width, model.height = 100, 30
	model.browse.loaded = true

	if header := model.renderHeader(); !strings.Contains(header, bgElevated) {
		t.Errorf("the header bar is not on the elevated surface: %q", header)
	}
	if footer := model.renderFooter(); !strings.Contains(footer, bgSurface) {
		t.Errorf("the footer bar is not on the surface token: %q", footer)
	}

	panel := panelStyle.Width(10).Render("x")
	if !strings.Contains(panel, fgHighest) {
		t.Errorf("an unfocused panel border is not highest: %q", panel)
	}
	if strings.Contains(panel, fgPrimary) {
		t.Errorf("an unfocused panel border is primary, so focus is not distinguishable: %q", panel)
	}
	// Focus is a border-COLOUR change to primary, and nothing else changes.
	if focused := focusedPanelStyle.Width(10).Render("x"); !strings.Contains(focused, fgPrimary) {
		t.Errorf("a focused panel border is not primary: %q", focused)
	}
}

// TestNodeGlyphStateMapping pins every state to its glyph and semantic kind.
//
// It matters because renderGlyph switches on the KIND string: if a state returned
// a kind that renderGlyph does not handle it would silently fall through to dim,
// which looks deliberate but communicates nothing.
func TestNodeGlyphStateMapping(t *testing.T) {
	cases := []struct {
		state     NodeState
		wantGlyph string
		wantKind  string
		wantANSI  string
	}{
		{NodePending, "○", "dim", fgDim},
		{NodeSearching, "◎", "progress", fgPrimary},
		{NodeComplete, "◉", "success", fgSuccess},
	}
	for _, testCase := range cases {
		glyph, kind := nodeGlyph(testCase.state)
		if glyph != testCase.wantGlyph {
			t.Errorf("nodeGlyph(%v) glyph = %q, want %q", testCase.state, glyph, testCase.wantGlyph)
		}
		if kind != testCase.wantKind {
			t.Errorf("nodeGlyph(%v) kind = %q, want %q", testCase.state, kind, testCase.wantKind)
		}
		rendered := renderGlyph(testCase.state)
		if !strings.Contains(rendered, testCase.wantANSI) {
			t.Errorf("renderGlyph(%v) = %q, want it coloured with %q",
				testCase.state, rendered, testCase.wantKind)
		}
	}
}

// TestSearchingUsesPrimaryNotInfo is design rule 10: an indeterminate meter is
// primary, so "in progress" never reads as informational chrome.
func TestSearchingUsesPrimaryNotInfo(t *testing.T) {
	rendered := renderGlyph(NodeSearching)
	if !strings.Contains(rendered, fgPrimary) {
		t.Errorf("the searching glyph is not primary: %q", rendered)
	}
	if strings.Contains(rendered, fgInfo) {
		t.Errorf("the searching glyph is info; rule 10 assigns primary to "+
			"indeterminate progress: %q", rendered)
	}
}

// TestGroundingThresholdIsSemantic checks that the grounding badge uses success
// when the answer is well grounded and warning when it is not -- the one place
// this front-end is allowed to use the warning token.
func TestGroundingThresholdIsSemantic(t *testing.T) {
	renderGrounding := func(score float64) string {
		model := newViewOnlyModel()
		model.width, model.height = 100, 30
		model.search.Grounding = &api.GroundingData{
			Score: score, Label: "x", TotalClaims: 4, GroundedClaims: 3,
		}
		return model.renderSearch(20)
	}

	strong := renderGrounding(0.9)
	if !strings.Contains(strong, fgSuccess) {
		t.Errorf("a well-grounded answer is not marked success: %q", strong)
	}
	if strings.Contains(strong, fgWarning) {
		t.Errorf("a well-grounded answer is marked warning: %q", strong)
	}

	weak := renderGrounding(0.2)
	if !strings.Contains(weak, fgWarning) {
		t.Errorf("a poorly-grounded answer is not marked warning: %q", weak)
	}
	if strings.Contains(weak, fgSuccess) {
		t.Errorf("a poorly-grounded answer is marked success: %q", weak)
	}
}

// TestFailedFindingsUseErrorOnlyForFailures keeps the error token semantic.
func TestFailedFindingsUseErrorOnlyForFailures(t *testing.T) {
	var state ResearchState
	state.Reset("q")
	applyAll(t, &state,
		event(t, "plan", api.PlanData{SubQueries: []string{"ok", "broken"}}),
		event(t, "finding", api.FindingData{SubQuery: "ok", NumSources: 1}),
		event(t, "finding", api.FindingData{SubQuery: "broken", SummaryFailed: true}),
	)

	rendered := RenderTree(&state, 100)
	if !strings.Contains(rendered, fgError) {
		t.Errorf("a failed finding is not shown in error colour: %q", rendered)
	}
	// A healthy tree must not contain error colour at all.
	var healthy ResearchState
	healthy.Reset("q")
	applyAll(t, &healthy,
		event(t, "plan", api.PlanData{SubQueries: []string{"ok"}}),
		event(t, "finding", api.FindingData{SubQuery: "ok", NumSources: 1}),
	)
	if clean := RenderTree(&healthy, 100); strings.Contains(clean, fgError) {
		t.Errorf("a healthy tree contains error colour, so error is decorative: %q", clean)
	}
}

// TestNoPureBlackOrWhite is design rule 5.
func TestNoPureBlackOrWhite(t *testing.T) {
	model := newViewOnlyModel()
	model.width, model.height = 100, 30
	model.browse.loaded = true
	model.browse.health = api.Health{Oracle: true, Searxng: true, LLM: true}

	view := model.View().Content
	for _, forbidden := range []string{
		"38;2;0;0;0", "48;2;0;0;0",
		"38;2;255;255;255", "48;2;255;255;255",
	} {
		if strings.Contains(view, forbidden) {
			t.Errorf("view uses pure black or white: %q", forbidden)
		}
	}
}

// TestSelectionsAreInverted is design rule 6: selection is an accent background
// with dark text and bold.
func TestSelectionsAreInverted(t *testing.T) {
	tab := activeTabStyle.Render("x")
	if !strings.Contains(tab, fgBG) {
		t.Errorf("the active tab does not use dark bg text: %q", tab)
	}
	if !strings.Contains(tab, bgPrimary) {
		t.Errorf("the active tab does not use a primary background: %q", tab)
	}

	// Each badge must pair dark text with its own accent background.
	for name, testCase := range map[string]struct {
		render func(...string) string
		bg     string
	}{
		"cache badge": {badgeCacheStyle.Render, bgSuccess},
		"web badge":   {badgeWebStyle.Render, bgInfo},
	} {
		got := testCase.render("x")
		if !strings.Contains(got, fgBG) {
			t.Errorf("%s does not use dark text on its accent background: %q", name, got)
		}
		if !strings.Contains(got, testCase.bg) {
			t.Errorf("%s does not use its accent background: %q", name, got)
		}
	}
}

// TestFrameUsesRoundedBorders is design rule 1.
func TestFrameUsesRoundedBorders(t *testing.T) {
	rendered := panelStyle.Width(10).Render("x")
	if !strings.Contains(rendered, "╭") || !strings.Contains(rendered, "╯") {
		t.Errorf("panels are not drawn with rounded borders: %q", rendered)
	}
	for _, double := range []string{"╔", "╗", "╚", "╝", "═"} {
		if strings.Contains(rendered, double) {
			t.Errorf("panel uses a double border (%q): %q", double, rendered)
		}
	}
}

// TestStyleHelpersProduceTheTokens documents the palette itself, so a token swap
// has to be a deliberate edit here too.
func TestStyleHelpersProduceTheTokens(t *testing.T) {
	cases := []struct {
		name  string
		style func(...string) string
		want  string
	}{
		{"title", titleStyle.Render, fgPrimary},
		{"text", textStyle.Render, fgText},
		{"muted", mutedStyle.Render, fgMuted},
		{"dim", dimStyle.Render, fgDim},
		{"success", successStyle.Render, fgSuccess},
		{"warning", warningStyle.Render, fgWarning},
		{"error", errorStyle.Render, fgError},
		{"info", accentStyle.Render, fgInfo},
		{"progress", progressStyle.Render, fgPrimary},
	}
	for _, testCase := range cases {
		if got := testCase.style("x"); !strings.Contains(got, testCase.want) {
			t.Errorf("%s style = %q, want it to carry %q", testCase.name, got, testCase.want)
		}
	}
	// Panel chrome accounting must match the style, or split layouts overflow.
	if panelChrome != 4 {
		t.Errorf("panelChrome = %d; the split layout math assumes 4 "+
			"(1 border + 1 padding per side)", panelChrome)
	}
}
