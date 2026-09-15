package tui

import (
	"strings"
	"testing"

	"charm.land/lipgloss/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// This file pins the narrow-terminal behaviour.
//
// A prior bug in this workspace was a panic at a terminal narrower than six
// columns, where layout arithmetic such as (width-6)/2 went negative and was
// handed to a styling function that could not accept it. Every dimension in
// this package is therefore clamped before use, and this test proves the whole
// view renders at every size rather than trusting the arithmetic by eye.

// TestViewRendersAtEverySupportedSize sweeps widths and heights, including
// absurdly small ones, and asserts the view is produced without a panic.
func TestViewRendersAtEverySupportedSize(t *testing.T) {
	for width := 1; width <= 120; width += 1 {
		for _, height := range []int{1, 2, 3, 4, 5, 6, 8, 12, 24, 40} {
			model := newViewOnlyModel()
			model.width, model.height = width, height
			for _, screen := range []Screen{ScreenSearch, ScreenResearch, ScreenHistory, ScreenDashboard} {
				model.active = screen
				view := model.View().Content
				if view == "" && height > 0 {
					t.Fatalf("width %d height %d screen %v rendered nothing", width, height, screen)
				}
			}
		}
	}
}

// TestViewFitsTheTerminalWidth is the overflow guard: a line wider than the
// terminal wraps and pushes the footer off the bottom of the screen, which makes
// the TUI unusable rather than merely ugly.
func TestViewFitsTheTerminalWidth(t *testing.T) {
	for _, width := range []int{20, 21, 25, 34, 40, 60, 69, 70, 71, 100, 140} {
		for _, screen := range []Screen{ScreenSearch, ScreenResearch, ScreenHistory, ScreenDashboard} {
			model := newViewOnlyModel()
			model.width, model.height = width, 30
			model.active = screen
			model = typeQuery(t, model, "a fairly long question about retrieval augmented generation")

			// Populate every pane with long, realistic content so the layout is
			// measured under load, not on an empty screen.
			model.search.appendAnswer(strings.Repeat("Long answer prose with details. ", 40))
			model.search.Sources = []api.Source{
				{Index: 1, Title: strings.Repeat("source title ", 5), URL: "https://example.test/very/long/path"},
				{Index: 2, Title: "second", URL: "https://example.test/two"},
			}
			model.search.Suggestions = []string{"A very long follow-up suggestion that must be elided"}
			model.research.Reset("a fairly long research question about scaling")
			model.research.Rounds = []*Round{{
				Number: 1,
				Queries: []*TreeNode{
					{SubQuery: strings.Repeat("sub query ", 8), State: NodeComplete, Sources: 12},
					{SubQuery: strings.Repeat("another ", 10), State: NodeSearching},
				},
				Prompt: strings.Repeat("reasoning ", 10),
			}}
			model.research.MaxRounds = 3
			model.browse.loaded = true
			model.browse.history = []api.HistoryItem{
				{Query: strings.Repeat("historical question ", 6), CacheHit: true, ResponseTimeMS: 40, ModelUsed: "qwen3.5:9b"},
			}
			model.browse.skills = []api.Skill{{Name: "deep-dive", Description: strings.Repeat("long description ", 6)}}
			model.browse.stats = api.Stats{TotalSearches: 999, CacheHits: 500, CacheHitRate: 0.5, AvgResponseMS: 850}
			model.browse.health = api.Health{Oracle: true, Searxng: false, LLM: true, CacheSize: 42}

			for index, line := range strings.Split(model.View().Content, "\n") {
				if got := lipgloss.Width(line); got > width {
					t.Errorf("width %d screen %v: line %d is %d cells wide: %q",
						width, screen, index, got, line)
				}
			}
		}
	}
}

// TestClampGuards pins the helpers themselves, including the negative inputs
// that caused the original panic.
func TestClampGuards(t *testing.T) {
	for _, input := range []int{-100, -1, 0, 1, minWidth - 1} {
		if got := clampWidth(input); got != minWidth {
			t.Errorf("clampWidth(%d) = %d, want %d", input, got, minWidth)
		}
	}
	if got := clampWidth(200); got != 200 {
		t.Errorf("clampWidth(200) = %d, want 200 (a wide terminal must not be shrunk)", got)
	}

	for _, input := range []int{-100, -1, 0, 1, minHeight - 1} {
		if got := clampHeight(input); got != minHeight {
			t.Errorf("clampHeight(%d) = %d, want %d", input, got, minHeight)
		}
	}

	for _, input := range []int{-100, -1, 0} {
		if got := clampPositive(input); got != 1 {
			t.Errorf("clampPositive(%d) = %d, want 1", input, got)
		}
	}
	if got := clampPositive(7); got != 7 {
		t.Errorf("clampPositive(7) = %d, want 7", got)
	}
}

// TestTruncateRespectsDisplayWidth checks that elision measures cells, not
// bytes: the tree contains box-drawing and status glyphs that are wider than one
// byte, and the emoji headers are two cells each.
func TestTruncateRespectsDisplayWidth(t *testing.T) {
	cases := []struct {
		input string
		width int
	}{
		{"plain ascii text", 5},
		{"├─ ◉ sub-query with glyphs", 8},
		{"📈 Knowledge evolution", 6},
		{"a", 1},
	}
	for _, testCase := range cases {
		got := truncate(testCase.input, testCase.width)
		if width := lipgloss.Width(got); width > testCase.width {
			t.Errorf("truncate(%q, %d) = %q (%d cells), exceeds the budget",
				testCase.input, testCase.width, got, width)
		}
	}

	// Short enough input is returned untouched.
	if got := truncate("short", 20); got != "short" {
		t.Errorf("truncate padded or altered a short string: %q", got)
	}
	// A zero or negative budget must not panic.
	if got := truncate("anything", 0); got != "" {
		t.Errorf("truncate(width 0) = %q, want empty", got)
	}
	if got := truncate("anything", -5); got != "" {
		t.Errorf("truncate(width -5) = %q, want empty", got)
	}
}

// TestWrapHardFitsAndLosesNothing checks the answer wrapper, which must not drop
// characters: a report is the whole point of a research run.
func TestWrapHardFitsAndLosesNothing(t *testing.T) {
	for _, width := range []int{1, 2, 5, 10, 40} {
		source := "The report mentions https://a-very-long-url.example.test/path and more prose."
		lines := wrapHard(source, width)
		for _, line := range lines {
			if got := lipgloss.Width(line); got > width {
				t.Errorf("width %d: wrapped line is %d cells: %q", width, got, line)
			}
		}
		// Every non-space character must survive the wrap.
		joined := strings.Join(lines, "")
		want := strings.ReplaceAll(source, " ", "")
		if strings.ReplaceAll(joined, " ", "") != want {
			t.Errorf("width %d: wrapping lost characters\n got %q\nwant %q",
				width, joined, source)
		}
	}

	// Newlines are preserved as paragraph breaks.
	if lines := wrapHard("first\nsecond", 40); len(lines) != 2 {
		t.Errorf("wrapHard lost a newline: %v", lines)
	}
}

// TestTabBarElidesOnNarrowTerminals: the tab bar must never be the reason the
// header overflows.
func TestTabBarElidesOnNarrowTerminals(t *testing.T) {
	for width := 1; width <= 60; width++ {
		bar := tabBar(screenNames, 0, width)
		if got := lipgloss.Width(bar); got > width {
			t.Errorf("tabBar width %d rendered %d cells: %q", width, got, bar)
		}
	}
	// On a wide terminal every tab is present.
	wide := tabBar(screenNames, 0, 120)
	for _, name := range screenNames {
		if !strings.Contains(wide, name) {
			t.Errorf("wide tab bar is missing %q: %q", name, wide)
		}
	}
}

// TestWindowLinesClampsStart covers scrolling past the end, which must show the
// last page rather than an empty pane.
func TestWindowLinesClampsStart(t *testing.T) {
	lines := []string{"a", "b", "c", "d", "e"}
	if got := windowLines(lines, 2, 99); len(got) != 2 || got[1] != "e" {
		t.Errorf("windowLines past the end = %v, want the last page", got)
	}
	if got := windowLines(lines, 2, -5); len(got) != 2 || got[0] != "a" {
		t.Errorf("windowLines with a negative offset = %v, want the first page", got)
	}
	if got := windowLines(lines, 0, 0); got != nil {
		t.Errorf("windowLines with no rows = %v, want nil", got)
	}
}

// TestFitLinesProducesAFixedHeight keeps the footer from jumping as the answer
// streams in.
func TestFitLinesProducesAFixedHeight(t *testing.T) {
	for _, rows := range []int{0, 1, 3, 10} {
		got := fitLines([]string{"one", "two"}, rows)
		if rows == 0 {
			continue
		}
		if count := strings.Count(got, "\n") + 1; count != rows {
			t.Errorf("fitLines(rows=%d) produced %d lines: %q", rows, count, got)
		}
	}
	// Over-long input is trimmed.
	got := fitLines([]string{"1", "2", "3", "4"}, 2)
	if count := strings.Count(got, "\n") + 1; count != 2 {
		t.Errorf("fitLines trimmed to %d lines, want 2: %q", count, got)
	}
}
