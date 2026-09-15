package tui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// renderHistory draws the recent-query list.
//
// A cache hit is shown with the same wording the Python TUI uses ("cache" vs
// "web") because it is the one piece of information that explains why an
// identical query returned instantly.
//
// width is the real terminal width: every line is truncated to it, because
// lipgloss pads a joined block to its widest line and a single over-long row
// would make the whole body wider than the terminal.
func renderHistory(state browseState, height, width int) string {
	width = clampPositive(width)
	lines := make([]string, 0, height)

	if !state.loaded {
		lines = append(lines, dimStyle.Render(truncate("loading history…", width)))
		return fitLines(lines, height)
	}
	if len(state.history) == 0 {
		lines = append(lines, dimStyle.Render(truncate("No searches recorded yet.", width)))
		lines = append(lines, dimStyle.Render(truncate("Run a search and it will appear here.", width)))
		return fitLines(lines, height)
	}

	lines = append(lines, titleStyle.Render(truncate(
		fmt.Sprintf("Recent searches (%d)", len(state.history)), width)))
	for _, item := range state.history {
		if len(lines) >= height {
			break
		}
		badge := badgeWebStyle.Render(" web ")
		if item.CacheHit {
			badge = badgeCacheStyle.Render(" cache ")
		}
		// The badge and the metadata are fixed-width chrome, so the query gets
		// whatever is left rather than a guessed constant.
		chrome := 7 + 2 + lipgloss.Width(formatDuration(item.ResponseTimeMS)) +
			3 + lipgloss.Width(item.ModelUsed) + 2
		query := textStyle.Render(truncate(item.Query, clampPositive(width-chrome)))
		meta := dimStyle.Render(fmt.Sprintf("%s · %s",
			formatDuration(item.ResponseTimeMS), item.ModelUsed))
		lines = append(lines, truncate(badge+" "+query+"  "+meta, width))
	}
	return fitLines(lines, height)
}

// renderDashboard draws service health, cache statistics and available skills.
func renderDashboard(state browseState, height, width int) string {
	width = clampPositive(width)
	lines := make([]string, 0, height)

	if !state.loaded {
		lines = append(lines, dimStyle.Render(truncate("loading dashboard…", width)))
		return fitLines(lines, height)
	}

	lines = append(lines, titleStyle.Render(truncate("Services", width)))
	lines = append(lines, renderHealthRow(state.health, width)...)
	lines = append(lines, "")

	lines = append(lines, titleStyle.Render(truncate("Semantic cache", width)))
	stats := state.stats
	lines = append(lines, textStyle.Render(truncate(fmt.Sprintf(
		"  searches %d · hits %d · hit rate %.0f%% · avg %s",
		stats.TotalSearches, stats.CacheHits, stats.CacheHitRate*100,
		formatDuration(stats.AvgResponseMS)), width)))
	lines = append(lines, dimStyle.Render(truncate(fmt.Sprintf(
		"  cached answers %d · active days %d", state.health.CacheSize, stats.ActiveDays), width)))
	lines = append(lines, "")

	if len(state.skills) > 0 {
		lines = append(lines, titleStyle.Render(truncate(
			fmt.Sprintf("Skills (%d)", len(state.skills)), width)))
		for _, skill := range state.skills {
			if len(lines) >= height {
				break
			}
			lines = append(lines, textStyle.Render(truncate("  "+skill.Name, width)))
			lines = append(lines, dimStyle.Render(truncate("    "+skill.Description, width)))
		}
	} else {
		lines = append(lines, dimStyle.Render(truncate("No research skills registered.", width)))
	}

	return fitLines(lines, height)
}

// renderHealthRow renders the three backing services the server reports.
//
// Each service is named rather than counted, because "2/3 degraded" does not
// tell a user whether to start Docker or to check Ollama. It returns a slice
// because the degraded hint is a second line.
func renderHealthRow(health api.Health, width int) []string {
	type service struct {
		name string
		up   bool
	}
	services := []service{
		{"oracle", health.Oracle},
		{"searxng", health.Searxng},
		{"ollama", health.LLM},
	}
	parts := make([]string, 0, len(services))
	for _, item := range services {
		if item.up {
			parts = append(parts, successStyle.Render("● "+item.name))
			continue
		}
		parts = append(parts, errorStyle.Render("○ "+item.name))
	}
	out := []string{truncate("  "+strings.Join(parts, "  "), width)}
	if !health.Healthy() {
		out = append(out, dimStyle.Render(truncate(
			"  degraded: start backends with `docker compose up -d` and ensure Ollama is running",
			width)))
	}
	return out
}
