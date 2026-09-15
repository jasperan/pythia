package tui

import (
	"strings"

	"charm.land/lipgloss/v2"
)

// researchSplitWidth is the terminal width at which the research screen splits
// into a tree pane and a report pane.
//
// Below it the panes stack instead: two bordered panes side by side under about
// 70 columns leaves each one too narrow to show a sub-query and its source
// count without wrapping.
const researchSplitWidth = 70

// treePaneWidth is the preferred width of the research-tree pane.
const treePaneWidth = 38

// renderResearch draws the research tree beside the streaming report.
func (m Model) renderResearch(height int) string {
	state := &m.research
	width := clampPositive(m.width - 2)

	tree := RenderTree(state, treePaneWidth)
	progress := RenderProgress(state, width)

	// Reserve two rows: the progress line and one blank separator.
	panesHeight := clampPositive(height - 2)

	report := m.renderReport(panesHeight, width)

	// Each pane's Width() is its CONTENT width, so the outer width that was
	// budgeted for it equals content + panelChrome. Getting this wrong is what
	// makes a split layout overflow its terminal.
	contentRows := clampPositive(panesHeight - panelChrome)

	var body string
	if m.width >= researchSplitWidth {
		leftOuter := clampPositive(treePaneWidth)
		rightOuter := clampPositive(width - leftOuter - 1)
		left := focusedPanelStyle.
			Width(clampPositive(leftOuter - panelChrome)).
			Height(contentRows).
			Render(clipToRows(tree, contentRows))
		right := panelStyle.
			Width(clampPositive(rightOuter - panelChrome)).
			Height(contentRows).
			Render(clipToRows(report, contentRows))
		body = lipgloss.JoinHorizontal(lipgloss.Top, left, " ", right)
	} else {
		// Stacked: the tree keeps about a third of the space so the report stays
		// readable, and both get real content rather than being cut off.
		treeRows := clampPositive(panesHeight/3 - 1)
		reportRows := clampPositive(panesHeight - treeRows - 1)
		left := panelStyle.
			Width(clampPositive(width - panelChrome)).
			Height(treeRows).
			Render(clipToRows(tree, treeRows))
		right := panelStyle.
			Width(clampPositive(width - panelChrome)).
			Height(reportRows).
			Render(clipToRows(report, reportRows))
		body = lipgloss.JoinVertical(lipgloss.Left, left, right)
	}

	return body + "\n" + progress
}

// renderReport renders the streaming report pane content.
func (m Model) renderReport(rows, width int) string {
	state := &m.research

	if len(state.Rounds) == 0 && strings.TrimSpace(state.AnswerText()) == "" {
		hint := "Deep research runs several rounds of search, then writes a cited report here."
		return dimStyle.Render(truncate(hint, clampPositive(width-6)))
	}

	lines := wrapHard(state.AnswerText(), clampPositive(width-6))
	if len(lines) == 0 {
		return dimStyle.Render("waiting for the report…")
	}
	visible := windowLines(lines, rows, m.scrollTop)
	return answerStyle.Render(strings.Join(visible, "\n"))
}

// clipToRows trims a rendered block to at most rows lines.
//
// A lipgloss pane given content taller than its Height still renders the extra
// lines, which pushes the footer off screen; clipping here is what keeps the
// layout fixed at every terminal size.
func clipToRows(content string, rows int) string {
	if rows <= 0 {
		return ""
	}
	lines := strings.Split(content, "\n")
	if len(lines) > rows {
		lines = lines[:rows]
	}
	return strings.Join(lines, "\n")
}
