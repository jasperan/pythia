package tui

import (
	"strings"

	"charm.land/lipgloss/v2"
)

// Palette: the 14 approved "Premium Dark-Tech" tokens.
//
// Canonical spec: docs/tui-design-tokens.md, enforced by
// scripts/tui-shot/check_palette.py. Every hex literal in this package must be
// one of these 14 -- no project-local palette may be introduced, because the
// whole point of the design system is that every owned TUI looks like the same
// product.
//
// NOTE ON PYTHIA'S OWN PALETTE: src/pythia/tui/colors.py (the Python Textual
// TUI) still carries a project-specific sage/amber palette. That palette is
// legacy, and the spec's own per-project table already lists how pythia's old
// values map onto these tokens (its old background to bg, old border blue to
// primary, old cyan accent to info, and so on). This front-end follows the
// design system rather than colors.py, which is a deliberate departure -- the
// Python TUI has not yet been re-pointed.
var (
	colBG        = lipgloss.Color("#1e1e2e") // bg
	colSurface   = lipgloss.Color("#181825") // surface
	colElevated  = lipgloss.Color("#313244") // elevated
	colHighest   = lipgloss.Color("#45475a") // highest
	colText      = lipgloss.Color("#cdd6f4") // text
	colSubtext   = lipgloss.Color("#a6adc8") // subtext
	colMuted     = lipgloss.Color("#6c7086") // muted
	colDim       = lipgloss.Color("#585b70") // dim
	colPrimary   = lipgloss.Color("#89b4fa") // primary
	colSecondary = lipgloss.Color("#cba6f7") // secondary
	colInfo      = lipgloss.Color("#89dceb") // info
	colSuccess   = lipgloss.Color("#a6e3a1") // success
	colWarning   = lipgloss.Color("#f9e2af") // warning
	colError     = lipgloss.Color("#f38ba8") // error
)

var (
	// Hierarchy (rule 8): titles bold + primary, body text, metadata subtext,
	// muted for disabled, dim for separators.
	titleStyle  = lipgloss.NewStyle().Bold(true).Foreground(colPrimary)
	textStyle   = lipgloss.NewStyle().Foreground(colText)
	mutedStyle  = lipgloss.NewStyle().Foreground(colMuted)
	dimStyle    = lipgloss.NewStyle().Foreground(colDim)
	accentStyle = lipgloss.NewStyle().Foreground(colInfo)
	recallStyle = lipgloss.NewStyle().Foreground(colSecondary)

	// Semantic styles. Rule 4: these appear only where the content genuinely is
	// a success, a warning or an error.
	successStyle = lipgloss.NewStyle().Foreground(colSuccess)
	warningStyle = lipgloss.NewStyle().Foreground(colWarning)
	errorStyle   = lipgloss.NewStyle().Bold(true).Foreground(colError)

	// answerStyle is the streaming report body. Not bold: it is long-form prose
	// and bold body text is unreadable in a terminal.
	answerStyle = lipgloss.NewStyle().Foreground(colText)

	// progressStyle is for INDETERMINATE progress -- a spinner, or a sub-query
	// being searched right now. Rule 10 assigns primary (not info) to
	// indeterminate meters, so "something is happening" never reads as
	// informational chrome.
	progressStyle = lipgloss.NewStyle().Foreground(colPrimary)

	// Panels (rule 1 + 2): rounded border, 1px everywhere, focus expressed as a
	// border-COLOUR change only -- never a thickness change. Rule 7 padding.
	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(colHighest).
			Padding(0, 1)
	focusedPanelStyle = panelStyle.BorderForeground(colPrimary)

	// Tabs (rule 9): the active mode is an accent-background segment with dark
	// text and bold; inactive tabs are muted.
	activeTabStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(colBG).
			Background(colPrimary).
			Padding(0, 1)
	inactiveTabStyle = lipgloss.NewStyle().
				Foreground(colMuted).
				Padding(0, 1)

	// Bars (rule 9). The header sits above the body and the footer below it, so
	// they take the two lowest surfaces and let the body stay on bg.
	headerBarStyle = lipgloss.NewStyle().Background(colElevated).Padding(0, 1)
	footerBarStyle = lipgloss.NewStyle().Background(colSurface).Padding(0, 1)

	// Badges. A cache hit is the fast, positive path (success); a web search is
	// informational (info). Both get dark text on the accent, per rule 9.
	badgeCacheStyle = lipgloss.NewStyle().Bold(true).Foreground(colBG).Background(colSuccess).
			Padding(0, 1)
	badgeWebStyle = lipgloss.NewStyle().Bold(true).Foreground(colBG).Background(colInfo).
			Padding(0, 1)
)

// panelChrome is the number of columns a bordered, padded panel spends on
// presentation rather than content: 1 cell of border plus 1 cell of padding on
// each side.
//
// Layout arithmetic uses it so a pane's RENDERED width equals the width that was
// budgeted for it. Guessing this number is what makes a split layout overflow.
const panelChrome = 4

// Minimum usable dimensions.
//
// A terminal narrower than this cannot show a bordered pane at all, and the
// arithmetic that lays panes out (m.width - chrome) goes negative below them. A
// prior bug in this workspace was a panic at six columns, so every dimension
// passes through clampWidth/clampHeight before it reaches lipgloss.
const (
	minWidth  = 20
	minHeight = 6
)

// clampWidth floors a width at the minimum usable size.
func clampWidth(width int) int {
	if width < minWidth {
		return minWidth
	}
	return width
}

// clampHeight floors a height at the minimum usable size.
func clampHeight(height int) int {
	if height < minHeight {
		return minHeight
	}
	return height
}

// clampPositive floors a dimension at 1.
//
// This is the last line of defence before arithmetic such as (width - chrome)/2
// produces a negative, zero or absurd value. It is separate from clampWidth
// because inner boxes are legitimately allowed to be smaller than a whole
// terminal.
func clampPositive(value int) int {
	if value < 1 {
		return 1
	}
	return value
}

// truncate shortens s to at most width cells, appending an ellipsis when it cut.
//
// It measures with lipgloss.Width so it counts display cells, not bytes: the
// research tree contains box-drawing and status glyphs that are wider than one
// byte and, in a few cases, two cells.
func truncate(s string, width int) string {
	if width <= 0 {
		return ""
	}
	if lipgloss.Width(s) <= width {
		return s
	}
	if width == 1 {
		return "…"
	}
	// Trim rune by rune from the end until the rendered width fits the budget
	// minus the ellipsis itself.
	runes := []rune(s)
	const ellipsis = "…"
	target := width - lipgloss.Width(ellipsis)
	for len(runes) > 0 && lipgloss.Width(string(runes)) > target {
		runes = runes[:len(runes)-1]
	}
	return string(runes) + ellipsis
}

// wrapHard breaks s into lines of at most width cells.
//
// It is a hard wrap on spaces where possible and mid-word only when a single
// word cannot fit. Used for the streaming answer, where a soft wrap would let a
// long URL push the pane border off screen.
func wrapHard(s string, width int) []string {
	if width < 1 {
		width = 1
	}
	var lines []string
	for _, paragraph := range strings.Split(s, "\n") {
		if paragraph == "" {
			lines = append(lines, "")
			continue
		}
		remaining := paragraph
		for lipgloss.Width(remaining) > width {
			cut := breakPoint(remaining, width)
			lines = append(lines, strings.TrimRight(remaining[:cut], " "))
			remaining = strings.TrimLeft(remaining[cut:], " ")
		}
		lines = append(lines, remaining)
	}
	return lines
}

// breakPoint returns the byte index at which to split s so the first part fits
// width cells, preferring a space boundary.
func breakPoint(s string, width int) int {
	consumed := 0
	lastSpace := -1
	for i, r := range s {
		if r == ' ' {
			lastSpace = i
		}
		consumed += lipgloss.Width(string(r))
		if consumed > width {
			if lastSpace > 0 {
				return lastSpace
			}
			return i
		}
	}
	return len(s)
}

// tabBar renders the screen switcher.
//
// Every tab width is MEASURED from the styled string rather than estimated: the
// tab styles add horizontal padding, so len(name)+2 under-counts and the bar
// overflows on exactly the terminals where it matters. A tab bar that overflows
// wraps and pushes the whole layout off the bottom of the screen.
func tabBar(screens []string, active int, width int) string {
	if width <= 0 {
		return ""
	}
	parts := make([]string, 0, len(screens))
	rendered := 0
	for i, name := range screens {
		style := inactiveTabStyle
		if i == active {
			style = activeTabStyle
		}
		piece := style.Render(" " + name + " ")
		pieceWidth := lipgloss.Width(piece)
		separator := 0
		if len(parts) > 0 {
			separator = 1
		}
		if rendered+separator+pieceWidth > width {
			break
		}
		if separator == 1 {
			parts = append(parts, " ")
		}
		parts = append(parts, piece)
		rendered += separator + pieceWidth
	}
	if len(parts) == 0 {
		return ""
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, parts...)
}
