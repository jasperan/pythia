package tui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// NodeState is the lifecycle of one sub-query in the research tree.
type NodeState int

const (
	// NodePending is planned but not yet searched.
	NodePending NodeState = iota
	// NodeSearching is being searched now.
	NodeSearching
	// NodeComplete finished, successfully or not.
	NodeComplete
)

// TreeNode is one sub-query.
type TreeNode struct {
	SubQuery string
	State    NodeState
	Sources  int
	Preview  string
	Failed   bool
}

// Round groups the sub-queries searched in one pass.
//
// A round is created by a "plan" event and again by every "gap_analysis" event
// that reports new gaps, which is why Reasoning is only set on gap rounds -- the
// same shape src/pythia/tui/widgets/research_tree.py builds.
type Round struct {
	Number  int
	Queries []*TreeNode
	Prompt  string
}

// Complete reports whether every query in the round finished.
func (r *Round) Complete() bool {
	if len(r.Queries) == 0 {
		return false
	}
	for _, query := range r.Queries {
		if query.State != NodeComplete {
			return false
		}
	}
	return true
}

// ResearchState is the whole of what the research screen displays.
//
// It is a plain reducer over the server's SSE events: every field is set from an
// event payload and nothing is derived locally, so the Go TUI cannot drift from
// what the ResearchAgent actually did. Keeping it separate from the Bubble Tea
// model makes the event handling testable without a terminal.
type ResearchState struct {
	Query string

	Rounds      []*Round
	ActiveRound int
	MaxRounds   int

	Recall    []api.RecallItem
	Evolution []api.EvolutionChange

	// answer is a POINTER on purpose. Bubble Tea's Update receives the model by
	// value, so a strings.Builder held directly in this struct would be copied
	// on every message; strings.Builder panics ("illegal use of non-zero
	// Builder copied by value") the next time it is written to.
	answer     *strings.Builder
	Findings   int
	Sources    int
	ElapsedMS  int
	Slug       string
	Status     string
	Done       bool
	VerifyNote string
	Failed     int
}

// Reset clears the state for a new research run.
func (s *ResearchState) Reset(query string) {
	*s = ResearchState{Query: query, Status: "Starting research...", answer: &strings.Builder{}}
}

// appendAnswer adds a streamed report chunk.
func (s *ResearchState) appendAnswer(chunk string) {
	if s.answer == nil {
		s.answer = &strings.Builder{}
	}
	s.answer.WriteString(chunk)
}

// AnswerText returns the report written so far.
func (s *ResearchState) AnswerText() string {
	if s.answer == nil {
		return ""
	}
	return s.answer.String()
}

// ApplyEvent folds one server event into the state.
//
// An unrecognised event type is ignored rather than treated as an error: the
// server already emits verify and completeness_check events that this front-end
// does not render, and a future event must not break the stream.
func (s *ResearchState) ApplyEvent(event api.Event) error {
	switch event.Type {
	case "status":
		var data api.StatusData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.Status = data.Message

	case "recall":
		var data api.RecallData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.Recall = data.Findings

	case "evolution":
		var data api.EvolutionData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.Evolution = data.Changes

	case "plan":
		var data api.PlanData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.Slug = data.Slug
		s.addRound(data.SubQueries, "")

	case "round_start":
		var data api.RoundStartData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.ActiveRound = data.Round
		if data.MaxRounds > 0 {
			s.MaxRounds = data.MaxRounds
		}
		s.markSearching()

	case "finding":
		var data api.FindingData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.completeFinding(data)

	case "gap_analysis":
		var data api.GapAnalysisData
		if err := event.Decode(&data); err != nil {
			return err
		}
		// A sufficient result ends the loop; only real gaps open a new round,
		// matching research.py:444.
		if !data.Sufficient && len(data.Gaps) > 0 {
			s.addRound(data.Gaps, data.Reasoning)
		}

	case "token":
		var data api.TokenData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.appendAnswer(data.Content)

	case "verify":
		s.VerifyNote = "report verified"

	case "done":
		var data api.ResearchDoneData
		if err := event.Decode(&data); err != nil {
			return err
		}
		s.Done = true
		s.Findings = data.TotalFindings
		s.Sources = data.TotalSources
		s.ElapsedMS = data.ElapsedMS
		s.Failed = data.FailedFindings
		if data.Slug != "" {
			s.Slug = data.Slug
		}
		if data.VerificationStatus != "" {
			s.VerifyNote = data.VerificationStatus
		}
		s.finishOpenNodes()
	}
	return nil
}

// addRound appends a round of pending sub-queries.
func (s *ResearchState) addRound(queries []string, reason string) {
	if len(queries) == 0 {
		return
	}
	round := &Round{Number: len(s.Rounds) + 1, Prompt: reason}
	for _, query := range queries {
		round.Queries = append(round.Queries, &TreeNode{SubQuery: query, State: NodePending})
	}
	s.Rounds = append(s.Rounds, round)
	s.ActiveRound = round.Number
}

// markSearching moves the active round's pending nodes to searching.
func (s *ResearchState) markSearching() {
	for _, round := range s.Rounds {
		if round.Number != s.ActiveRound {
			continue
		}
		for _, query := range round.Queries {
			if query.State == NodePending {
				query.State = NodeSearching
			}
		}
	}
}

// completeFinding marks the node for a sub-query complete.
//
// It matches on the sub-query text because that is the only identifier the
// server sends with a finding event (research.py:411), and it prefers the active
// round so a repeated question in a later round completes the later node.
func (s *ResearchState) completeFinding(data api.FindingData) {
	if node := s.findNode(data.SubQuery); node != nil {
		node.State = NodeComplete
		node.Sources = data.NumSources
		node.Preview = data.SummaryPreview
		node.Failed = data.SummaryFailed
	}
	s.Findings++
	s.Sources += data.NumSources
}

// findNode returns the node for a sub-query, preferring the active round and
// then the last round that still has it open.
func (s *ResearchState) findNode(subQuery string) *TreeNode {
	for i := len(s.Rounds) - 1; i >= 0; i-- {
		round := s.Rounds[i]
		if round.Number == s.ActiveRound {
			for _, query := range round.Queries {
				if query.SubQuery == subQuery && query.State != NodeComplete {
					return query
				}
			}
		}
	}
	for i := len(s.Rounds) - 1; i >= 0; i-- {
		for _, query := range s.Rounds[i].Queries {
			if query.SubQuery == subQuery && query.State != NodeComplete {
				return query
			}
		}
	}
	// Already complete: return it so the counters stay attached to the node.
	for i := len(s.Rounds) - 1; i >= 0; i-- {
		for _, query := range s.Rounds[i].Queries {
			if query.SubQuery == subQuery {
				return query
			}
		}
	}
	return nil
}

// finishOpenNodes settles any node the server never reported, so the tree does
// not keep spinning after a done event.
func (s *ResearchState) finishOpenNodes() {
	for _, round := range s.Rounds {
		for _, query := range round.Queries {
			if query.State != NodeComplete {
				query.State = NodeComplete
			}
		}
	}
}

// nodeGlyph returns the status glyph and the style KIND for a node state.
//
// The kind is a name, not a color, so the mapping from state to palette lives in
// exactly one place (renderGlyph) and a caller cannot invent a shade.
func nodeGlyph(state NodeState) (string, string) {
	switch state {
	case NodeSearching:
		// Indeterminate progress, per design rule 10.
		return "◎", "progress"
	case NodeComplete:
		return "◉", "success"
	default:
		return "○", "dim"
	}
}

func renderGlyph(state NodeState) string {
	glyph, kind := nodeGlyph(state)
	switch kind {
	case "progress":
		return progressStyle.Render(glyph)
	case "success":
		return successStyle.Render(glyph)
	default:
		return dimStyle.Render(glyph)
	}
}

// evolutionMarker maps an evolution change type to its glyph and style.
//
// The glyphs match widgets/research_tree.py:_EVOLUTION_MARKERS.
func evolutionMarker(changeType string) (string, func(...string) string) {
	switch changeType {
	case "contradiction":
		return "⚠", errorStyle.Render
	case "update":
		return "↻", accentStyle.Render
	case "confirmation":
		return "✓", successStyle.Render
	default:
		return "•", dimStyle.Render
	}
}

// RenderTree draws the research tree at the given width.
//
// It is rendered by hand rather than through a tree component because each node
// carries its own state colour and the structure grows one node at a time as
// events arrive; the glyphs and branch characters are kept identical to
// widgets/research_tree.py so the two front-ends look like the same product.
func RenderTree(state *ResearchState, width int) string {
	if state == nil {
		return ""
	}
	// The pane has a border and a one-cell gutter on each side.
	inner := clampPositive(clampWidth(width) - 4)

	var out strings.Builder

	if len(state.Evolution) > 0 {
		out.WriteString(textStyle.Bold(true).Render(truncate("📈 Knowledge evolution", inner)) + "\n")
		for i, change := range state.Evolution {
			if i >= 5 {
				break
			}
			glyph, style := evolutionMarker(change.Type)
			line := style(glyph + " " + truncate(change.PastFinding, inner-4))
			out.WriteString("  " + line + "\n")
		}
		out.WriteString("\n")
	}

	if len(state.Recall) > 0 {
		header := "🧠 Recalled " + fmt.Sprint(len(state.Recall)) + " prior finding(s)"
		out.WriteString(recallStyle.Render(truncate(header, inner)) + "\n")
		for _, item := range state.Recall {
			label := truncate(item.FromQuery, inner-8)
			out.WriteString("  " + dimStyle.Render("└ ") + mutedStyle.Render(label) +
				dimStyle.Render(fmt.Sprintf(" (%.0f%%)", item.Similarity*100)) + "\n")
		}
		out.WriteString("\n")
	}

	if len(state.Rounds) == 0 {
		if state.Status != "" {
			out.WriteString(mutedStyle.Render(truncate(state.Status, inner)) + "\n")
		}
		// Returning the builder's text rather than the wrapped output keeps a
		// narrow pane from rendering a stray border.
		return strings.TrimRight(out.String(), "\n")
	}

	total := state.MaxRounds
	if total < len(state.Rounds) {
		total = len(state.Rounds)
	}

	for _, round := range state.Rounds {
		isActive := round.Number == state.ActiveRound
		label := fmt.Sprintf("Round %d/%d", round.Number, total)
		switch {
		case isActive && !round.Complete():
			out.WriteString(textStyle.Bold(true).Render("  ● "+label) + "\n")
		case round.Complete():
			out.WriteString(mutedStyle.Render("  ● "+label) + "\n")
		default:
			out.WriteString(dimStyle.Render("  ○ "+label) + "\n")
		}

		for i, query := range round.Queries {
			branch := "├─"
			if i == len(round.Queries)-1 {
				branch = "└─"
			}
			out.WriteString("  " + dimStyle.Render(branch+" ") + renderGlyph(query.State) + " ")

			suffix := ""
			switch {
			case query.Failed:
				suffix = " (failed)"
			case query.State == NodeComplete && query.Sources > 0:
				suffix = fmt.Sprintf(" (%d sources)", query.Sources)
			case query.State == NodeSearching:
				suffix = " searching…"
			}

			// Truncate the query, not the suffix: the source count is the
			// progress signal and must survive a narrow pane.
			budget := inner - 6 - lipgloss.Width(suffix)
			out.WriteString(styledNodeLabel(query, budget) + dimStyle.Render(suffix))
			out.WriteString("\n")
		}

		if round.Prompt != "" && round.Number > 1 {
			out.WriteString("    " + dimStyle.Render("ℹ "+truncate(round.Prompt, inner-6)) + "\n")
		}
		out.WriteString("\n")
	}

	return strings.TrimRight(out.String(), "\n")
}

// styledNodeLabel renders a sub-query in the colour of its state.
func styledNodeLabel(node *TreeNode, budget int) string {
	label := truncate(node.SubQuery, budget)
	switch {
	case node.Failed:
		return errorStyle.Render(label)
	case node.State == NodeSearching:
		return progressStyle.Render(label)
	case node.State == NodeComplete:
		return successStyle.Render(label)
	default:
		return dimStyle.Render(label)
	}
}

// RenderProgress renders the one-line progress summary that sits under the tree.
func RenderProgress(state *ResearchState, width int) string {
	if state == nil {
		return ""
	}
	inner := clampPositive(clampWidth(width) - 2)

	if state.Done {
		summary := fmt.Sprintf("✓ %d findings · %d sources · %s",
			state.Findings, state.Sources, formatDuration(state.ElapsedMS))
		if state.Failed > 0 {
			summary += fmt.Sprintf(" · %d failed", state.Failed)
		}
		if state.VerifyNote != "" {
			summary += " · " + state.VerifyNote
		}
		return successStyle.Render(truncate(summary, inner))
	}
	if state.Status != "" {
		return mutedStyle.Render(truncate("◎ "+state.Status, inner))
	}
	return dimStyle.Render(truncate("◎ working…", inner))
}

// formatDuration renders milliseconds as the Python TUI's progress bar does.
func formatDuration(ms int) string {
	switch {
	case ms <= 0:
		return "0s"
	case ms < 1000:
		return fmt.Sprintf("%dms", ms)
	case ms < 60_000:
		return fmt.Sprintf("%.1fs", float64(ms)/1000)
	default:
		return fmt.Sprintf("%dm%02ds", ms/60000, (ms%60000)/1000)
	}
}
