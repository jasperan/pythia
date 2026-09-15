package tui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// apply folds one search SSE event into the screen state.
//
// The event names are the SearchEventType values from src/pythia/server/search.py.
func (s *searchState) apply(event api.Event) {
	switch event.Type {
	case "status":
		var data api.StatusData
		if event.Decode(&data) == nil {
			s.Status = data.Message
		}

	case "source":
		var source api.Source
		if event.Decode(&source) == nil {
			s.Sources = append(s.Sources, source)
		}

	case "token":
		var data api.TokenData
		if event.Decode(&data) == nil {
			s.appendAnswer(data.Content)
		}

	case "grounding":
		var data api.GroundingData
		if event.Decode(&data) == nil {
			grounding := data
			s.Grounding = &grounding
		}

	case "suggestions":
		var data api.SuggestionsData
		if event.Decode(&data) == nil {
			s.Suggestions = data.Suggestions
		}

	case "done":
		var data api.SearchDoneData
		if event.Decode(&data) == nil {
			s.Done = true
			s.DoneData = data
			s.Status = ""
		}
		s.rememberTurn()
	}
}

// appendAnswer adds a streamed answer chunk.
func (s *searchState) appendAnswer(chunk string) {
	if s.answer == nil {
		s.answer = &strings.Builder{}
	}
	s.answer.WriteString(chunk)
}

// AnswerText returns the answer written so far.
func (s *searchState) AnswerText() string {
	if s.answer == nil {
		return ""
	}
	return s.answer.String()
}

// maxConversationTurns matches the Python TUI's cap of 10 messages
// (src/pythia/tui/screens/search.py:245) so the two front-ends send equivalent
// context to the server.
const maxConversationTurns = 10

// rememberTurn appends the exchange to the multi-turn context.
//
// It is called on "done" rather than on submit so an aborted or failed search
// does not leave a user turn with no assistant reply, which would make every
// later turn look like the assistant went silent.
func (s *searchState) rememberTurn() {
	answer := strings.TrimSpace(s.AnswerText())
	if answer == "" {
		return
	}
	// The user turn was already appended by the caller via beginTurn.
	if len(s.Conversation) == 0 || s.Conversation[len(s.Conversation)-1].Role != "user" {
		return
	}
	s.Conversation = append(s.Conversation, api.Message{Role: "assistant", Content: answer})
	if len(s.Conversation) > maxConversationTurns {
		s.Conversation = s.Conversation[len(s.Conversation)-maxConversationTurns:]
	}
}

// beginTurn records the user side of an exchange before streaming starts.
func (s *searchState) beginTurn(query string) {
	s.Conversation = append(s.Conversation, api.Message{Role: "user", Content: query})
}

// renderSearch draws the answer pane, grounding badge, sources and suggestions.
func (m Model) renderSearch(height int) string {
	state := m.search
	width := clampPositive(m.width - 2)
	lines := make([]string, 0, height)

	// --- status / badges -----------------------------------------------------
	badges := make([]string, 0, 3)
	switch {
	case state.Done && state.DoneData.CacheHit:
		badges = append(badges, badgeCacheStyle.Render(
			fmt.Sprintf("CACHE %.2f · %s", state.DoneData.Similarity,
				formatDuration(state.DoneData.ResponseTimeMS))))
	case state.Done:
		badges = append(badges, badgeWebStyle.Render(
			fmt.Sprintf("WEB %d sources · %s", state.DoneData.SourcesCount,
				formatDuration(state.DoneData.ResponseTimeMS))))
	case state.Status != "":
		badges = append(badges, mutedStyle.Render("◎ "+state.Status))
	}
	if state.Grounding != nil {
		g := state.Grounding
		style := successStyle
		if g.Score < 0.5 {
			style = warningStyle
		}
		badges = append(badges, style.Render(fmt.Sprintf("grounding %.0f%% · %d/%d claims",
			g.Score*100, g.GroundedClaims, g.TotalClaims)))
	}
	if len(badges) > 0 {
		lines = append(lines, truncate(strings.Join(badges, "  "), width))
	}

	// --- answer ---------------------------------------------------------------
	answer := state.AnswerText()
	if strings.TrimSpace(answer) == "" {
		if !state.Done && m.busy {
			lines = append(lines, "")
			lines = append(lines, dimStyle.Render("waiting for the first token…"))
		} else if !m.busy {
			lines = append(lines, "")
			lines = append(lines, dimStyle.Render("No answer yet. Type a question and press enter."))
			lines = append(lines, "")
			lines = append(lines, dimStyle.Render("Tip: ctrl+d enables deep scrape, tab switches to Research."))
		}
	} else {
		// The answer pane gets the room left after the badges, the sources
		// header and one suggestion line, floored so a short terminal still
		// shows text rather than an empty box.
		reserved := len(lines) + 4
		answerRows := clampPositive(height - reserved)
		wrapped := wrapHard(answer, width-2)
		for _, line := range windowLines(wrapped, answerRows, m.scrollTop) {
			lines = append(lines, answerStyle.Render(line))
		}
	}

	// --- sources --------------------------------------------------------------
	if len(state.Sources) > 0 {
		lines = append(lines, "")
		lines = append(lines, titleStyle.Render(truncate(
			fmt.Sprintf("Sources (%d)", len(state.Sources)), width)))
		for index, source := range state.Sources {
			if index >= 8 {
				lines = append(lines, dimStyle.Render(fmt.Sprintf("  … %d more", len(state.Sources)-index)))
				break
			}
			label := fmt.Sprintf("  %d. %s", source.Index, source.Title)
			lines = append(lines, textStyle.Render(truncate(label, width)))
			if source.URL != "" {
				lines = append(lines, dimStyle.Render(truncate("     "+source.URL, width)))
			}
		}
	}

	// --- follow-up suggestions ------------------------------------------------
	if len(state.Suggestions) > 0 {
		lines = append(lines, "")
		lines = append(lines, titleStyle.Render("Follow-ups"))
		for _, suggestion := range state.Suggestions {
			lines = append(lines, accentStyle.Render(truncate("  → "+suggestion, width)))
		}
	}

	return fitLines(lines, height)
}

// windowLines returns at most `rows` lines starting at offset `top`.
//
// It exists so the answer pane can scroll without a viewport model owning the
// content: the answer is generated here, and a viewport would need the same
// wrapping applied before being handed over anyway.
func windowLines(lines []string, rows, top int) []string {
	if rows <= 0 {
		return nil
	}
	if top < 0 {
		top = 0
	}
	if top >= len(lines) {
		// Clamp to the last page rather than returning nothing, so "end" shows
		// the tail of the answer instead of a blank pane.
		top = len(lines) - rows
		if top < 0 {
			top = 0
		}
	}
	end := top + rows
	if end > len(lines) {
		end = len(lines)
	}
	return lines[top:end]
}

// fitLines pads or trims a block to exactly rows lines.
//
// The body must occupy a fixed number of rows or the footer jumps around as the
// answer streams in, which makes the TUI unreadable.
func fitLines(lines []string, rows int) string {
	if rows <= 0 {
		return ""
	}
	if len(lines) > rows {
		lines = lines[:rows]
	}
	for len(lines) < rows {
		lines = append(lines, "")
	}
	return lipgloss.JoinVertical(lipgloss.Left, lines...)
}
