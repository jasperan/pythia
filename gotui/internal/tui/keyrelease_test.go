package tui

import (
	"testing"

	tea "charm.land/bubbletea/v2"
)

// These tests guard the COMPILER-INVISIBLE half of the v2 key migration.
//
// In v2, tea.KeyMsg is an interface -- "KeyMsg represents a key event. This can
// be either a key press or a key release event" -- while tea.KeyPressMsg and
// tea.KeyReleaseMsg are the concrete types. A handler written as
// `case tea.KeyMsg:` therefore matches BOTH and runs TWICE for every keystroke.
// Nothing fails to compile, which is why grep rather than the build has to find
// it, and no other test notices, because every other test only ever sends
// presses.
//
// The damage is silent and user-visible: a cursor advances two rows per press,
// and a toggle that flips a bool fires twice and cancels itself out, so the key
// looks dead. A key RELEASE must therefore have no effect at all.
func TestKeyReleaseDoesNotTypeIntoTheQuery(t *testing.T) {
	model := newTestModel(t)
	model = typeQuery(t, model, "ab")

	before := model.query.Value()
	next, _ := model.Update(tea.KeyReleaseMsg{Code: 'c'})
	model = next.(Model)

	if after := model.query.Value(); after != before {
		t.Errorf("a key RELEASE changed the query from %q to %q (the handler is matching "+
			"the KeyMsg interface, so it runs twice per keystroke)", before, after)
	}
}

// TestKeyReleaseDoesNotScroll covers a second double-fire site: scrolling.
func TestKeyReleaseDoesNotScroll(t *testing.T) {
	model := newTestModel(t)
	model.typing = false
	model.query.Blur()

	next, _ := model.Update(tea.KeyPressMsg{Code: 'j', Text: "j"})
	model = next.(Model)
	afterPress := model.scrollTop
	if afterPress != 1 {
		t.Fatalf("press j moved the scroll offset to %d, want 1", afterPress)
	}

	next, _ = model.Update(tea.KeyReleaseMsg{Code: 'j'})
	model = next.(Model)
	if model.scrollTop != afterPress {
		t.Errorf("a key RELEASE scrolled again: offset %d, want %d", model.scrollTop, afterPress)
	}
}

// TestKeyReleaseDoesNotSwitchScreens covers a third: the number-key tab jump,
// where a double fire would advance two tabs and look like a skipped screen.
func TestKeyReleaseDoesNotSwitchScreens(t *testing.T) {
	model := newTestModel(t)
	model.typing = false
	model.query.Blur()

	next, _ := model.Update(tea.KeyPressMsg{Code: '3', Text: "3"})
	model = next.(Model)
	if model.active != ScreenHistory {
		t.Fatalf("press 3 selected %v, want History", model.active)
	}

	next, _ = model.Update(tea.KeyReleaseMsg{Code: '3'})
	model = next.(Model)
	if model.active != ScreenHistory {
		t.Errorf("a key RELEASE switched screens to %v; the tab jump fired twice", model.active)
	}
}

// TestKeyReleaseDoesNotToggleDeepMode is the toggle case that cancels itself
// out: press+release would leave deep mode exactly as it started, so the key
// would look dead.
func TestKeyReleaseDoesNotToggleDeepMode(t *testing.T) {
	model := newTestModel(t)

	next, _ := model.Update(tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl})
	model = next.(Model)
	if !model.deep {
		t.Fatal("ctrl+d did not enable deep mode on press")
	}

	next, _ = model.Update(tea.KeyReleaseMsg{Code: 'd', Mod: tea.ModCtrl})
	model = next.(Model)
	if !model.deep {
		t.Error("a key RELEASE re-toggled deep mode; the toggle fired twice and cancelled out")
	}
}
