package tui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/jasperan/pythia/gotui/internal/api"
)

// Action names for the scripted, non-interactive path.
const (
	ActionQuery      = "query"
	ActionResearch   = "research"
	ActionHistory    = "history"
	ActionStats      = "stats"
	ActionHealth     = "health"
	ActionSkills     = "skills"
	ActionClearCache = "clear-cache"
)

// ActionRequest is a scripted request.
//
// Every one of these maps onto an endpoint the Python CLI also exposes, so a
// pipeline gets the same numbers from either front-end. Destructive work
// (clear-cache) requires Confirmed, which only the --yes flag sets: a prompt is
// never the only route through it, and a pipe is never mistaken for consent.
type ActionRequest struct {
	Action    string
	Query     string
	Model     string
	Deep      bool
	MaxRounds int
	Limit     int
	Confirmed bool
	JSON      bool
}

// RunAction executes a scripted request and writes its result.
func RunAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	switch req.Action {
	case ActionQuery:
		return runQueryAction(ctx, client, req, out)
	case ActionResearch:
		return runResearchAction(ctx, client, req, out)
	case ActionHistory:
		return runHistoryAction(ctx, client, req, out)
	case ActionStats:
		return runStatsAction(ctx, client, req, out)
	case ActionHealth:
		return runHealthAction(ctx, client, req, out)
	case ActionSkills:
		return runSkillsAction(ctx, client, req, out)
	case ActionClearCache:
		return runClearCacheAction(ctx, client, req, out)
	}
	return fmt.Errorf("unknown action %q", req.Action)
}

// runQueryAction streams a search and prints the answer, the sources and the
// grounding score. It is the headless twin of the Search tab.
func runQueryAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	var (
		answer    strings.Builder
		sources   []api.Source
		done      api.SearchDoneData
		grounding *api.GroundingData
	)
	err := client.StreamSearch(ctx, api.SearchRequest{
		Query: req.Query, Model: req.Model, Deep: req.Deep,
	}, func(event api.Event) error {
		switch event.Type {
		case "source":
			var source api.Source
			if err := event.Decode(&source); err != nil {
				return err
			}
			sources = append(sources, source)
		case "token":
			var token api.TokenData
			if err := event.Decode(&token); err != nil {
				return err
			}
			answer.WriteString(token.Content)
		case "grounding":
			var data api.GroundingData
			if err := event.Decode(&data); err != nil {
				return err
			}
			grounding = &data
		case "done":
			return event.Decode(&done)
		}
		return nil
	})
	if err != nil {
		return err
	}

	if req.JSON {
		return writeJSON(out, map[string]any{
			"query":         req.Query,
			"answer":        answer.String(),
			"sources":       sources,
			"cache_hit":     done.CacheHit,
			"similarity":    done.Similarity,
			"response_ms":   done.ResponseTimeMS,
			"sources_count": done.SourcesCount,
			"grounding":     grounding,
		})
	}

	fmt.Fprintln(out, answer.String())
	fmt.Fprintln(out)
	origin := "web"
	if done.CacheHit {
		origin = fmt.Sprintf("cache (%.2f similarity)", done.Similarity)
	}
	fmt.Fprintf(out, "source: %s · %s\n", origin, formatDuration(done.ResponseTimeMS))
	if grounding != nil {
		fmt.Fprintf(out, "grounding: %.0f%% (%d/%d claims) · %s\n",
			grounding.Score*100, grounding.GroundedClaims, grounding.TotalClaims, grounding.Label)
	}
	for _, source := range sources {
		fmt.Fprintf(out, "  %d. %s\n     %s\n", source.Index, source.Title, source.URL)
	}
	return nil
}

// runResearchAction streams a deep-research run and prints the report.
//
// It prints the sub-query plan as it arrives so a headless caller can see the
// same progress the tree view shows.
func runResearchAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	var state ResearchState
	state.Reset(req.Query)

	err := client.StreamResearch(ctx, api.ResearchRequest{
		Query: req.Query, Model: req.Model, MaxRounds: req.MaxRounds,
	}, func(event api.Event) error {
		if err := state.ApplyEvent(event); err != nil {
			return err
		}
		if !req.JSON {
			switch event.Type {
			case "plan":
				var data api.PlanData
				_ = event.Decode(&data)
				fmt.Fprintf(out, "plan (%d sub-queries):\n", len(data.SubQueries))
				for _, query := range data.SubQueries {
					fmt.Fprintf(out, "  - %s\n", query)
				}
			case "finding":
				var data api.FindingData
				_ = event.Decode(&data)
				fmt.Fprintf(out, "  ✓ %s (%d sources)\n", data.SubQuery, data.NumSources)
			case "round_start":
				var data api.RoundStartData
				_ = event.Decode(&data)
				fmt.Fprintf(out, "round %d/%d\n", data.Round, data.MaxRounds)
			}
		}
		return nil
	})
	if err != nil {
		return err
	}

	if req.JSON {
		return writeJSON(out, map[string]any{
			"query":      req.Query,
			"slug":       state.Slug,
			"report":     state.AnswerText(),
			"findings":   state.Findings,
			"sources":    state.Sources,
			"failed":     state.Failed,
			"elapsed_ms": state.ElapsedMS,
			"rounds":     len(state.Rounds),
			"verified":   state.VerifyNote,
		})
	}

	fmt.Fprintln(out)
	fmt.Fprintln(out, state.AnswerText())
	fmt.Fprintln(out)
	fmt.Fprintf(out, "%d findings · %d sources · %d rounds · %s\n",
		state.Findings, state.Sources, len(state.Rounds), formatDuration(state.ElapsedMS))
	if state.Slug != "" {
		fmt.Fprintf(out, "slug: %s\n", state.Slug)
	}
	return nil
}

func runHistoryAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	items, err := client.History(ctx, req.Limit)
	if err != nil {
		return err
	}
	if req.JSON {
		return writeJSON(out, items)
	}
	if len(items) == 0 {
		fmt.Fprintln(out, "No searches recorded yet.")
		return nil
	}
	fmt.Fprintf(out, "%d search(es)\n", len(items))
	for _, item := range items {
		origin := "web"
		if item.CacheHit {
			origin = "cache"
		}
		fmt.Fprintf(out, "  %-5s %6s  %-14s %s\n", origin,
			formatDuration(item.ResponseTimeMS), item.ModelUsed, item.Query)
	}
	return nil
}

func runStatsAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	stats, err := client.Stats(ctx)
	if err != nil {
		return err
	}
	if req.JSON {
		return writeJSON(out, stats)
	}
	fmt.Fprintf(out, "searches      %d\n", stats.TotalSearches)
	fmt.Fprintf(out, "cache hits    %d\n", stats.CacheHits)
	fmt.Fprintf(out, "hit rate      %.1f%%\n", stats.CacheHitRate*100)
	fmt.Fprintf(out, "avg response  %s\n", formatDuration(stats.AvgResponseMS))
	fmt.Fprintf(out, "active days   %d\n", stats.ActiveDays)
	return nil
}

func runHealthAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	health, err := client.Health(ctx)
	if err != nil {
		return err
	}
	if req.JSON {
		return writeJSON(out, health)
	}
	fmt.Fprintf(out, "oracle    %s\n", upDown(health.Oracle))
	fmt.Fprintf(out, "searxng   %s\n", upDown(health.Searxng))
	fmt.Fprintf(out, "ollama    %s\n", upDown(health.LLM))
	fmt.Fprintf(out, "cache     %d entries\n", health.CacheSize)
	return nil
}

func runSkillsAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	skills, err := client.Skills(ctx)
	if err != nil {
		return err
	}
	if req.JSON {
		return writeJSON(out, skills)
	}
	if len(skills) == 0 {
		fmt.Fprintln(out, "No research skills registered.")
		return nil
	}
	for _, skill := range skills {
		fmt.Fprintf(out, "  %-24s %s\n", skill.Name, skill.Description)
		if len(skill.Triggers) > 0 {
			fmt.Fprintf(out, "  %-24s triggers: %s\n", "", strings.Join(skill.Triggers, ", "))
		}
	}
	return nil
}

// runClearCacheAction refuses to run without --yes.
//
// Clearing the semantic cache is irreversible: every stored answer and its
// embedding is deleted, along with the history that makes hit-rate statistics
// meaningful.
func runClearCacheAction(ctx context.Context, client *api.Client, req ActionRequest, out io.Writer) error {
	if !req.Confirmed {
		return errors.New("refusing to clear the cache without --yes: it deletes every cached " +
			"answer, its embedding and the recorded history")
	}
	deleted, err := client.ClearCache(ctx)
	if err != nil {
		return err
	}
	if req.JSON {
		return writeJSON(out, map[string]any{"deleted": deleted})
	}
	fmt.Fprintf(out, "cache cleared: %d entries\n", deleted)
	return nil
}

func upDown(ok bool) string {
	if ok {
		return "up"
	}
	return "down"
}

// writeJSON emits an indented JSON document for scripting.
func writeJSON(out io.Writer, value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode json: %w", err)
	}
	_, err = out.Write(append(encoded, '\n'))
	return err
}

// ParseActionFlags decides which scripted action a flag set requests.
//
// Order matters: the most specific flag wins, so "--stats --clear-cache" cannot
// silently degrade into a read.
func ParseActionFlags(query, research string, history, stats, health, skills, clearCache bool) (string, bool) {
	switch {
	case clearCache:
		return ActionClearCache, true
	case strings.TrimSpace(research) != "":
		return ActionResearch, true
	case strings.TrimSpace(query) != "":
		return ActionQuery, true
	case history:
		return ActionHistory, true
	case stats:
		return ActionStats, true
	case health:
		return ActionHealth, true
	case skills:
		return ActionSkills, true
	}
	return "", false
}

// AccessibleNotice is printed before the plain prompt path so a screen-reader
// user knows why the full-screen UI did not open.
const AccessibleNotice = "ACCESSIBLE is set: using plain prompts instead of the full-screen TUI."
