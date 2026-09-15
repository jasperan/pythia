// Package api is the HTTP client for the pythia FastAPI service.
//
// Every type here mirrors a real payload produced by src/pythia/server/app.py and
// src/pythia/server/search.py / research.py. Field names and SSE event names are
// copied from the server, not invented, so the Go front-end and the Python
// Textual TUI read exactly the same data.
package api

import "encoding/json"

// Health is the payload of GET /health (app.py:163).
type Health struct {
	Oracle    bool `json:"oracle"`
	Searxng   bool `json:"searxng"`
	LLM       bool `json:"llm"`
	CacheSize int  `json:"cache_size"`
}

// Healthy reports whether every backing service answered.
func (h Health) Healthy() bool { return h.Oracle && h.Searxng && h.LLM }

// Stats is the payload of GET /stats (oracle_cache.get_stats).
type Stats struct {
	TotalSearches int     `json:"total_searches"`
	CacheHits     int     `json:"cache_hits"`
	CacheHitRate  float64 `json:"cache_hit_rate"`
	AvgResponseMS int     `json:"avg_response_ms"`
	ActiveDays    int     `json:"active_days"`
}

// HistoryItem is one element of GET /history (oracle_cache.get_history).
type HistoryItem struct {
	Query          string `json:"query"`
	CacheHit       bool   `json:"cache_hit"`
	ResponseTimeMS int    `json:"response_time_ms"`
	ModelUsed      string `json:"model_used"`
	CreatedAt      string `json:"created_at"`
}

// Skill is one element of GET /skills (app.py:201).
type Skill struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Triggers    []string `json:"triggers"`
}

// Source is one search result, emitted as a "source" event (search.py:260).
type Source struct {
	Index   int    `json:"index"`
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

// SearchRequest is the body of POST /search. Only these keys are read by the
// server; deep and conversation_history are optional, matching search.py:182.
type SearchRequest struct {
	Query   string    `json:"query"`
	Model   string    `json:"model,omitempty"`
	Deep    bool      `json:"deep,omitempty"`
	History []Message `json:"conversation_history,omitempty"`
}

// Message is one turn of multi-turn context.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ResearchRequest is the body of POST /research.
type ResearchRequest struct {
	Query     string `json:"query"`
	Model     string `json:"model,omitempty"`
	MaxRounds int    `json:"max_rounds,omitempty"`
	Skill     string `json:"skill,omitempty"`
}

// Event is one decoded server-sent event.
type Event struct {
	Type string
	Data json.RawMessage
}

// Decode unmarshals the event payload into v.
func (e Event) Decode(v any) error { return json.Unmarshal(e.Data, v) }

// --- search event payloads (search.py EventType) ---

// StatusData is the payload of a "status" event.
type StatusData struct {
	Message string `json:"message"`
}

// TokenData is the payload of a "token" event.
type TokenData struct {
	Content string `json:"content"`
}

// GroundingData is the payload of a "grounding" event (search.py:267).
type GroundingData struct {
	Score          float64 `json:"score"`
	Label          string  `json:"label"`
	TotalClaims    int     `json:"total_claims"`
	GroundedClaims int     `json:"grounded_claims"`
}

// SuggestionsData is the payload of a "suggestions" event.
type SuggestionsData struct {
	Suggestions []string `json:"suggestions"`
}

// SearchDoneData is the payload of a search "done" event.
type SearchDoneData struct {
	CacheHit       bool    `json:"cache_hit"`
	Similarity     float64 `json:"similarity"`
	ResponseTimeMS int     `json:"response_time_ms"`
	SourcesCount   int     `json:"sources_count"`
}

// --- research event payloads (research.py ResearchEventType) ---

// RecallData is the payload of a "recall" event (research.py:369).
type RecallData struct {
	Count    int          `json:"count"`
	Findings []RecallItem `json:"findings"`
}

// RecallItem is one prior finding surfaced from the cache.
type RecallItem struct {
	SubQuery   string  `json:"sub_query"`
	Similarity float64 `json:"similarity"`
	FromQuery  string  `json:"from_query"`
}

// PlanData is the payload of a "plan" event (research.py:387).
type PlanData struct {
	SubQueries []string `json:"sub_queries"`
	Slug       string   `json:"slug"`
}

// RoundStartData is the payload of a "round_start" event (research.py:393).
type RoundStartData struct {
	Round     int `json:"round"`
	MaxRounds int `json:"max_rounds"`
	NumQuery  int `json:"num_queries"`
}

// FindingData is the payload of a "finding" event (research.py:411).
type FindingData struct {
	SubQuery       string `json:"sub_query"`
	SummaryPreview string `json:"summary_preview"`
	NumSources     int    `json:"num_sources"`
	Round          int    `json:"round"`
	SummaryFailed  bool   `json:"summary_failed"`
	Error          string `json:"error"`
}

// GapAnalysisData is the payload of a "gap_analysis" event (research.py:442).
type GapAnalysisData struct {
	Sufficient bool     `json:"sufficient"`
	Gaps       []string `json:"gaps"`
	Reasoning  string   `json:"reasoning"`
}

// EvolutionData is the payload of an "evolution" event (research.py:468).
type EvolutionData struct {
	Changes  []EvolutionChange `json:"changes"`
	Degraded bool              `json:"degraded"`
}

// EvolutionChange is one past-vs-new finding comparison.
type EvolutionChange struct {
	Type        string `json:"type"`
	PastFinding string `json:"past_finding"`
	NewFinding  string `json:"new_finding"`
	Explanation string `json:"explanation"`
}

// ResearchDoneData is the payload of a research "done" event (research.py:634).
type ResearchDoneData struct {
	RoundsUsed         int    `json:"rounds_used"`
	TotalFindings      int    `json:"total_findings"`
	TotalSources       int    `json:"total_sources"`
	RecalledFindings   int    `json:"recalled_findings"`
	ElapsedMS          int    `json:"elapsed_ms"`
	Slug               string `json:"slug"`
	VerificationStatus string `json:"verification_status"`
	FailedFindings     int    `json:"failed_findings"`
	EvolutionChanges   int    `json:"evolution_changes"`
}
