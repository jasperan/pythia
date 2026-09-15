package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// sseBody is captured-shaped output: what sse_starlette writes for
// _sse_wrap in app.py:119 -- an event: line, a data: line, then a blank line.
const sseBody = "event: status\n" +
	"data: {\"message\": \"Searching...\"}\n" +
	"\n" +
	"event: source\n" +
	"data: {\"index\": 1, \"title\": \"T\", \"url\": \"https://x\", \"snippet\": \"s\"}\n" +
	"\n" +
	"event: done\n" +
	"data: {\"cache_hit\": true, \"response_time_ms\": 42, \"sources_count\": 3}\n" +
	"\n"

func TestDecodeSSEReadsServerFraming(t *testing.T) {
	var got []Event
	err := DecodeSSE(strings.NewReader(sseBody), func(e Event) error {
		got = append(got, e)
		return nil
	})
	if err != nil {
		t.Fatalf("DecodeSSE returned %v", err)
	}

	want := []string{"status", "source", "done"}
	if len(got) != len(want) {
		t.Fatalf("decoded %d events, want %d: %+v", len(got), len(want), got)
	}
	for i, eventType := range want {
		if got[i].Type != eventType {
			t.Errorf("event %d type = %q, want %q", i, got[i].Type, eventType)
		}
	}

	var status StatusData
	if err := got[0].Decode(&status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if status.Message != "Searching..." {
		t.Errorf("status message = %q", status.Message)
	}

	var source Source
	if err := got[1].Decode(&source); err != nil {
		t.Fatalf("decode source: %v", err)
	}
	if source.Title != "T" || source.Index != 1 {
		t.Errorf("source = %+v", source)
	}

	var done SearchDoneData
	if err := got[2].Decode(&done); err != nil {
		t.Fatalf("decode done: %v", err)
	}
	// The done payload deliberately has no "similarity" key: a missing optional
	// field must decode as the zero value, not an error.
	if !done.CacheHit || done.ResponseTimeMS != 42 || done.SourcesCount != 3 {
		t.Errorf("done = %+v", done)
	}
}

// TestDecodeSSEHandlesKeepAliveAndMultiLine pins the two framings that a naive
// line-by-line reader gets wrong: comment heartbeats must be skipped, and a
// payload split over several data: lines must be joined with newlines rather
// than treated as separate events.
func TestDecodeSSEHandlesKeepAliveAndMultiLine(t *testing.T) {
	body := ": keep-alive\n" +
		"\n" +
		"event: plan\n" +
		"data: {\"sub_queries\": [\n" +
		"data: \"a\", \"b\"]}\n" +
		"\n"

	var got []Event
	if err := DecodeSSE(strings.NewReader(body), func(e Event) error {
		got = append(got, e)
		return nil
	}); err != nil {
		t.Fatalf("DecodeSSE returned %v", err)
	}

	if len(got) != 1 {
		t.Fatalf("decoded %d events, want 1 (the keep-alive must not be an event): %+v",
			len(got), got)
	}
	var plan PlanData
	if err := got[0].Decode(&plan); err != nil {
		t.Fatalf("multi-line data did not join into valid JSON: %v (raw %q)",
			err, string(got[0].Data))
	}
	if !reflect.DeepEqual(plan.SubQueries, []string{"a", "b"}) {
		t.Errorf("sub_queries = %v, want [a b]", plan.SubQueries)
	}
}

// TestDecodeSSEFlushesTrailingEventWithoutBlankLine covers a server that closes
// right after the final data: line.
func TestDecodeSSEFlushesTrailingEventWithoutBlankLine(t *testing.T) {
	body := "event: token\ndata: {\"content\": \"tail\"}"
	var got []Event
	if err := DecodeSSE(strings.NewReader(body), func(e Event) error {
		got = append(got, e)
		return nil
	}); err != nil {
		t.Fatalf("DecodeSSE returned %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("decoded %d events, want 1", len(got))
	}
	var token TokenData
	if err := got[0].Decode(&token); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if token.Content != "tail" {
		t.Errorf("content = %q", token.Content)
	}
}

// TestDecodeSSEPropagatesConsumerError guards the only error path that should
// stop a stream: the consumer itself failing.
func TestDecodeSSEPropagatesConsumerError(t *testing.T) {
	sentinel := errors.New("stop")
	err := DecodeSSE(strings.NewReader(sseBody), func(Event) error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Errorf("DecodeSSE returned %v, want %v", err, sentinel)
	}
}

// TestHistoryClampsLimit is not cosmetic: GET /history declares le=100, so
// passing 1000 (which the Python dashboard does at dashboard.py:95) is rejected
// with a 422 by the server.
func TestHistoryClampsLimit(t *testing.T) {
	var sawQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[]`))
	}))
	defer server.Close()

	client := NewClient(server.URL)
	if _, err := client.History(context.Background(), 1000); err != nil {
		t.Fatalf("History returned %v", err)
	}
	if sawQuery != "limit=100" {
		t.Errorf("query = %q, want limit=100 (the server's documented ceiling)", sawQuery)
	}

	if _, err := client.History(context.Background(), 0); err != nil {
		t.Fatalf("History(0) returned %v", err)
	}
	if sawQuery != "limit=20" {
		t.Errorf("query = %q, want the documented default limit=20", sawQuery)
	}
}

// TestStreamSearchPostsRealBody checks the request actually sent matches what
// the FastAPI route reads.
func TestStreamSearchPostsRealBody(t *testing.T) {
	var (
		sawPath string
		sawBody map[string]any
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&sawBody)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: token\ndata: {\"content\": \"hi\"}\n\n"))
	}))
	defer server.Close()

	var tokens []string
	err := NewClient(server.URL).StreamSearch(context.Background(),
		SearchRequest{Query: "what is rag", Model: "qwen3.5:9b", Deep: true},
		func(e Event) error {
			if e.Type == "token" {
				var token TokenData
				if err := e.Decode(&token); err != nil {
					return err
				}
				tokens = append(tokens, token.Content)
			}
			return nil
		})
	if err != nil {
		t.Fatalf("StreamSearch returned %v", err)
	}
	if sawPath != "/search" {
		t.Errorf("posted to %q, want /search", sawPath)
	}
	if sawBody["query"] != "what is rag" || sawBody["model"] != "qwen3.5:9b" || sawBody["deep"] != true {
		t.Errorf("body = %+v", sawBody)
	}
	if strings.Join(tokens, "") != "hi" {
		t.Errorf("tokens = %v", tokens)
	}
}

// TestStreamSearchSurfacesServerError covers an HTTP-level failure.
func TestStreamSearchSurfacesServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"detail":"boom"}`, http.StatusInternalServerError)
	}))
	defer server.Close()

	err := NewClient(server.URL).StreamSearch(context.Background(),
		SearchRequest{Query: "x"}, func(Event) error { return nil })
	if err == nil {
		t.Fatal("a 500 response produced no error")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %v, want it to name the status", err)
	}
}

// TestConnectionErrorNamesTheFix is the offline path: a user with no server
// must be told how to start one, not just shown "connection refused".
func TestConnectionErrorNamesTheFix(t *testing.T) {
	// Port 1 is reserved and cannot be listening.
	err := NewClient("http://127.0.0.1:1").StreamSearch(context.Background(),
		SearchRequest{Query: "x"}, func(Event) error { return nil })
	if err == nil {
		t.Fatal("expected a connection error")
	}
	if !strings.Contains(err.Error(), "--start-service") {
		t.Errorf("error = %v, want it to mention --start-service", err)
	}
}

func TestValidateBaseURL(t *testing.T) {
	for _, valid := range []string{"http://127.0.0.1:8900", "https://pythia.example"} {
		if err := ValidateBaseURL(valid); err != nil {
			t.Errorf("ValidateBaseURL(%q) = %v, want nil", valid, err)
		}
	}
	for _, invalid := range []string{"", "   ", "127.0.0.1:8900", "ftp://x", "http://"} {
		if err := ValidateBaseURL(invalid); err == nil {
			t.Errorf("ValidateBaseURL(%q) = nil, want an error", invalid)
		}
	}
}
