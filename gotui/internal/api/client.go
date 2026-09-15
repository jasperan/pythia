package api

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Client talks to a running pythia API server.
//
// It never reimplements search, research or caching: every value it returns came
// from the server, so a Go user and a Python user get identical results.
type Client struct {
	baseURL string
	http    *http.Client

	// streamClient has no overall timeout: a deep research run streams for
	// minutes and an overall deadline would cut it off mid-report. Cancellation
	// is the caller's context instead.
	streamClient *http.Client
}

// NewClient builds a client for a base URL such as http://127.0.0.1:8900.
func NewClient(baseURL string) *Client {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	return &Client{
		baseURL:      base,
		http:         &http.Client{Timeout: 15 * time.Second},
		streamClient: &http.Client{},
	}
}

// BaseURL returns the normalized server address.
func (c *Client) BaseURL() string { return c.baseURL }

// ValidateBaseURL rejects an address the client could not talk to.
func ValidateBaseURL(raw string) error {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return errors.New("address is empty")
	}
	parsed, err := url.Parse(trimmed)
	if err != nil {
		return fmt.Errorf("not a URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("scheme must be http or https, got %q", parsed.Scheme)
	}
	if parsed.Host == "" {
		return errors.New("missing host")
	}
	return nil
}

// Health calls GET /health.
func (c *Client) Health(ctx context.Context) (Health, error) {
	var out Health
	err := c.getJSON(ctx, "/health", &out)
	return out, err
}

// Stats calls GET /stats.
func (c *Client) Stats(ctx context.Context) (Stats, error) {
	var out Stats
	err := c.getJSON(ctx, "/stats", &out)
	return out, err
}

// MaxHistoryLimit is the server's own ceiling: GET /history declares
// Query(20, ge=1, le=100), so a larger limit is rejected with a 422.
const MaxHistoryLimit = 100

// History calls GET /history, clamping limit to the server's accepted range.
func (c *Client) History(ctx context.Context, limit int) ([]HistoryItem, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > MaxHistoryLimit {
		limit = MaxHistoryLimit
	}
	var out []HistoryItem
	err := c.getJSON(ctx, fmt.Sprintf("/history?limit=%d", limit), &out)
	return out, err
}

// ClearCache calls DELETE /cache and returns the number of deleted entries.
func (c *Client) ClearCache(ctx context.Context) (int, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodDelete, c.baseURL+"/cache", nil)
	if err != nil {
		return 0, err
	}
	response, err := c.http.Do(request)
	if err != nil {
		return 0, connectionError(c.baseURL, err)
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return 0, fmt.Errorf("DELETE /cache returned %s: %s", response.Status,
			strings.TrimSpace(string(body)))
	}
	var payload struct {
		Deleted int `json:"deleted"`
	}
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		return 0, fmt.Errorf("decode DELETE /cache: %w", err)
	}
	return payload.Deleted, nil
}

// Skills calls GET /skills.
func (c *Client) Skills(ctx context.Context) ([]Skill, error) {
	var out []Skill
	err := c.getJSON(ctx, "/skills", &out)
	return out, err
}

// StreamSearch posts to /search and calls yield for every event.
func (c *Client) StreamSearch(ctx context.Context, req SearchRequest, yield func(Event) error) error {
	return c.stream(ctx, "/search", req, yield)
}

// StreamResearch posts to /research and calls yield for every event.
func (c *Client) StreamResearch(ctx context.Context, req ResearchRequest, yield func(Event) error) error {
	return c.stream(ctx, "/research", req, yield)
}

// stream posts JSON and decodes the server-sent event response.
func (c *Client) stream(ctx context.Context, path string, body any, yield func(Event) error) error {
	encoded, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("encode request: %w", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path,
		strings.NewReader(string(encoded)))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	// sse-starlette streams; asking for the event stream explicitly keeps a
	// proxy from buffering the whole response into one blob.
	request.Header.Set("Accept", "text/event-stream")

	response, err := c.streamClient.Do(request)
	if err != nil {
		return connectionError(c.baseURL, err)
	}
	defer response.Body.Close()

	if response.StatusCode/100 != 2 {
		payload, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("%s returned %s: %s", path, response.Status,
			strings.TrimSpace(string(payload)))
	}
	return DecodeSSE(response.Body, yield)
}

// DecodeSSE parses a text/event-stream body.
//
// It is exported and reader-based so the framing can be tested against captured
// server output without a live HTTP server.
//
// sse-starlette writes an "event:" line, a "data:" line and a blank separator
// (app.py:119 _sse_wrap). A data payload is never assumed to fit on one line:
// every consecutive data: line is joined with a newline, which is what the SSE
// specification requires and what the Python client's aiter_lines loop does not
// have to think about.
func DecodeSSE(r io.Reader, yield func(Event) error) error {
	scanner := bufio.NewScanner(r)
	// A research report can exceed bufio's 64KiB default line limit; without
	// this a long token chunk would fail the whole stream.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	var (
		eventType string
		dataLines []string
	)

	flush := func() error {
		if eventType == "" && len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		event := Event{Type: eventType, Data: json.RawMessage(data)}
		eventType, dataLines = "", nil
		// A keep-alive comment or a payload we cannot use must not abort the
		// stream; only a real consumer error does.
		return yield(event)
	}

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		// SSE comment / keep-alive heartbeat.
		if strings.HasPrefix(line, ":") {
			continue
		}

		field, value, found := strings.Cut(line, ":")
		if !found {
			field, value = line, ""
		}
		value = strings.TrimPrefix(value, " ")

		switch field {
		case "event":
			eventType = value
		case "data":
			dataLines = append(dataLines, value)
		default:
			// id:, retry: and anything else are not needed here.
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read event stream: %w", err)
	}
	return flush()
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return err
	}
	response, err := c.http.Do(request)
	if err != nil {
		return connectionError(c.baseURL, err)
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("GET %s returned %s: %s", path, response.Status,
			strings.TrimSpace(string(body)))
	}
	if err := json.NewDecoder(response.Body).Decode(out); err != nil {
		return fmt.Errorf("decode GET %s: %w", path, err)
	}
	return nil
}

// connectionError turns a transport failure into something a user can act on,
// because "connection refused" alone does not say which service is missing.
func connectionError(baseURL string, err error) error {
	return fmt.Errorf("cannot reach the pythia API at %s (start it with `pythia serve` "+
		"or pass --start-service): %w", baseURL, err)
}
