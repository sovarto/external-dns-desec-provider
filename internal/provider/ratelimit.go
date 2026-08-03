package provider

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
)

// deSEC applies several throttle scopes; the one that bites the read path is
// the account-wide `user` scope of 2000 requests/day, shared across ALL
// domains and both read and write. When it is exhausted deSEC answers 429 with
// a Retry-After that can be ~12.5h (the remaining seconds of the daily bucket).
// The per-request read scope `dns_api_cheap` (10/s 50/min) is comparatively
// generous and is not the limit we hit in practice.
// https://desec.readthedocs.io/en/latest/rate-limits.html
const userDailyRequestLimit = 2000

// retryAfterCap bounds how long a single retryablehttp attempt may sleep on a
// 429. The deSEC library builds its retry client internally and exposes no
// Backoff hook, so we clamp the Retry-After header the transport hands upward:
// DefaultBackoff sleeps the header value verbatim (not clamped to
// RetryWaitMax), which is how a 429 turned into a silent ~12.5h sleep. The
// unclamped duration is still recorded in the tracker so the next call
// short-circuits for the real window.
const retryAfterCap = 30 * time.Second

// RateLimitError is returned when a deSEC API call is short-circuited because
// the client is still inside a throttle window observed from a prior 429
// response. Returning this instead of hitting the API prevents the daily quota
// from being burned while external-dns keeps retrying.
type RateLimitError struct {
	RetryAfter time.Duration
}

func (e *RateLimitError) Error() string {
	return fmt.Sprintf("deSEC rate limit active, retry after %s", e.RetryAfter)
}

// rateLimitTracker remembers the next time the client should attempt to call
// the deSEC API, derived from Retry-After headers seen on 429 responses. It
// also counts requests within the current UTC day to warn before the
// account-wide `user` daily quota is exhausted.
type rateLimitTracker struct {
	mu            sync.Mutex
	nextAllowedAt time.Time
	now           func() time.Time

	dayStart  time.Time
	dayCount  int
	warnedDay bool
}

func newRateLimitTracker() *rateLimitTracker {
	return &rateLimitTracker{now: time.Now}
}

// observeRequest counts every request against the daily `user` budget and warns
// once per day when the remaining budget runs low, so the eventual 429 is not
// the first sign that the account is out of quota. Records roll over on the UTC
// day boundary deSEC uses to reset the bucket.
func (t *rateLimitTracker) observeRequest() {
	t.mu.Lock()
	defer t.mu.Unlock()

	day := t.now().UTC().Truncate(24 * time.Hour)
	if day.After(t.dayStart) {
		t.dayStart = day
		t.dayCount = 0
		t.warnedDay = false
	}
	t.dayCount++

	const warnThreshold = userDailyRequestLimit * 9 / 10
	if t.dayCount >= warnThreshold && !t.warnedDay {
		t.warnedDay = true
		log.Warnf("approaching deSEC daily request limit: %d/%d requests today (account-wide, all domains)",
			t.dayCount, userDailyRequestLimit)
	}
}

// record extends the throttle window. A shorter delay than the current
// deadline is ignored so a later 429 with a smaller Retry-After can't shorten
// a longer window already in effect.
func (t *rateLimitTracker) record(d time.Duration) {
	if d <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	deadline := t.now().Add(d)
	if deadline.After(t.nextAllowedAt) {
		t.nextAllowedAt = deadline
	}
}

// wait returns the remaining throttle duration; 0 means the client may proceed.
func (t *rateLimitTracker) wait() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	remaining := t.nextAllowedAt.Sub(t.now())
	if remaining < 0 {
		return 0
	}
	return remaining
}

// rateLimitTransport is an http.RoundTripper that records Retry-After from any
// 429 Too Many Requests responses observed on the wire, including ones
// retryablehttp will subsequently retry.
type rateLimitTransport struct {
	inner   http.RoundTripper
	tracker *rateLimitTracker
}

func (rt *rateLimitTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.tracker.observeRequest()
	resp, err := rt.inner.RoundTrip(req)
	if err != nil || resp == nil {
		return resp, err
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		retryAfter := resp.Header.Get("Retry-After")
		log.Warnf("deSEC throttled %s %s: HTTP 429, Retry-After=%q, detail=%q",
			req.Method, req.URL.Path, retryAfter, peekThrottleDetail(resp))
		if d, ok := parseRetryAfter(retryAfter, rt.tracker.now()); ok {
			// Record the real window so the next call short-circuits, but clamp
			// the header retryablehttp sees so a single attempt can never sleep
			// for hours if the request context is not cancelled in time.
			rt.tracker.record(d)
			if d > retryAfterCap {
				resp.Header.Set("Retry-After", strconv.Itoa(int(retryAfterCap/time.Second)))
			}
		}
	}
	return resp, nil
}

// peekThrottleDetail reads deSEC's JSON throttle body ("Request was throttled.
// Expected available in N seconds.") for logging, then rewinds it so the caller
// can still consume the response.
func peekThrottleDetail(resp *http.Response) string {
	if resp.Body == nil {
		return ""
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 512))
	if err != nil {
		return ""
	}
	rest, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	resp.Body = io.NopCloser(io.MultiReader(bytes.NewReader(body), bytes.NewReader(rest)))
	return string(body)
}

// parseRetryAfter parses an HTTP Retry-After header value, which RFC 7231
// defines as either delta-seconds or an HTTP-date, relative to `now`.
func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	if value == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(value); err == nil {
		return time.Duration(secs) * time.Second, true
	}
	if t, err := http.ParseTime(value); err == nil {
		return t.Sub(now), true
	}
	return 0, false
}
