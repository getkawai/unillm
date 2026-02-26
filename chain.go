package fantasy

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

// ChainLanguageModel wraps multiple LanguageModels and tries them in order.
// If the primary model fails, it automatically falls back to the next model in the chain.
// This enables resilient LLM calls with graceful degradation.
type ChainLanguageModel struct {
	models []LanguageModel
	name   string

	// Circuit breaker state per model
	mu               sync.RWMutex
	failures         map[int]int       // failure count per model index
	lastFailure      map[int]time.Time // last failure time per model index
	circuitOpen      map[int]bool      // whether circuit is open (skip model)
	failureThreshold int               // failures before opening circuit
	resetTimeout     time.Duration     // time before trying again
}

// ChainOption configures a ChainLanguageModel.
type ChainOption func(*ChainLanguageModel)

// WithCircuitBreaker enables circuit breaker pattern.
// After failureThreshold consecutive failures, the model will be skipped for resetTimeout duration.
func WithCircuitBreaker(failureThreshold int, resetTimeout time.Duration) ChainOption {
	return func(c *ChainLanguageModel) {
		c.failureThreshold = failureThreshold
		c.resetTimeout = resetTimeout
	}
}

// WithChainName sets a descriptive name for the chain (for logging).
func WithChainName(name string) ChainOption {
	return func(c *ChainLanguageModel) {
		c.name = name
	}
}

// NewChain creates a new ChainLanguageModel that tries models in order until one succeeds.
// At least one model must be provided.
func NewChain(models []LanguageModel, opts ...ChainOption) (*ChainLanguageModel, error) {
	if len(models) == 0 {
		return nil, fmt.Errorf("at least one model is required for chain")
	}

	// Filter out nil models
	validModels := make([]LanguageModel, 0, len(models))
	for _, m := range models {
		if m != nil {
			validModels = append(validModels, m)
		}
	}

	if len(validModels) == 0 {
		return nil, fmt.Errorf("at least one non-nil model is required for chain")
	}

	// Build name from model names
	names := make([]string, len(validModels))
	for i, m := range validModels {
		names[i] = fmt.Sprintf("%s/%s", m.Provider(), m.Model())
	}

	chain := &ChainLanguageModel{
		models:           validModels,
		name:             strings.Join(names, " -> "),
		failures:         make(map[int]int),
		lastFailure:      make(map[int]time.Time),
		circuitOpen:      make(map[int]bool),
		failureThreshold: 0, // disabled by default
		resetTimeout:     5 * time.Minute,
	}

	for _, opt := range opts {
		opt(chain)
	}

	return chain, nil
}

// isCircuitOpen checks if a model's circuit breaker is open (should skip).
// If the reset timeout has passed, it also resets the circuit state.
func (c *ChainLanguageModel) isCircuitOpen(idx int) bool {
	if c.failureThreshold == 0 {
		return false // circuit breaker disabled
	}

	c.mu.RLock()
	isOpen := c.circuitOpen[idx]
	lastFailTime := c.lastFailure[idx]
	c.mu.RUnlock()

	if !isOpen {
		return false
	}

	// If resetTimeout is 0, circuit stays open until app restart (for daily rate limits)
	if c.resetTimeout == 0 {
		return true
	}

	// Check if reset timeout has passed
	if time.Since(lastFailTime) > c.resetTimeout {
		// Reset the circuit state (half-open -> allow one retry)
		c.mu.Lock()
		// Double-check in case another goroutine already reset it
		if c.circuitOpen[idx] && time.Since(c.lastFailure[idx]) > c.resetTimeout {
			c.circuitOpen[idx] = false
			c.failures[idx] = 0
			log.Printf("⚡ Chain[%s]: Circuit half-open for model %d, allowing retry",
				c.name, idx)
		}
		c.mu.Unlock()
		return false
	}

	return true
}

// recordFailure records a failure for a model.
func (c *ChainLanguageModel) recordFailure(idx int) {
	if c.failureThreshold == 0 {
		return // circuit breaker disabled
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.failures[idx]++
	c.lastFailure[idx] = time.Now()

	if c.failures[idx] >= c.failureThreshold {
		c.circuitOpen[idx] = true
		log.Printf("⚡ Chain[%s]: Circuit opened for model %d (%s/%s) after %d failures",
			c.name, idx, c.models[idx].Provider(), c.models[idx].Model(), c.failures[idx])
	}
}

// recordSuccess records a success for a model (resets circuit breaker).
func (c *ChainLanguageModel) recordSuccess(idx int) {
	if c.failureThreshold == 0 {
		return // circuit breaker disabled
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.circuitOpen[idx] {
		log.Printf("⚡ Chain[%s]: Circuit closed for model %d (%s/%s)",
			c.name, idx, c.models[idx].Provider(), c.models[idx].Model())
	}

	c.failures[idx] = 0
	c.circuitOpen[idx] = false
}

// Generate tries each model in order until one succeeds.
// Timeout errors trigger fallback to next model with fresh context.
func (c *ChainLanguageModel) Generate(ctx context.Context, call Call) (*Response, error) {
	var lastErr error
	var attemptedModels []string

	for i, model := range c.models {
		// Check circuit breaker
		if c.isCircuitOpen(i) {
			log.Printf("🔀 Chain[%s]: Skipping model %d (%s/%s) - circuit open",
				c.name, i, model.Provider(), model.Model())
			continue
		}

		modelName := fmt.Sprintf("%s/%s", model.Provider(), model.Model())
		attemptedModels = append(attemptedModels, modelName)

		log.Printf("🔀 Chain[%s]: Trying model %d/%d (%s)",
			c.name, i+1, len(c.models), modelName)

		// Create fresh context for each model
		modelCtx := ctx

		resp, err := model.Generate(modelCtx, call)
		if err == nil {
			c.recordSuccess(i)
			if i > 0 {
				log.Printf("✅ Chain[%s]: Fallback succeeded with model %s", c.name, modelName)
			}
			return resp, nil
		}

		c.recordFailure(i)
		lastErr = err
		log.Printf("⚠️  Chain[%s]: Model %s failed: %v", c.name, modelName, err)

		// Only stop if user explicitly cancelled (not timeout)
		// Timeout should trigger fallback to local model
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		}
		// For DeadlineExceeded (timeout), continue to next model
	}

	return nil, fmt.Errorf("all models failed (tried: %s): %w",
		strings.Join(attemptedModels, ", "), lastErr)
}

// Stream tries each model in order.
// If a model fails during setup or *during* streaming, it seamlessly falls back to the next model.
func (c *ChainLanguageModel) Stream(ctx context.Context, call Call) (StreamResponse, error) {
	// We return a single stream iterator that internally manages switching between models
	return func(yield func(StreamPart) bool) {
		var lastErr error
		var attemptedModels []string

		// Setup context for the chain - we might need to recreate context if one model fails/timeouts
		// to give the next model a fresh start, but usually we respect the parent context.
		// If dealing with timeouts per model, complex context handling is needed.
		// For now, we assume the parent context is sufficient or the models handle their own timeouts.

		for i, model := range c.models {
			// Wrapped in anonymous function to ensure defer executes at the end of each iteration (no resource leaks)
			shouldReturn, finished := func() (bool, bool) {
				// Check circuit breaker
				if c.isCircuitOpen(i) {
					log.Printf("🔀 Chain[%s]: Skipping stream model %d (%s/%s) - circuit open",
						c.name, i, model.Provider(), model.Model())
					return false, false
				}

				modelName := fmt.Sprintf("%s/%s", model.Provider(), model.Model())
				attemptedModels = append(attemptedModels, modelName)

				log.Printf("🔀 Chain[%s]: Trying stream model %d/%d (%s)", c.name, i+1, len(c.models), modelName)
				startTime := time.Now()

				// Create a controllable context for this specific model attempt
				modelCtx, modelCancel := context.WithCancel(ctx)
				defer modelCancel()

				// Start the stream with the sub-context
				log.Printf("🔍 Chain[%s]: Calling model.Stream() for %s...", c.name, modelName)
				stream, err := model.Stream(modelCtx, call)
				if err != nil {
					c.recordFailure(i)
					lastErr = err
					log.Printf("⚠️  Chain[%s]: Setup failed for %s: %v (falling back)", c.name, modelName, err)
					return false, false
				}
				log.Printf("✅ Chain[%s]: Stream setup successful for %s", c.name, modelName)

				// Consumption with Watchdog
				// Buffered channel to prevent goroutine leak if main loop exits
				partChan := make(chan StreamPart, 100)

				// Start consumption in background
				go func() {
					defer close(partChan)
					for part := range stream {
						select {
						case partChan <- part:
						case <-modelCtx.Done():
							return
						}
					}
				}()

				streamFailed := false
				partCount := 0
				firstTokenReceived := false

				// Watchdog: Max 5s to receive the first token
				watchdog := time.NewTimer(5 * time.Second)
				defer watchdog.Stop()

			consumeLoop:
				for {
					select {
					case <-ctx.Done():
						modelCancel() // Kill current model stream immediately
						return true, false
					case <-watchdog.C:
						// HANG DETECTED!
						if !firstTokenReceived {
							c.recordFailure(i)
							lastErr = fmt.Errorf("first token timeout (hang) after 5s")
							log.Printf("⚠️  Chain[%s]: Model %s hung for 5s. Forcing fallback...", c.name, modelName)
							streamFailed = true
							modelCancel() // Terminate this model connection
							break consumeLoop
						}
					case part, ok := <-partChan:
						if !ok {
							break consumeLoop // Stream completed naturally
						}

						if !firstTokenReceived {
							firstTokenReceived = true
							watchdog.Stop() // Token received, stop hang watchdog
							log.Printf("⚡ Chain[%s]: First token from %s after %v", c.name, modelName, time.Since(startTime))
						}

						partCount++

						// Check for mid-stream error parts
						if part.Type == StreamPartTypeError || part.Error != nil {
							c.recordFailure(i)
							lastErr = part.Error
							if lastErr == nil {
								lastErr = fmt.Errorf("unknown stream error")
							}
							log.Printf("⚠️  Chain[%s]: Mid-stream error from %s after %d parts (%v elapsed): %v. Switching...",
								c.name, modelName, partCount, time.Since(startTime), lastErr)
							streamFailed = true
							modelCancel() // Kill connection
							break consumeLoop
						}

						// Yield successful part
						if !yield(part) {
							log.Printf("🛑 Chain[%s]: Consumer stopped yielding after %d parts from %s", c.name, partCount, modelName)
							modelCancel() // Kill connection
							return true, false
						}
					}
				}

				if !streamFailed {
					c.recordSuccess(i)
					log.Printf("✅ Chain[%s]: Stream completed successfully from %s - %d parts in %v",
						c.name, modelName, partCount, time.Since(startTime))
					return false, true // Success!
				}

				log.Printf("🔄 Chain[%s]: Model %s failed/hung after %d parts in %v, trying next model...",
					c.name, modelName, partCount, time.Since(startTime))

				return false, false // Move to next model
			}()

			if shouldReturn {
				return
			}
			if finished {
				return
			}
		}

		// If we exhausted all models, yield the final error
		finalErr := fmt.Errorf("all chain models failed (tried: %s): %w", strings.Join(attemptedModels, ", "), lastErr)
		yield(StreamPart{
			Type:  StreamPartTypeError,
			Error: finalErr,
		})
	}, nil
}

// GenerateObject tries each model in order until one succeeds.
// Timeout errors trigger fallback to next model with fresh context.
func (c *ChainLanguageModel) GenerateObject(ctx context.Context, call ObjectCall) (*ObjectResponse, error) {
	var lastErr error
	var attemptedModels []string

	for i, model := range c.models {
		if c.isCircuitOpen(i) {
			log.Printf("🔀 Chain[%s]: Skipping object model %d (%s/%s) - circuit open",
				c.name, i, model.Provider(), model.Model())
			continue
		}

		modelName := fmt.Sprintf("%s/%s", model.Provider(), model.Model())
		attemptedModels = append(attemptedModels, modelName)

		// Create fresh context for each model
		modelCtx := ctx

		resp, err := model.GenerateObject(modelCtx, call)
		if err == nil {
			c.recordSuccess(i)
			return resp, nil
		}

		c.recordFailure(i)
		lastErr = err
		log.Printf("⚠️  Chain[%s]: GenerateObject %s failed: %v", c.name, modelName, err)

		// Only stop if user explicitly cancelled (not timeout)
		// Timeout should trigger fallback to next model
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		}
		// For DeadlineExceeded (timeout), continue to next model
	}

	return nil, fmt.Errorf("all models failed (tried: %s): %w",
		strings.Join(attemptedModels, ", "), lastErr)
}

// StreamObject tries each model in order until one succeeds.
// If a model fails during setup or *during* streaming, it seamlessly falls back to the next model.
func (c *ChainLanguageModel) StreamObject(ctx context.Context, call ObjectCall) (ObjectStreamResponse, error) {
	// We return a single stream iterator that internally manages switching between models
	return func(yield func(ObjectStreamPart) bool) {
		var lastErr error
		var attemptedModels []string

		for i, model := range c.models {
			// Check circuit breaker
			if c.isCircuitOpen(i) {
				log.Printf("🔀 Chain[%s]: Skipping StreamObject model %d (%s/%s) - circuit open",
					c.name, i, model.Provider(), model.Model())
				continue
			}

			modelName := fmt.Sprintf("%s/%s", model.Provider(), model.Model())
			attemptedModels = append(attemptedModels, modelName)

			log.Printf("🔀 Chain[%s]: Trying StreamObject with model %d/%d (%s)", c.name, i+1, len(c.models), modelName)

			// Attempt to start streaming
			stream, err := model.StreamObject(ctx, call)
			if err != nil {
				c.recordFailure(i)
				lastErr = err
				log.Printf("⚠️  Chain[%s]: StreamObject setup failed for %s: %v", c.name, modelName, err)

				// Only stop if user explicitly cancelled (not timeout)
				if ctx.Err() == context.Canceled {
					yield(ObjectStreamPart{Error: fmt.Errorf("context cancelled: %w", ctx.Err())})
					return
				}
				continue // Try next model
			}

			// Stream started successfully. Now we consume it.
			streamFailed := false
			for part := range stream {
				// Check for errors in the stream
				if part.Error != nil {
					// Mid-stream failure detected!
					c.recordFailure(i)
					lastErr = part.Error
					log.Printf("⚠️  Chain[%s]: Mid-stream error from StreamObject %s: %v. Switching to next model...", c.name, modelName, lastErr)

					streamFailed = true
					break // Stop consuming this stream, move to next model
				}

				// Yield successful part
				if !yield(part) {
					return // Consumer stopped
				}
			}

			if !streamFailed {
				// Model finished successfully
				c.recordSuccess(i)
				return // We are done
			}

			// If we are here, streamFailed was true.
			// We loop back to try the next model.
		}

		// If we exhausted all models, yield the final error
		finalErr := fmt.Errorf("all chain models failed StreamObject (tried: %s): %w", strings.Join(attemptedModels, ", "), lastErr)
		yield(ObjectStreamPart{Error: finalErr})
	}, nil
}

// Provider returns "chain" as the provider name.
func (c *ChainLanguageModel) Provider() string {
	return "chain"
}

// Model returns a descriptive name of the chain.
func (c *ChainLanguageModel) Model() string {
	return c.name
}

// PrimaryModel returns the first (primary) model in the chain.
func (c *ChainLanguageModel) PrimaryModel() LanguageModel {
	if len(c.models) > 0 {
		return c.models[0]
	}
	return nil
}

// Models returns all models in the chain.
func (c *ChainLanguageModel) Models() []LanguageModel {
	return c.models
}
