// Package pooled provides a LanguageModel implementation that uses CLIProxyAPI's
// fallback mechanism for robust API key rotation and error handling.
package pooled

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/kawai-network/veridium/pkg/cliproxy/sdk/cliproxy/auth"
	"github.com/kawai-network/veridium/pkg/cliproxy/sdk/cliproxy/executor"
	"github.com/getkawai/unillm"
)

// PooledProvider wraps a fantasy provider with CLIProxyAPI's fallback mechanism.
type PooledProvider struct {
	providerName      string
	baseURL           string
	modelName         string
	manager           *auth.Manager
	executor          *PooledExecutor
	metrics           *PoolMetrics           // NEW: Metrics tracking
	rotationManager   *AutoRotationManager   // NEW: Auto rotation
	currentKeyIndex   int                    // NEW: Current active key
	indexMu           sync.RWMutex           // FIXED: Protect currentKeyIndex from race conditions
	apiKeys           []string               // NEW: Store keys for rotation
	cancelFunc        context.CancelFunc     // FIXED: For proper goroutine cleanup
}

// PooledExecutor implements auth.ProviderExecutor for fantasy providers.
type PooledExecutor struct {
	providerName string
	baseURL      string
	modelName    string
	createClient func(apiKey string) (fantasy.LanguageModel, error)
}

// Config holds configuration for creating a pooled provider.
type Config struct {
	ProviderName     string
	BaseURL          string
	ModelName        string
	APIKeys          []string
	CreateClient     func(apiKey string) (fantasy.LanguageModel, error)
	EnableMetrics    bool              // NEW: Enable metrics tracking (default: true)
	EnableRotation   bool              // NEW: Enable auto rotation (default: true)
	RotationStrategy RotationStrategy  // NEW: Custom rotation strategy (optional)
}

// New creates a new PooledProvider with multiple API keys.
func New(cfg Config) (*PooledProvider, error) {
	if len(cfg.APIKeys) == 0 {
		return nil, fmt.Errorf("at least one API key is required")
	}
	if cfg.CreateClient == nil {
		return nil, fmt.Errorf("CreateClient function is required")
	}

	// Create memory store for auth state
	store := auth.NewMemoryStore()

	// Use RoundRobinSelector for load balancing
	selector := &auth.RoundRobinSelector{}

	// Create manager
	manager := auth.NewManager(store, selector, nil)

	// Configure retry settings
	manager.SetRetryConfig(3, 5*time.Minute)

	// Create executor
	executor := &PooledExecutor{
		providerName: cfg.ProviderName,
		baseURL:      cfg.BaseURL,
		modelName:    cfg.ModelName,
		createClient: cfg.CreateClient,
	}

	// Register executor
	manager.RegisterExecutor(executor)

	// Register all API keys
	ctx := context.Background()
	for i, apiKey := range cfg.APIKeys {
		authEntry := &auth.Auth{
			ID:        uuid.New().String(),
			Provider:  cfg.ProviderName,
			Label:     fmt.Sprintf("%s-key-%d", cfg.ProviderName, i+1),
			Status:    auth.StatusActive,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			Metadata: map[string]any{
				"api_key": apiKey,
			},
		}

		_, err := manager.Register(ctx, authEntry)
		if err != nil {
			log.Printf("Warning: Failed to register API key %d: %v", i+1, err)
		}
	}

	log.Printf("PooledProvider[%s]: Registered %d API keys", cfg.ProviderName, len(cfg.APIKeys))

	// Create provider instance
	provider := &PooledProvider{
		providerName:    cfg.ProviderName,
		baseURL:         cfg.BaseURL,
		modelName:       cfg.ModelName,
		manager:         manager,
		executor:        executor,
		currentKeyIndex: 0,
		apiKeys:         cfg.APIKeys,
	}

	// Initialize metrics if enabled (default: true)
	if cfg.EnableMetrics || (!cfg.EnableMetrics && cfg.EnableRotation) {
		provider.metrics = NewPoolMetrics(cfg.ProviderName, len(cfg.APIKeys))
		log.Printf("PooledProvider[%s]: Metrics enabled", cfg.ProviderName)
	}

	// Initialize auto rotation if enabled (default: true)
	if cfg.EnableRotation {
		if provider.metrics == nil {
			provider.metrics = NewPoolMetrics(cfg.ProviderName, len(cfg.APIKeys))
		}

		// Use custom strategy or default to health-based
		strategy := cfg.RotationStrategy
		if strategy == nil {
			strategy = &HealthBasedStrategy{
				MaxConsecutiveFailures: 3,
			}
		}

		provider.rotationManager = NewAutoRotationManager(strategy, provider.metrics, provider)
		
		// FIXED: Create cancellable context for proper cleanup
		rotationCtx, cancel := context.WithCancel(context.Background())
		provider.cancelFunc = cancel
		
		// Start rotation in background with cancellable context
		go provider.rotationManager.Start(rotationCtx)
		
		log.Printf("PooledProvider[%s]: Auto-rotation enabled", cfg.ProviderName)
	}

	return provider, nil
}

// Provider returns the provider name.
func (p *PooledProvider) Provider() string {
	return p.providerName
}

// Model returns the model name.
func (p *PooledProvider) Model() string {
	return p.modelName
}

// Generate implements fantasy.LanguageModel.
func (p *PooledProvider) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	// Convert fantasy.Call to executor.Request
	req := convertCallToRequest(call)

	// Execute with fallback
	resp, err := p.manager.Execute(ctx, []string{p.providerName}, req, executor.Options{})
	
	// Record metrics
	if p.metrics != nil {
		keyID := p.getCurrentKeyID()
		p.metrics.RecordRequest(keyID, err == nil, err)
	}
	
	if err != nil {
		return nil, err
	}

	// Convert executor.Response to fantasy.Response
	return convertResponseToFantasy(resp)
}

// Stream implements fantasy.LanguageModel.
func (p *PooledProvider) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	// Convert fantasy.Call to executor.Request
	req := convertCallToRequest(call)

	// Execute stream with fallback
	chunks, err := p.manager.ExecuteStream(ctx, []string{p.providerName}, req, executor.Options{})
	if err != nil {
		return nil, err
	}

	// Convert to fantasy.StreamResponse
	return convertStreamToFantasy(chunks), nil
}

// GenerateObject implements fantasy.LanguageModel (optional).
func (p *PooledProvider) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, fmt.Errorf("GenerateObject not implemented for pooled provider")
}

// StreamObject implements fantasy.LanguageModel (optional).
func (p *PooledProvider) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, fmt.Errorf("StreamObject not implemented for pooled provider")
}

// GetManager returns the underlying auth.Manager for advanced usage.
func (p *PooledProvider) GetManager() *auth.Manager {
	return p.manager
}

// GetMetrics returns current metrics snapshot
func (p *PooledProvider) GetMetrics() *PoolMetricsSnapshot {
	if p.metrics == nil {
		return nil
	}
	snapshot := p.metrics.GetSnapshot()
	return &snapshot
}

// RotateToKey manually rotates to a specific key index
// FIXED: Added mutex protection to prevent race conditions
func (p *PooledProvider) RotateToKey(index int) {
	if index < 0 || index >= len(p.apiKeys) {
		log.Printf("⚠️  [PooledProvider] Invalid key index: %d", index)
		return
	}
	
	p.indexMu.Lock()
	p.currentKeyIndex = index
	p.indexMu.Unlock()
	
	log.Printf("🔄 [PooledProvider] Rotated to key index: %d", index)
}

// getCurrentKeyID returns the current API key ID for metrics
// FIXED: Added mutex protection to prevent race conditions
func (p *PooledProvider) getCurrentKeyID() string {
	p.indexMu.RLock()
	defer p.indexMu.RUnlock()
	
	if p.currentKeyIndex >= 0 && p.currentKeyIndex < len(p.apiKeys) {
		return p.apiKeys[p.currentKeyIndex]
	}
	return "unknown"
}

// Cleanup stops background services
// FIXED: Added context cancellation to prevent goroutine leaks
func (p *PooledProvider) Cleanup() {
	// Cancel rotation context first
	if p.cancelFunc != nil {
		p.cancelFunc()
		log.Printf("🛑 [PooledProvider] Rotation context cancelled")
	}
	
	// Then stop rotation manager
	if p.rotationManager != nil {
		p.rotationManager.Stop()
		log.Printf("🛑 [PooledProvider] Rotation manager stopped")
	}
}
