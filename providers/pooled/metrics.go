package pooled

import (
	"sync"
	"time"
)

// KeyMetrics tracks usage statistics for a single API key
type KeyMetrics struct {
	APIKey        string    // Masked key (first 8 chars only)
	TotalRequests int64     // Total requests made with this key
	SuccessCount  int64     // Successful requests
	FailureCount  int64     // Failed requests
	RateLimited   int64     // Number of times rate limited
	LastUsed      time.Time // Last time this key was used
	LastError     string    // Last error message
	LastErrorTime time.Time // When last error occurred
	IsHealthy     bool      // Current health status
}

// PoolMetrics tracks overall pool statistics
type PoolMetrics struct {
	mu sync.RWMutex

	ProviderName    string                 // Provider name (e.g., "openrouter")
	TotalKeys       int                    // Total number of keys in pool
	HealthyKeys     int                    // Number of healthy keys
	TotalRequests   int64                  // Total requests across all keys
	SuccessRate     float64                // Success rate (0.0 - 1.0)
	KeyMetrics      map[string]*KeyMetrics // Per-key metrics
	OrderedKeys     []string               // FIXED: Ordered list of keys to avoid map iteration non-determinism
	RotationCount   int64                  // Number of times keys were rotated
	LastRotation    time.Time              // Last rotation timestamp
	CreatedAt       time.Time              // When pool was created
	CurrentKeyIndex int                    // Current active key index
}

// NewPoolMetrics creates a new metrics tracker
func NewPoolMetrics(providerName string, totalKeys int) *PoolMetrics {
	return &PoolMetrics{
		ProviderName: providerName,
		TotalKeys:    totalKeys,
		HealthyKeys:  totalKeys,
		KeyMetrics:   make(map[string]*KeyMetrics),
		OrderedKeys:  make([]string, 0, totalKeys), // FIXED: Initialize ordered keys list
		CreatedAt:    time.Now(),
	}
}

// RecordRequest records a request attempt
func (m *PoolMetrics) RecordRequest(keyID string, success bool, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Get or create key metrics
	km, exists := m.KeyMetrics[keyID]
	if !exists {
		km = &KeyMetrics{
			APIKey:    maskAPIKey(keyID),
			IsHealthy: true,
		}
		m.KeyMetrics[keyID] = km
		// FIXED: Add to ordered keys list for deterministic iteration
		m.OrderedKeys = append(m.OrderedKeys, keyID)
	}

	// Update metrics
	km.TotalRequests++
	km.LastUsed = time.Now()
	m.TotalRequests++

	if success {
		km.SuccessCount++
		km.IsHealthy = true
	} else {
		km.FailureCount++
		if err != nil {
			km.LastError = err.Error()
			km.LastErrorTime = time.Now()

			// Check if rate limited
			if isRateLimitError(err) {
				km.RateLimited++
				km.IsHealthy = false
			}
		}
	}

	// Recalculate success rate
	if m.TotalRequests > 0 {
		totalSuccess := int64(0)
		for _, k := range m.KeyMetrics {
			totalSuccess += k.SuccessCount
		}
		m.SuccessRate = float64(totalSuccess) / float64(m.TotalRequests)
	}

	// Recalculate healthy keys
	m.HealthyKeys = 0
	for _, k := range m.KeyMetrics {
		if k.IsHealthy {
			m.HealthyKeys++
		}
	}
}

// RecordRotation records a key rotation event
func (m *PoolMetrics) RecordRotation(newIndex int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.RotationCount++
	m.LastRotation = time.Now()
	m.CurrentKeyIndex = newIndex
}

// GetSnapshot returns a snapshot of current metrics
func (m *PoolMetrics) GetSnapshot() PoolMetricsSnapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()

	snapshot := PoolMetricsSnapshot{
		ProviderName:    m.ProviderName,
		TotalKeys:       m.TotalKeys,
		HealthyKeys:     m.HealthyKeys,
		TotalRequests:   m.TotalRequests,
		SuccessRate:     m.SuccessRate,
		RotationCount:   m.RotationCount,
		LastRotation:    m.LastRotation,
		CreatedAt:       m.CreatedAt,
		CurrentKeyIndex: m.CurrentKeyIndex,
		KeyMetrics:      make([]*KeyMetrics, 0, len(m.KeyMetrics)),
	}

	// Copy key metrics
	for _, km := range m.KeyMetrics {
		kmCopy := *km
		snapshot.KeyMetrics = append(snapshot.KeyMetrics, &kmCopy)
	}

	return snapshot
}

// PoolMetricsSnapshot is a point-in-time snapshot of pool metrics
type PoolMetricsSnapshot struct {
	ProviderName    string
	TotalKeys       int
	HealthyKeys     int
	TotalRequests   int64
	SuccessRate     float64
	RotationCount   int64
	LastRotation    time.Time
	CreatedAt       time.Time
	CurrentKeyIndex int
	KeyMetrics      []*KeyMetrics
}

// Helper functions

func maskAPIKey(key string) string {
	if len(key) <= 8 {
		return "***"
	}
	return key[:8] + "***"
}

func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	return contains(errStr, "rate limit") ||
		contains(errStr, "429") ||
		contains(errStr, "too many requests") ||
		contains(errStr, "quota exceeded")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || len(s) > len(substr) && 
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
		findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

