package pooled

import (
	"context"
	"log"
	"time"
)

// RotationStrategy defines how keys should be rotated
type RotationStrategy interface {
	// ShouldRotate returns true if rotation is needed
	ShouldRotate(metrics *PoolMetrics) bool
	// SelectNextKey returns the index of the next key to use
	SelectNextKey(currentIndex int, totalKeys int, metrics *PoolMetrics) int
}

// RoundRobinStrategy rotates keys in round-robin fashion
type RoundRobinStrategy struct {
	RotateAfterRequests int64 // Rotate after N requests
}

func (s *RoundRobinStrategy) ShouldRotate(metrics *PoolMetrics) bool {
	if s.RotateAfterRequests <= 0 {
		return false
	}
	
	// Get current key metrics
	currentKeyID := getKeyIDFromIndex(metrics.CurrentKeyIndex, metrics)
	if currentKeyID == "" {
		return false
	}
	
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()
	
	km, exists := metrics.KeyMetrics[currentKeyID]
	if !exists {
		return false
	}
	
	// Rotate if current key has processed enough requests
	return km.TotalRequests >= s.RotateAfterRequests
}

func (s *RoundRobinStrategy) SelectNextKey(currentIndex int, totalKeys int, metrics *PoolMetrics) int {
	// Simple round-robin
	nextIndex := (currentIndex + 1) % totalKeys
	
	// Skip unhealthy keys
	for i := 0; i < totalKeys; i++ {
		keyID := getKeyIDFromIndex(nextIndex, metrics)
		if keyID != "" {
			metrics.mu.RLock()
			km, exists := metrics.KeyMetrics[keyID]
			metrics.mu.RUnlock()
			
			if exists && km.IsHealthy {
				return nextIndex
			}
		}
		nextIndex = (nextIndex + 1) % totalKeys
	}
	
	// If all keys are unhealthy, return next index anyway
	return (currentIndex + 1) % totalKeys
}

// LeastUsedStrategy selects the key with least usage
type LeastUsedStrategy struct {
	CheckInterval time.Duration // How often to check for rotation
	lastCheck     time.Time
}

func (s *LeastUsedStrategy) ShouldRotate(metrics *PoolMetrics) bool {
	now := time.Now()
	if now.Sub(s.lastCheck) < s.CheckInterval {
		return false
	}
	s.lastCheck = now
	return true
}

func (s *LeastUsedStrategy) SelectNextKey(currentIndex int, totalKeys int, metrics *PoolMetrics) int {
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()
	
	// Find key with least requests that is healthy
	minRequests := int64(-1)
	selectedIndex := currentIndex
	
	i := 0
	for keyID, km := range metrics.KeyMetrics {
		if !km.IsHealthy {
			i++
			continue
		}
		
		if minRequests == -1 || km.TotalRequests < minRequests {
			minRequests = km.TotalRequests
			selectedIndex = getIndexFromKeyID(keyID, metrics)
		}
		i++
	}
	
	return selectedIndex
}

// HealthBasedStrategy rotates when current key becomes unhealthy
type HealthBasedStrategy struct {
	MaxConsecutiveFailures int // Rotate after N consecutive failures
}

func (s *HealthBasedStrategy) ShouldRotate(metrics *PoolMetrics) bool {
	currentKeyID := getKeyIDFromIndex(metrics.CurrentKeyIndex, metrics)
	if currentKeyID == "" {
		return false
	}
	
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()
	
	km, exists := metrics.KeyMetrics[currentKeyID]
	if !exists {
		return false
	}
	
	// Rotate if key is unhealthy or rate limited
	return !km.IsHealthy || km.RateLimited > 0
}

func (s *HealthBasedStrategy) SelectNextKey(currentIndex int, totalKeys int, metrics *PoolMetrics) int {
	// Find next healthy key
	for i := 1; i <= totalKeys; i++ {
		nextIndex := (currentIndex + i) % totalKeys
		keyID := getKeyIDFromIndex(nextIndex, metrics)
		
		metrics.mu.RLock()
		
		km, exists := metrics.KeyMetrics[keyID]
		metrics.mu.RUnlock()
		
		if keyID != "" && exists && km.IsHealthy && km.RateLimited == 0 {
			return nextIndex
		}
	}
	
	// If no healthy key found, return next index
	return (currentIndex + 1) % totalKeys
}

// AutoRotationManager manages automatic key rotation
type AutoRotationManager struct {
	strategy RotationStrategy
	metrics  *PoolMetrics
	provider *PooledProvider
	stopCh   chan struct{}
}

// NewAutoRotationManager creates a new rotation manager
func NewAutoRotationManager(strategy RotationStrategy, metrics *PoolMetrics, provider *PooledProvider) *AutoRotationManager {
	return &AutoRotationManager{
		strategy: strategy,
		metrics:  metrics,
		provider: provider,
		stopCh:   make(chan struct{}),
	}
}

// Start begins automatic rotation monitoring
func (m *AutoRotationManager) Start(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second) // Check every 10 seconds
	defer ticker.Stop()
	
	log.Printf("🔄 [AutoRotation] Started for provider: %s", m.metrics.ProviderName)
	
	for {
		select {
		case <-ctx.Done():
			log.Printf("🔄 [AutoRotation] Stopped (context cancelled)")
			return
		case <-m.stopCh:
			log.Printf("🔄 [AutoRotation] Stopped (manual stop)")
			return
		case <-ticker.C:
			m.checkAndRotate()
		}
	}
}

// Stop stops the rotation manager
func (m *AutoRotationManager) Stop() {
	close(m.stopCh)
}

func (m *AutoRotationManager) checkAndRotate() {
	if !m.strategy.ShouldRotate(m.metrics) {
		return
	}
	
	m.metrics.mu.RLock()
	currentIndex := m.metrics.CurrentKeyIndex
	totalKeys := m.metrics.TotalKeys
	m.metrics.mu.RUnlock()
	
	nextIndex := m.strategy.SelectNextKey(currentIndex, totalKeys, m.metrics)
	
	if nextIndex != currentIndex {
		m.provider.RotateToKey(nextIndex)
		m.metrics.RecordRotation(nextIndex)
		
		log.Printf("🔄 [AutoRotation] Rotated from key %d to key %d (provider: %s)", 
			currentIndex, nextIndex, m.metrics.ProviderName)
	}
}

// Helper functions
// FIXED: Use OrderedKeys instead of map iteration to avoid non-determinism

func getKeyIDFromIndex(index int, metrics *PoolMetrics) string {
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()
	
	if index >= 0 && index < len(metrics.OrderedKeys) {
		return metrics.OrderedKeys[index]
	}
	return ""
}

func getIndexFromKeyID(targetKeyID string, metrics *PoolMetrics) int {
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()
	
	for i, keyID := range metrics.OrderedKeys {
		if keyID == targetKeyID {
			return i
		}
	}
	return 0
}

