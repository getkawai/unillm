package vercel

import (
	_ "embed"
	"encoding/json"
	"sync"
)

//go:embed vercel.json
var catalogJSON []byte

// ModelInfo contains metadata about a specific model.
type ModelInfo struct {
	ID                      string         `json:"id"`
	Name                    string         `json:"name"`
	CostPerMillionIn        float64        `json:"cost_per_1m_in"`
	CostPerMillionOut       float64        `json:"cost_per_1m_out"`
	CostPerMillionInCached  float64        `json:"cost_per_1m_in_cached"`
	CostPerMillionOutCached float64        `json:"cost_per_1m_out_cached"`
	ContextWindow           int            `json:"context_window"`
	DefaultMaxTokens        int            `json:"default_max_tokens"`
	CanReason               bool           `json:"can_reason"`
	ReasoningLevels         []string       `json:"reasoning_levels,omitempty"`
	DefaultReasoningEffort  string         `json:"default_reasoning_effort,omitempty"`
	SupportsAttachments     bool           `json:"supports_attachments"`
	Options                 map[string]any `json:"options"`
}

// CatalogConfig represents the embedded vercel.json structure.
type CatalogConfig struct {
	Name              string            `json:"name"`
	ID                string            `json:"id"`
	APIKey            string            `json:"api_key"`
	APIEndpoint       string            `json:"api_endpoint"`
	Type              string            `json:"type"`
	DefaultLargeModel string            `json:"default_large_model_id"`
	DefaultSmallModel string            `json:"default_small_model_id"`
	Models            []ModelInfo       `json:"models"`
	DefaultHeaders    map[string]string `json:"default_headers"`
}

// Catalog provides access to the Vercel model catalog.
type Catalog struct {
	config CatalogConfig
}

var (
	globalCatalog *Catalog
	catalogOnce   sync.Once
)

// GetCatalog returns the global Vercel model catalog (singleton).
func GetCatalog() *Catalog {
	catalogOnce.Do(func() {
		globalCatalog = &Catalog{}
		_ = json.Unmarshal(catalogJSON, &globalCatalog.config)
	})
	return globalCatalog
}

// ModelSelectionCriteria defines criteria for dynamic model selection.
type ModelSelectionCriteria struct {
	RequireReasoning   bool
	RequireAttachments bool
	MinContextWindow   int
}

// SelectModel selects the best model based on criteria.
// Priority: 1. Filter by criteria, 2. Rank by context_window > attachments > max_tokens.
func (c *Catalog) SelectModel(criteria ModelSelectionCriteria) *ModelInfo {
	var candidates []*ModelInfo

	for i := range c.config.Models {
		model := &c.config.Models[i]
		if criteria.RequireReasoning && !model.CanReason {
			continue
		}
		if criteria.RequireAttachments && !model.SupportsAttachments {
			continue
		}
		if criteria.MinContextWindow > 0 && model.ContextWindow < criteria.MinContextWindow {
			continue
		}
		candidates = append(candidates, model)
	}

	if len(candidates) == 0 {
		return nil
	}

	best := candidates[0]
	for _, model := range candidates[1:] {
		if compareModels(model, best) > 0 {
			best = model
		}
	}
	return best
}

// compareModels: >0 if a is better, <0 if b is better.
func compareModels(a, b *ModelInfo) int {
	if a.ContextWindow != b.ContextWindow {
		return a.ContextWindow - b.ContextWindow
	}
	aAttach, bAttach := 0, 0
	if a.SupportsAttachments {
		aAttach = 1
	}
	if b.SupportsAttachments {
		bAttach = 1
	}
	if aAttach != bAttach {
		return aAttach - bAttach
	}
	return a.DefaultMaxTokens - b.DefaultMaxTokens
}

// GetModel returns a model by ID.
func (c *Catalog) GetModel(modelID string) *ModelInfo {
	for i := range c.config.Models {
		if c.config.Models[i].ID == modelID {
			return &c.config.Models[i]
		}
	}
	return nil
}

// ListModels returns all models.
func (c *Catalog) ListModels() []*ModelInfo {
	models := make([]*ModelInfo, 0, len(c.config.Models))
	for i := range c.config.Models {
		models = append(models, &c.config.Models[i])
	}
	return models
}

// GetDefaultLargeModel returns the default large model ID.
func (c *Catalog) GetDefaultLargeModel() string {
	return c.config.DefaultLargeModel
}

// GetDefaultSmallModel returns the default small model ID.
func (c *Catalog) GetDefaultSmallModel() string {
	return c.config.DefaultSmallModel
}
