package openrouter

import (
	_ "embed"
	"encoding/json"
	"strings"
	"sync"
)

//go:embed openrouter.json
var catalogJSON []byte

// ModelInfo contains metadata about a specific model
type ModelInfo struct {
	ID                     string   `json:"id"`
	Name                   string   `json:"name"`
	CostPerMillionIn       float64  `json:"cost_per_1m_in"`
	CostPerMillionOut      float64  `json:"cost_per_1m_out"`
	ContextWindow          int      `json:"context_window"`
	DefaultMaxTokens       int      `json:"default_max_tokens"`
	CanReason              bool     `json:"can_reason"`
	ReasoningLevels        []string `json:"reasoning_levels,omitempty"`
	DefaultReasoningEffort string   `json:"default_reasoning_effort,omitempty"`
	SupportsAttachments    bool     `json:"supports_attachments"`
}

// CatalogConfig represents the embedded openrouter.json structure
type CatalogConfig struct {
	Name              string      `json:"name"`
	ID                string      `json:"id"`
	APIEndpoint       string      `json:"api_endpoint"`
	DefaultLargeModel string      `json:"default_large_model_id"`
	DefaultSmallModel string      `json:"default_small_model_id"`
	Models            []ModelInfo `json:"models"`
}

// Catalog provides access to OpenRouter model catalog
type Catalog struct {
	config     CatalogConfig
	freeModels map[string]*ModelInfo
}

var (
	globalCatalog *Catalog
	catalogOnce   sync.Once
)

// GetCatalog returns the global OpenRouter model catalog (singleton)
func GetCatalog() *Catalog {
	catalogOnce.Do(func() {
		globalCatalog = &Catalog{
			freeModels: make(map[string]*ModelInfo),
		}
		_ = json.Unmarshal(catalogJSON, &globalCatalog.config)

		for i := range globalCatalog.config.Models {
			model := &globalCatalog.config.Models[i]
			if strings.HasSuffix(model.ID, ":free") {
				globalCatalog.freeModels[model.ID] = model
			}
		}
	})
	return globalCatalog
}

// ModelSelectionCriteria defines criteria for dynamic model selection
type ModelSelectionCriteria struct {
	RequireReasoning   bool
	RequireAttachments bool
	MinContextWindow   int
}

// Models to exclude from auto-selection (frequently rate limited on OpenRouter)
var excludedModels = map[string]bool{
	"google/gemini-2.0-flash-exp:free": true,
	"google/gemini-flash-1.5-8b:free":  true,
}

// SelectFreeModel selects the best free model based on criteria
// Priority: 1. Free only, 2. Filter by criteria, 3. Rank by context_window > attachments > max_tokens
func (c *Catalog) SelectFreeModel(criteria ModelSelectionCriteria) *ModelInfo {
	var candidates []*ModelInfo

	for _, model := range c.freeModels {
		// Skip frequently rate-limited models
		if excludedModels[model.ID] {
			continue
		}
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

// compareModels: >0 if a is better, <0 if b is better
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

// GetModel returns a model by ID
func (c *Catalog) GetModel(modelID string) *ModelInfo {
	for i := range c.config.Models {
		if c.config.Models[i].ID == modelID {
			return &c.config.Models[i]
		}
	}
	return nil
}

// ListFreeModels returns all free models, optionally filtered by reasoning
func (c *Catalog) ListFreeModels(requireReasoning bool) []*ModelInfo {
	var models []*ModelInfo
	for _, model := range c.freeModels {
		if requireReasoning && !model.CanReason {
			continue
		}
		models = append(models, model)
	}
	return models
}

// GetDefaultLargeModel returns the default large model ID
func (c *Catalog) GetDefaultLargeModel() string {
	return c.config.DefaultLargeModel
}

// GetDefaultSmallModel returns the default small model ID
func (c *Catalog) GetDefaultSmallModel() string {
	return c.config.DefaultSmallModel
}
