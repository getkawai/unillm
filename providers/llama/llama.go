// Package llama provides an implementation of the unillm AI SDK for local llama.cpp models.
// This provider handles text generation using llama.cpp.
// For Vision-Language (VL) capabilities, use unillm/providers/llama-vl.
package llama

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/getkawai/llamalib"
	"github.com/getkawai/tools"
	"github.com/getkawai/unillm"
)

const (
	// Name is the name of the llama provider.
	Name = "llama"
)

// Provider extends unillm.Provider with llama-specific capabilities.
type Provider interface {
	unillm.Provider

	// Resource management
	GetService() *llamalib.Service
	Cleanup()
}

type provider struct {
	options options
}

type options struct {
	name         string
	service      *llamalib.Service
	toolRegistry *tools.ToolRegistry
	modelPath    string
}

// Option defines a function that configures llama provider options.
type Option = func(*options)

// New creates a new llama provider with the given options.
// Returns Provider interface which extends unillm.Provider.
func New(opts ...Option) (Provider, error) {
	providerOptions := options{
		name: Name,
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	// Create Service if not provided
	if providerOptions.service == nil {
		service := llamalib.NewService()
		providerOptions.service = service
	}

	return &provider{options: providerOptions}, nil
}

// WithName sets the name for the llama provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithService sets a pre-configured llamalib.Service.
func WithService(service *llamalib.Service) Option {
	return func(o *options) {
		o.service = service
	}
}

// WithToolRegistry sets the tool registry for tool calling support.
func WithToolRegistry(registry *tools.ToolRegistry) Option {
	return func(o *options) {
		o.toolRegistry = registry
	}
}

// WithModelPath sets a specific model path to load.
func WithModelPath(modelPath string) Option {
	return func(o *options) {
		o.modelPath = modelPath
	}
}

// LanguageModel implements unillm.Provider.
func (p *provider) LanguageModel(ctx context.Context, modelID string) (unillm.LanguageModel, error) {
	// Wait for library initialization
	if err := p.options.service.WaitForInitialization(ctx); err != nil {
		return nil, fmt.Errorf("library initialization failed: %w", err)
	}

	// Load model if not loaded or if a specific model is requested
	modelPath := p.options.modelPath
	if modelID != "" {
		modelPath = modelID
	}

	if !p.options.service.IsChatModelLoaded() || modelPath != "" {
		if err := p.options.service.LoadChatModel(modelPath); err != nil {
			return nil, fmt.Errorf("failed to load chat model: %w", err)
		}
	}

	// Use only filename for modelID, not full path
	loadedModel := p.options.service.GetLoadedChatModel()
	modelName := filepath.Base(loadedModel)

	return newLanguageModel(
		modelName,
		p.options.name,
		p.options.service,
		p.options.toolRegistry,
	), nil
}

// Name implements unillm.Provider.
func (p *provider) Name() string {
	return p.options.name
}

// GetService returns the underlying llamalib.Service for advanced usage.
func (p *provider) GetService() *llamalib.Service {
	return p.options.service
}

// Cleanup releases all resources held by the provider.
func (p *provider) Cleanup() {
	if p.options.service != nil {
		p.options.service.Cleanup()
	}
}

// Ensure provider implements Provider interface
var _ Provider = (*provider)(nil)
