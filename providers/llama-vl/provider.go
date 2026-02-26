// Package llamavl provides a Vision-Language (VL) provider for local llama.cpp models.
// This provider handles image processing and description using VL models like Qwen-VL.
// For text generation, use unillm/providers/llama.
package llamavl

import (
	"context"
	"fmt"

	"github.com/getkawai/llamalib"
)

const (
	// Name is the name of the llama-vl provider.
	Name = "llama-vl"
)

// Provider interface for Vision-Language capabilities.
type Provider interface {
	// ProcessImage processes an image with a text prompt and returns a description.
	ProcessImage(ctx context.Context, imagePath, prompt string, maxTokens int32) (string, error)

	// IsVLModelLoaded returns true if a VL model is currently loaded.
	IsVLModelLoaded() bool

	// LoadVLModel loads a Vision-Language model.
	// If modelPath is empty, automatically selects the best available VL model.
	LoadVLModel(ctx context.Context, modelPath string) error

	// GetService returns the underlying llamalib.Service.
	GetService() *llamalib.Service

	// Name returns the provider name.
	Name() string

	// Cleanup releases all resources held by the provider.
	Cleanup()
}

type provider struct {
	options options
}

type options struct {
	name    string
	service *llamalib.Service
}

// Option defines a function that configures llama-vl provider options.
type Option = func(*options)

// New creates a new llama-vl provider with the given options.
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

// WithName sets the name for the llama-vl provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithService sets a pre-configured llamalib.Service.
// Use this to share the service with other providers (e.g., llama text provider).
func WithService(service *llamalib.Service) Option {
	return func(o *options) {
		o.service = service
	}
}

// ProcessImage processes an image with accompanying text using VL model.
func (p *provider) ProcessImage(ctx context.Context, imagePath, prompt string, maxTokens int32) (string, error) {
	// Ensure library is initialized
	if err := p.options.service.WaitForInitialization(ctx); err != nil {
		return "", fmt.Errorf("library initialization failed: %w", err)
	}

	// Check if VL model is loaded
	if !p.options.service.IsVLModelLoaded() {
		// Try to auto-load VL model
		if err := p.options.service.LoadVLModel(""); err != nil {
			return "", fmt.Errorf("VL model not loaded and failed to auto-load: %w", err)
		}
	}

	return p.options.service.ProcessImageWithText(imagePath, prompt, maxTokens)
}

// IsVLModelLoaded returns true if a VL model is currently loaded.
func (p *provider) IsVLModelLoaded() bool {
	return p.options.service.IsVLModelLoaded()
}

// LoadVLModel loads a Vision-Language model.
func (p *provider) LoadVLModel(ctx context.Context, modelPath string) error {
	// Ensure library is initialized
	if err := p.options.service.WaitForInitialization(ctx); err != nil {
		return fmt.Errorf("library initialization failed: %w", err)
	}

	return p.options.service.LoadVLModel(modelPath)
}

// GetService returns the underlying llamalib.Service.
func (p *provider) GetService() *llamalib.Service {
	return p.options.service
}

// Name returns the provider name.
func (p *provider) Name() string {
	return p.options.name
}

// Cleanup releases all resources held by the provider.
func (p *provider) Cleanup() {
	if p.options.service != nil {
		p.options.service.Cleanup()
	}
}

// Ensure provider implements Provider interface
var _ Provider = (*provider)(nil)
