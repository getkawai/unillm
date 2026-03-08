// Package llamavl provides a Vision-Language (VL) provider for local llama.cpp models.
// This provider handles image processing and description using VL models like Qwen-VL.
// For text generation, use unillm/providers/llama.
package llamavl

import (
	"context"
	"fmt"
	"sync"

	"github.com/getkawai/llamalib"
	"github.com/getkawai/unillm/internal/llamautil"
)

const (
	// Name is the name of the llama-vl provider.
	Name = "llama-vl"
)

// Service keeps runtime and selected VL model metadata.
type Service struct {
	mu sync.Mutex

	installer     *llamalib.LlamaCppInstaller
	loadedVLModel string
}

// NewService creates a new VL service.
func NewService() *Service {
	return &Service{
		installer: llamalib.NewLlamaCppInstaller(),
	}
}

func (s *Service) ensureRuntime() error {
	return llamautil.EnsureLlamaRuntime(s.installer)
}

// WaitForInitialization ensures llama runtime is available.
func (s *Service) WaitForInitialization(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	return s.ensureRuntime()
}

// LoadVLModel resolves and records the VL model path for future use.
// For llamalib >= 0.2.3, full VL model/context initialization is intentionally
// deferred; this method validates runtime availability via s.ensureRuntime and records
// s.loadedVLModel after resolving/downloading via installer.AutoDownloadRecommendedVLModel.
func (s *Service) LoadVLModel(modelPath string) error {
	if err := s.ensureRuntime(); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if modelPath == "" {
		available, err := s.installer.GetAvailableVLModels()
		if err != nil {
			return fmt.Errorf("list available VL models: %w", err)
		}
		if len(available) == 0 {
			if err := s.installer.AutoDownloadRecommendedVLModel(); err != nil {
				return fmt.Errorf("auto-download VL model: %w", err)
			}
			available, err = s.installer.GetAvailableVLModels()
			if err != nil {
				return fmt.Errorf("list available VL models after download: %w", err)
			}
		}
		if len(available) == 0 {
			return fmt.Errorf("no VL model available")
		}
		s.loadedVLModel = available[0]
		return nil
	}

	s.loadedVLModel = modelPath
	return nil
}

// IsVLModelLoaded reports whether a VL model path has been selected.
func (s *Service) IsVLModelLoaded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loadedVLModel != ""
}

// Cleanup releases service metadata/resources.
func (s *Service) Cleanup() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loadedVLModel = ""
}

// Provider interface for Vision-Language capabilities.
type Provider interface {
	// ProcessImage processes an image with a text prompt and returns a description.
	ProcessImage(ctx context.Context, imagePath, prompt string, maxTokens int32) (string, error)

	// IsVLModelLoaded returns true if a VL model is currently loaded.
	IsVLModelLoaded() bool

	// LoadVLModel loads a Vision-Language model.
	// If modelPath is empty, automatically selects the best available VL model.
	LoadVLModel(ctx context.Context, modelPath string) error

	// GetService returns the underlying service.
	GetService() *Service

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
	service *Service
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

	if providerOptions.service == nil {
		providerOptions.service = NewService()
	}

	return &provider{options: providerOptions}, nil
}

// WithName sets the name for the llama-vl provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithService sets a pre-configured service.
func WithService(service *Service) Option {
	return func(o *options) {
		o.service = service
	}
}

// ProcessImage processes an image with accompanying text using VL model.
func (p *provider) ProcessImage(ctx context.Context, imagePath, prompt string, maxTokens int32) (string, error) {
	if err := p.options.service.WaitForInitialization(ctx); err != nil {
		return "", fmt.Errorf("library initialization failed: %w", err)
	}
	if !p.options.service.IsVLModelLoaded() {
		if err := p.options.service.LoadVLModel(""); err != nil {
			return "", fmt.Errorf("VL model not loaded and failed to auto-load: %w", err)
		}
	}

	return "", fmt.Errorf("llama-vl image processing is not implemented for llamalib >= 0.2.3 yet")
}

// IsVLModelLoaded returns true if a VL model is currently loaded.
func (p *provider) IsVLModelLoaded() bool {
	return p.options.service.IsVLModelLoaded()
}

// LoadVLModel loads a Vision-Language model.
func (p *provider) LoadVLModel(ctx context.Context, modelPath string) error {
	if err := p.options.service.WaitForInitialization(ctx); err != nil {
		return fmt.Errorf("library initialization failed: %w", err)
	}

	return p.options.service.LoadVLModel(modelPath)
}

// GetService returns the underlying service.
func (p *provider) GetService() *Service {
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
