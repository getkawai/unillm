// Package vercel provides an implementation of the unillm AI SDK for Vercel AI Gateway.
package vercel

import (
	"context"

	"github.com/getkawai/unillm"
	"github.com/getkawai/unillm/providers/openai"
	"github.com/openai/openai-go/v3/option"
)

type options struct {
	openaiOptions        []openai.Option
	languageModelOptions []openai.LanguageModelOption
	sdkOptions           []option.RequestOption
	objectMode           unillm.ObjectMode
	selectionCriteria    ModelSelectionCriteria
}

const (
	// DefaultURL is the default URL for the Vercel AI Gateway API.
	DefaultURL = "https://ai-gateway.vercel.sh/v1"
	// Name is the name of the Vercel provider.
	Name = "vercel"
)

// Option defines a function that configures Vercel provider options.
type Option = func(*options)

// provider wraps openai provider to add model selection.
type provider struct {
	inner    unillm.Provider
	criteria ModelSelectionCriteria
}

// New creates a new Vercel AI Gateway provider with the given options.
func New(opts ...Option) (unillm.Provider, error) {
	providerOptions := options{
		openaiOptions: []openai.Option{
			openai.WithName(Name),
			openai.WithBaseURL(DefaultURL),
		},
		languageModelOptions: []openai.LanguageModelOption{
			openai.WithLanguageModelPrepareCallFunc(languagePrepareModelCall),
			openai.WithLanguageModelUsageFunc(languageModelUsage),
			openai.WithLanguageModelStreamUsageFunc(languageModelStreamUsage),
			openai.WithLanguageModelStreamExtraFunc(languageModelStreamExtra),
			openai.WithLanguageModelExtraContentFunc(languageModelExtraContent),
			openai.WithLanguageModelToPromptFunc(languageModelToPrompt),
		},
		objectMode: unillm.ObjectModeTool, // Default to tool mode for vercel
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	// Handle object mode: convert unsupported modes to tool
	// Vercel AI Gateway doesn't support native JSON mode, so we use tool or text
	objectMode := providerOptions.objectMode
	if objectMode == unillm.ObjectModeAuto || objectMode == unillm.ObjectModeJSON {
		objectMode = unillm.ObjectModeTool
	}

	providerOptions.openaiOptions = append(
		providerOptions.openaiOptions,
		openai.WithSDKOptions(providerOptions.sdkOptions...),
		openai.WithLanguageModelOptions(providerOptions.languageModelOptions...),
		openai.WithObjectMode(objectMode),
	)
	inner, err := openai.New(providerOptions.openaiOptions...)
	if err != nil {
		return nil, err
	}
	return &provider{
		inner:    inner,
		criteria: providerOptions.selectionCriteria,
	}, nil
}

// Name implements unillm.Provider.
func (p *provider) Name() string {
	return Name
}

// LanguageModel implements unillm.Provider with optional auto model selection.
func (p *provider) LanguageModel(ctx context.Context, modelID string) (unillm.LanguageModel, error) {
	if modelID == "" {
		catalog := GetCatalog()
		if p.criteria != (ModelSelectionCriteria{}) {
			if selected := catalog.SelectModel(p.criteria); selected != nil {
				modelID = selected.ID
			}
		}
		if modelID == "" {
			modelID = catalog.GetDefaultSmallModel()
		}
		if modelID == "" {
			modelID = catalog.GetDefaultLargeModel()
		}
	}
	return p.inner.LanguageModel(ctx, modelID)
}

// WithModelSelection sets model selection criteria for auto-selecting models.
func WithModelSelection(criteria ModelSelectionCriteria) Option {
	return func(o *options) {
		o.selectionCriteria = criteria
	}
}

// WithAPIKey sets the API key for the Vercel provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithAPIKey(apiKey))
	}
}

// WithBaseURL sets the base URL for the Vercel provider.
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithBaseURL(url))
	}
}

// WithName sets the name for the Vercel provider.
func WithName(name string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithName(name))
	}
}

// WithHeaders sets the headers for the Vercel provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the Vercel provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHTTPClient(client))
	}
}

// WithSDKOptions sets the SDK options for the Vercel provider.
func WithSDKOptions(opts ...option.RequestOption) Option {
	return func(o *options) {
		o.sdkOptions = append(o.sdkOptions, opts...)
	}
}

// WithObjectMode sets the object generation mode for the Vercel provider.
// Supported modes: ObjectModeTool, ObjectModeText.
// ObjectModeAuto and ObjectModeJSON are automatically converted to ObjectModeTool
// since Vercel AI Gateway doesn't support native JSON mode.
func WithObjectMode(om unillm.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}
