// Package openrouter provides an implementation of the unillm AI SDK for OpenRouter's language models.
package openrouter

import (
	"context"
	"encoding/json"

	"github.com/getkawai/unillm"
	"github.com/getkawai/unillm/providers/openai"
	"github.com/openai/openai-go/v3/option"
)

type options struct {
	openaiOptions        []openai.Option
	languageModelOptions []openai.LanguageModelOption
	objectMode           unillm.ObjectMode
	selectionCriteria    ModelSelectionCriteria
}

// provider wraps openai provider to add model selection
type provider struct {
	inner    unillm.Provider
	criteria ModelSelectionCriteria
}

const (
	// DefaultURL is the default URL for the OpenRouter API.
	DefaultURL = "https://openrouter.ai/api/v1"
	// Name is the name of the OpenRouter provider.
	Name = "openrouter"
)

// Option defines a function that configures OpenRouter provider options.
type Option = func(*options)

// New creates a new OpenRouter provider with the given options.
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
		objectMode: unillm.ObjectModeTool, // Default to tool mode for openrouter
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	// Handle object mode: convert unsupported modes to tool
	// OpenRouter doesn't support native JSON mode, so we use tool or text
	objectMode := providerOptions.objectMode
	if objectMode == unillm.ObjectModeAuto || objectMode == unillm.ObjectModeJSON {
		objectMode = unillm.ObjectModeTool
	}

	providerOptions.openaiOptions = append(
		providerOptions.openaiOptions,
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

// LanguageModel implements unillm.Provider with auto model selection.
// If modelID is empty, selects the best free model based on criteria.
func (p *provider) LanguageModel(ctx context.Context, modelID string) (unillm.LanguageModel, error) {
	if modelID == "" {
		catalog := GetCatalog()
		if selected := catalog.SelectFreeModel(p.criteria); selected != nil {
			modelID = selected.ID
		}
	}
	return p.inner.LanguageModel(ctx, modelID)
}

// WithModelSelection sets model selection criteria for auto-selecting free models.
func WithModelSelection(criteria ModelSelectionCriteria) Option {
	return func(o *options) {
		o.selectionCriteria = criteria
	}
}

// WithAPIKey sets the API key for the OpenRouter provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithAPIKey(apiKey))
	}
}

// WithName sets the name for the OpenRouter provider.
func WithName(name string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithName(name))
	}
}

// WithHeaders sets the headers for the OpenRouter provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the OpenRouter provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHTTPClient(client))
	}
}

// WithObjectMode sets the object generation mode for the OpenRouter provider.
// Supported modes: ObjectModeTool, ObjectModeText.
// ObjectModeAuto and ObjectModeJSON are automatically converted to ObjectModeTool
// since OpenRouter doesn't support native JSON mode.
func WithObjectMode(om unillm.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}

func structToMapJSON(s any) (map[string]any, error) {
	var result map[string]any
	jsonBytes, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(jsonBytes, &result)
	if err != nil {
		return nil, err
	}
	return result, nil
}
