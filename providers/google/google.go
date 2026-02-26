package google

import (
	"cmp"
	"context"
	"errors"
	"maps"
	"net/http"

	"cloud.google.com/go/auth"
	"github.com/google/uuid"
	"github.com/getkawai/unillm"
	"google.golang.org/genai"
)

// Name is the name of the Google provider.
const Name = "google"

type provider struct {
	options options
}

// ToolCallIDFunc defines a function that generates a tool call ID.
type ToolCallIDFunc = func() string

type options struct {
	apiKey         string
	name           string
	baseURL        string
	headers        map[string]string
	client         *http.Client
	backend        genai.Backend
	project        string
	location       string
	skipAuth       bool
	toolCallIDFunc ToolCallIDFunc
	objectMode     fantasy.ObjectMode
}

// Option defines a function that configures Google provider options.
type Option = func(*options)

// New creates a new Google provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := options{
		headers: map[string]string{},
		toolCallIDFunc: func() string {
			return uuid.NewString()
		},
	}
	for _, o := range opts {
		o(&options)
	}

	options.name = cmp.Or(options.name, Name)

	return &provider{
		options: options,
	}, nil
}

// WithBaseURL sets the base URL for the Google provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithGeminiAPIKey sets the Gemini API key for the Google provider.
func WithGeminiAPIKey(apiKey string) Option {
	return func(o *options) {
		o.backend = genai.BackendGeminiAPI
		o.apiKey = apiKey
		o.project = ""
		o.location = ""
	}
}

// WithVertex configures the Google provider to use Vertex AI.
// Both project and location must be non-empty strings.
func WithVertex(project, location string) Option {
	return func(o *options) {
		o.backend = genai.BackendVertexAI
		o.apiKey = ""
		o.project = project
		o.location = location
	}
}

// WithSkipAuth configures whether to skip authentication for the Google provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}

// WithName sets the name for the Google provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithHeaders sets the headers for the Google provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		maps.Copy(o.headers, headers)
	}
}

// WithHTTPClient sets the HTTP client for the Google provider.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.client = client
	}
}

// WithToolCallIDFunc sets the function that generates a tool call ID.
func WithToolCallIDFunc(f ToolCallIDFunc) Option {
	return func(o *options) {
		o.toolCallIDFunc = f
	}
}

// WithObjectMode sets the object generation mode for the Google provider.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}

// Name implements fantasy.Provider.
func (*provider) Name() string {
	return Name
}

// LanguageModel implements fantasy.Provider.
func (a *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	// Validate Vertex AI configuration
	if a.options.backend == genai.BackendVertexAI {
		if a.options.project == "" || a.options.location == "" {
			return nil, errors.New("project and location must be provided for Vertex AI backend")
		}
	}

	cc := &genai.ClientConfig{
		HTTPClient: a.options.client,
		Backend:    a.options.backend,
		APIKey:     a.options.apiKey,
		Project:    a.options.project,
		Location:   a.options.location,
	}
	if a.options.skipAuth {
		cc.Credentials = &auth.Credentials{TokenProvider: dummyTokenProvider{}}
	} else if cc.Backend == genai.BackendVertexAI {
		if err := cc.UseDefaultCredentials(); err != nil {
			return nil, err
		}
	}

	if a.options.baseURL != "" || len(a.options.headers) > 0 {
		headers := http.Header{}
		for k, v := range a.options.headers {
			headers.Add(k, v)
		}
		cc.HTTPOptions = genai.HTTPOptions{
			BaseURL: a.options.baseURL,
			Headers: headers,
		}
	}
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, err
	}

	objectMode := a.options.objectMode
	if objectMode == "" {
		objectMode = fantasy.ObjectModeAuto
	}

	return &languageModel{
		modelID:         modelID,
		provider:        a.options.name,
		providerOptions: a.options,
		client:          client,
		objectMode:      objectMode,
	}, nil
}
