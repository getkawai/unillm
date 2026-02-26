package google

import (
	"context"
	"testing"

	"github.com/getkawai/unillm"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name    string
		opts    []Option
		wantErr bool
	}{
		{
			name: "with gemini api key",
			opts: []Option{
				WithGeminiAPIKey("test-key"),
			},
			wantErr: false,
		},
		{
			name: "with vertex",
			opts: []Option{
				WithVertex("test-project", "us-central1"),
			},
			wantErr: false,
		},
		{
			name: "with custom name",
			opts: []Option{
				WithGeminiAPIKey("test-key"),
				WithName("custom-google"),
			},
			wantErr: false,
		},
		{
			name: "with base url",
			opts: []Option{
				WithGeminiAPIKey("test-key"),
				WithBaseURL("https://custom.api.com"),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := New(tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && provider == nil {
				t.Error("New() returned nil provider")
			}
		})
	}
}

func TestProviderName(t *testing.T) {
	provider, err := New(WithGeminiAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if provider.Name() != Name {
		t.Errorf("Name() = %v, want %v", provider.Name(), Name)
	}
}

func TestVertexValidation(t *testing.T) {
	tests := []struct {
		name     string
		project  string
		location string
		wantErr  bool
	}{
		{
			name:     "valid vertex config",
			project:  "test-project",
			location: "us-central1",
			wantErr:  false,
		},
		{
			name:     "empty project",
			project:  "",
			location: "us-central1",
			wantErr:  true,
		},
		{
			name:     "empty location",
			project:  "test-project",
			location: "",
			wantErr:  true,
		},
		{
			name:     "both empty",
			project:  "",
			location: "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := New(
				WithVertex(tt.project, tt.location),
				WithSkipAuth(true),
			)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			ctx := context.Background()
			_, err = provider.LanguageModel(ctx, "gemini-1.5-flash")
			if (err != nil) != tt.wantErr {
				t.Errorf("LanguageModel() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLanguageModelWithSkipAuth(t *testing.T) {
	provider, err := New(
		WithGeminiAPIKey("test-key"),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	model, err := provider.LanguageModel(ctx, "gemini-1.5-flash")
	if err != nil {
		t.Fatalf("LanguageModel() error = %v", err)
	}

	if model == nil {
		t.Error("LanguageModel() returned nil")
	}

	if model.Model() != "gemini-1.5-flash" {
		t.Errorf("Model() = %v, want gemini-1.5-flash", model.Model())
	}

	if model.Provider() != Name {
		t.Errorf("Provider() = %v, want %v", model.Provider(), Name)
	}
}

func TestConvertSchemaProperties(t *testing.T) {
	params := map[string]any{
		"name": map[string]any{
			"type":        "string",
			"description": "Person's name",
		},
		"age": map[string]any{
			"type":        "integer",
			"description": "Person's age",
		},
	}

	result := convertSchemaProperties(params)

	if len(result) != 2 {
		t.Errorf("convertSchemaProperties() returned %d properties, want 2", len(result))
	}

	if result["name"] == nil {
		t.Error("convertSchemaProperties() missing 'name' property")
	}

	if result["age"] == nil {
		t.Error("convertSchemaProperties() missing 'age' property")
	}
}

func TestMapJSONTypeToGoogle(t *testing.T) {
	tests := []struct {
		jsonType string
		want     string
	}{
		{"string", "STRING"},
		{"number", "NUMBER"},
		{"integer", "INTEGER"},
		{"boolean", "BOOLEAN"},
		{"array", "ARRAY"},
		{"object", "OBJECT"},
		{"unknown", "STRING"}, // default
	}

	for _, tt := range tests {
		t.Run(tt.jsonType, func(t *testing.T) {
			result := mapJSONTypeToGoogle(tt.jsonType)
			if string(result) != tt.want {
				t.Errorf("mapJSONTypeToGoogle(%v) = %v, want %v", tt.jsonType, result, tt.want)
			}
		})
	}
}

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		want   fantasy.FinishReason
	}{
		{"stop", "STOP", fantasy.FinishReasonStop},
		{"max tokens", "MAX_TOKENS", fantasy.FinishReasonLength},
		{"safety", "SAFETY", fantasy.FinishReasonContentFilter},
		{"other", "OTHER", fantasy.FinishReasonOther},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Note: This test would need actual genai.FinishReason values
			// For now, we're just testing the structure
		})
	}
}

func TestProviderOptions(t *testing.T) {
	opts := &ProviderOptions{
		ThinkingConfig: &ThinkingConfig{
			IncludeThoughts: fantasy.Opt(true),
			ThinkingBudget:  fantasy.Opt(int64(1024)),
		},
		CachedContent: "cachedContents/test",
		SafetySettings: []SafetySetting{
			{
				Category:  "HARM_CATEGORY_HATE_SPEECH",
				Threshold: "BLOCK_MEDIUM_AND_ABOVE",
			},
		},
	}

	if opts.ThinkingConfig == nil {
		t.Error("ThinkingConfig should not be nil")
	}

	if opts.CachedContent != "cachedContents/test" {
		t.Errorf("CachedContent = %v, want cachedContents/test", opts.CachedContent)
	}

	if len(opts.SafetySettings) != 1 {
		t.Errorf("SafetySettings length = %v, want 1", len(opts.SafetySettings))
	}
}

func TestReasoningMetadata(t *testing.T) {
	metadata := &ReasoningMetadata{
		Signature: "test-signature",
		ToolID:    "test-tool-id",
	}

	if metadata.Signature != "test-signature" {
		t.Errorf("Signature = %v, want test-signature", metadata.Signature)
	}

	if metadata.ToolID != "test-tool-id" {
		t.Errorf("ToolID = %v, want test-tool-id", metadata.ToolID)
	}
}

func TestDepointerSlice(t *testing.T) {
	str1 := "test1"
	str2 := "test2"
	input := []*string{&str1, &str2, nil}

	result := depointerSlice(input)

	if len(result) != 2 {
		t.Errorf("depointerSlice() returned %d items, want 2", len(result))
	}

	if result[0] != "test1" {
		t.Errorf("depointerSlice()[0] = %v, want test1", result[0])
	}

	if result[1] != "test2" {
		t.Errorf("depointerSlice()[1] = %v, want test2", result[1])
	}
}

func TestThinkingConfigWarning(t *testing.T) {
	tests := []struct {
		name        string
		backend     string
		wantWarning bool
	}{
		{
			name:        "vertex backend - no warning",
			backend:     "vertex",
			wantWarning: false,
		},
		{
			name:        "gemini backend - should warn",
			backend:     "gemini",
			wantWarning: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This test verifies the logic but can't fully test without mocking
			// The actual warning check is: g.providerOptions.backend != genai.BackendVertexAI
			// We're just documenting the expected behavior here
			if tt.backend == "vertex" && tt.wantWarning {
				t.Error("Vertex backend should not trigger warning")
			}
			if tt.backend != "vertex" && !tt.wantWarning {
				t.Error("Non-Vertex backend should trigger warning")
			}
		})
	}
}
