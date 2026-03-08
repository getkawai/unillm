package providertests

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/getkawai/unillm"
	"github.com/getkawai/unillm/providers/google"
	"github.com/getkawai/unillm/providers/openai"
	"github.com/getkawai/unillm/providers/openaicompat"
	"github.com/getkawai/unillm/providers/openrouter"
	"github.com/stretchr/testify/require"
)

func TestProviderLanguageModelContract(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		expectedProvider string
		modelID          string
		newProvider      func(t *testing.T) unillm.Provider
	}{
		{
			name:             "openai",
			expectedProvider: openai.Name,
			modelID:          "gpt-4o-mini",
			newProvider: func(t *testing.T) unillm.Provider {
				t.Helper()
				p, err := openai.New(
					openai.WithAPIKey("test-key"),
					openai.WithBaseURL("https://example.invalid/v1"),
				)
				require.NoError(t, err)
				return p
			},
		},
		{
			name:             "openaicompat",
			expectedProvider: openaicompat.Name,
			modelID:          "xai/grok-4-fast:free",
			newProvider: func(t *testing.T) unillm.Provider {
				t.Helper()
				p, err := openaicompat.New(
					openaicompat.WithAPIKey("test-key"),
					openaicompat.WithBaseURL("https://example.invalid/v1"),
				)
				require.NoError(t, err)
				return p
			},
		},
		{
			name:             "openrouter",
			expectedProvider: openrouter.Name,
			modelID:          "openai/gpt-4o-mini",
			newProvider: func(t *testing.T) unillm.Provider {
				t.Helper()
				p, err := openrouter.New(
					openrouter.WithAPIKey("test-key"),
				)
				require.NoError(t, err)
				return p
			},
		},
		{
			name:             "google",
			expectedProvider: google.Name,
			modelID:          "gemini-1.5-flash",
			newProvider: func(t *testing.T) unillm.Provider {
				t.Helper()
				p, err := google.New(
					google.WithGeminiAPIKey("test-key"),
				)
				require.NoError(t, err)
				return p
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			p := tt.newProvider(t)
			require.Equal(t, tt.expectedProvider, p.Name())

			model, err := p.LanguageModel(context.Background(), tt.modelID)
			require.NoError(t, err)
			require.NotNil(t, model)
			require.Equal(t, tt.expectedProvider, model.Provider())
			require.Equal(t, tt.modelID, model.Model())
		})
	}
}

func TestProviderOptionsJSONRoundTrip(t *testing.T) {
	t.Parallel()

	call := unillm.Call{
		Prompt: unillm.Prompt{unillm.NewUserMessage("hello")},
		ProviderOptions: unillm.ProviderOptions{
			openai.Name: &openai.ProviderOptions{
				ParallelToolCalls: ptr(true),
			},
			openaicompat.Name: &openaicompat.ProviderOptions{
				User: ptr("u-123"),
			},
			google.Name: &google.ProviderOptions{
				ThinkingConfig: &google.ThinkingConfig{
					IncludeThoughts: ptr(true),
				},
			},
			openrouter.Name: &openrouter.ProviderOptions{
				IncludeUsage: ptr(true),
			},
		},
	}

	raw, err := json.Marshal(call)
	require.NoError(t, err)

	var decoded unillm.Call
	require.NoError(t, json.Unmarshal(raw, &decoded))
	require.Len(t, decoded.ProviderOptions, 4)

	_, ok := decoded.ProviderOptions[openai.Name].(*openai.ProviderOptions)
	require.True(t, ok, "expected openai provider options type")

	_, ok = decoded.ProviderOptions[openaicompat.Name].(*openaicompat.ProviderOptions)
	require.True(t, ok, "expected openaicompat provider options type")

	_, ok = decoded.ProviderOptions[google.Name].(*google.ProviderOptions)
	require.True(t, ok, "expected google provider options type")

	_, ok = decoded.ProviderOptions[openrouter.Name].(*openrouter.ProviderOptions)
	require.True(t, ok, "expected openrouter provider options type")
}

func ptr[T any](v T) *T {
	return &v
}
