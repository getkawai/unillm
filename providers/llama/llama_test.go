package llama

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/getkawai/unillm"
	"github.com/stretchr/testify/require"
)

func TestProviderName(t *testing.T) {
	t.Parallel()

	t.Run("should return default provider name", func(t *testing.T) {
		t.Parallel()

		// Create provider without LibraryService (will fail but we can test Name)
		p := &provider{
			options: options{
				name: Name,
			},
		}

		require.Equal(t, "llama", p.Name())
	})

	t.Run("should return custom provider name", func(t *testing.T) {
		t.Parallel()

		p := &provider{
			options: options{
				name: "custom-llama",
			},
		}

		require.Equal(t, "custom-llama", p.Name())
	})
}

func TestProviderOptions(t *testing.T) {
	t.Parallel()

	t.Run("WithName should set provider name", func(t *testing.T) {
		t.Parallel()

		opts := options{}
		WithName("my-llama")(&opts)

		require.Equal(t, "my-llama", opts.name)
	})

	t.Run("WithModelPath should set model path", func(t *testing.T) {
		t.Parallel()

		opts := options{}
		WithModelPath("/path/to/model.gguf")(&opts)

		require.Equal(t, "/path/to/model.gguf", opts.modelPath)
	})
}

func TestLanguageModelInterface(t *testing.T) {
	t.Parallel()

	t.Run("should implement fantasy.LanguageModel interface", func(t *testing.T) {
		t.Parallel()

		model := &languageModel{
			provider: "llama",
			modelID:  "test-model.gguf",
		}

		// Verify interface compliance
		var _ fantasy.LanguageModel = model

		require.Equal(t, "llama", model.Provider())
		require.Equal(t, "test-model.gguf", model.Model())
	})
}

func TestCleanToolCallTags(t *testing.T) {
	t.Parallel()

	model := &languageModel{}

	t.Run("should remove tool_call tags from response", func(t *testing.T) {
		t.Parallel()

		response := `Here is my answer.
<tool_call>
{"name": "web_search", "arguments": {"query": "AI news"}}
</tool_call>
Let me search for that.`

		cleaned := model.cleanToolCallTags(response)

		require.NotContains(t, cleaned, "<tool_call>")
		require.NotContains(t, cleaned, "</tool_call>")
		require.Contains(t, cleaned, "Here is my answer.")
		require.Contains(t, cleaned, "Let me search for that.")
	})

	t.Run("should handle multiple tool_call tags", func(t *testing.T) {
		t.Parallel()

		response := `<tool_call>
{"name": "search", "arguments": {"query": "test1"}}
</tool_call>
Some text
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>`

		cleaned := model.cleanToolCallTags(response)

		require.NotContains(t, cleaned, "<tool_call>")
		require.NotContains(t, cleaned, "</tool_call>")
		require.Contains(t, cleaned, "Some text")
	})

	t.Run("should handle response without tool_call tags", func(t *testing.T) {
		t.Parallel()

		response := "Just a simple response without any tool calls."

		cleaned := model.cleanToolCallTags(response)

		require.Equal(t, response, cleaned)
	})

	t.Run("should handle empty response", func(t *testing.T) {
		t.Parallel()

		cleaned := model.cleanToolCallTags("")

		require.Equal(t, "", cleaned)
	})
}

func TestEnhanceWithTools(t *testing.T) {
	t.Parallel()

	model := &languageModel{}

	t.Run("should return original messages when no tools provided", func(t *testing.T) {
		t.Parallel()

		messages := fantasy.Prompt{
			fantasy.NewUserMessage("Hello"),
		}

		result := model.enhanceWithTools(messages, nil)

		require.Equal(t, messages, result)
	})

	t.Run("should return original messages when empty tools provided", func(t *testing.T) {
		t.Parallel()

		messages := fantasy.Prompt{
			fantasy.NewUserMessage("Hello"),
		}

		result := model.enhanceWithTools(messages, []fantasy.Tool{})

		require.Equal(t, messages, result)
	})

	t.Run("should add system message with tools when no system message exists", func(t *testing.T) {
		t.Parallel()

		messages := fantasy.Prompt{
			fantasy.NewUserMessage("Hello"),
		}

		tools := []fantasy.Tool{
			fantasy.FunctionTool{
				Name:        "test_tool",
				Description: "A test tool",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{"type": "string"},
					},
				},
			},
		}

		result := model.enhanceWithTools(messages, tools)

		require.Len(t, result, 2) // System message added + original user message
		require.Equal(t, fantasy.MessageRoleSystem, result[0].Role)
	})

	t.Run("should enhance existing system message with tools", func(t *testing.T) {
		t.Parallel()

		messages := fantasy.Prompt{
			fantasy.NewSystemMessage("You are helpful."),
			fantasy.NewUserMessage("Hello"),
		}

		tools := []fantasy.Tool{
			fantasy.FunctionTool{
				Name:        "search",
				Description: "Search the web",
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		}

		result := model.enhanceWithTools(messages, tools)

		require.Len(t, result, 2)
		require.Equal(t, fantasy.MessageRoleSystem, result[0].Role)

		// Get system message text
		var systemText string
		for _, part := range result[0].Content {
			if tp, ok := part.(fantasy.TextPart); ok {
				systemText = tp.Text
				break
			}
		}

		require.Contains(t, systemText, "You are helpful.")
		require.Contains(t, systemText, "search")
	})
}

func TestProviderOptionsJSON(t *testing.T) {
	t.Parallel()

	t.Run("should marshal provider options with type wrapper", func(t *testing.T) {
		t.Parallel()

		temp := 0.7
		topP := 0.9
		topK := int64(40)
		seed := int64(42)

		opts := &ProviderOptions{
			Temperature:      &temp,
			TopP:             &topP,
			TopK:             &topK,
			Seed:             &seed,
			UseReasoningMode: true,
		}

		data, err := opts.MarshalJSON()
		require.NoError(t, err)
		require.NotEmpty(t, data)

		// Verify the JSON contains type wrapper
		jsonStr := string(data)
		require.Contains(t, jsonStr, `"type":"llama.options"`)
		require.Contains(t, jsonStr, `"data"`)
	})

	t.Run("should create provider options with values", func(t *testing.T) {
		t.Parallel()

		temp := 0.8
		opts := &ProviderOptions{
			Temperature:      &temp,
			UseReasoningMode: true,
		}

		require.NotNil(t, opts.Temperature)
		require.Equal(t, 0.8, *opts.Temperature)
		require.True(t, opts.UseReasoningMode)
	})

	t.Run("should handle nil values", func(t *testing.T) {
		t.Parallel()

		opts := &ProviderOptions{}

		require.Nil(t, opts.Temperature)
		require.Nil(t, opts.TopP)
		require.Nil(t, opts.TopK)
		require.Nil(t, opts.Seed)
		require.False(t, opts.UseReasoningMode)
	})
}

func TestConvertToTemplateMessages(t *testing.T) {
	t.Parallel()

	model := &languageModel{}

	t.Run("should pass through fantasy.Prompt unchanged", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			fantasy.NewSystemMessage("You are helpful."),
			fantasy.NewUserMessage("Hello"),
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hi there!"},
				},
			},
		}

		result := model.convertToTemplateMessages(prompt)

		require.Equal(t, prompt, result)
		require.Len(t, result, 3)
	})
}

// Integration tests - require actual model file
// Run with: go test -v -run TestIntegration -tags=integration

func TestIntegrationGenerate(t *testing.T) {
	modelPath := os.Getenv("LLAMA_MODEL_PATH")
	if modelPath == "" {
		modelPath = "/Users/yuda/.llama-cpp/models/llama-3.2-3b-instruct-q8_0.gguf"
	}

	// Skip if model doesn't exist
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Skipping integration test: model file not found at", modelPath)
	}

	t.Run("should generate response with llama model", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		prov, err := New(WithModelPath(modelPath))
		require.NoError(t, err)

		model, err := prov.LanguageModel(ctx, "")
		require.NoError(t, err)
		require.NotNil(t, model)

		// Simple generation test
		maxTokens := int64(100)
		response, err := model.Generate(ctx, fantasy.Call{
			Prompt: fantasy.Prompt{
				fantasy.NewUserMessage("Say hello in exactly 3 words."),
			},
			MaxOutputTokens: &maxTokens,
		})

		require.NoError(t, err)
		require.NotNil(t, response)
		require.NotEmpty(t, response.Content)

		// Check we got text content
		var hasText bool
		for _, content := range response.Content {
			if tc, ok := content.(fantasy.TextContent); ok {
				t.Logf("Response: %s", tc.Text)
				hasText = len(tc.Text) > 0
				break
			}
		}
		require.True(t, hasText, "Expected text content in response")

		// Verify usage is populated
		require.Greater(t, response.Usage.InputTokens, int64(0))
		require.Greater(t, response.Usage.OutputTokens, int64(0))

		// Cleanup
		if p, ok := prov.(*provider); ok {
			p.Cleanup()
		}
	})
}

func TestIntegrationStream(t *testing.T) {
	modelPath := os.Getenv("LLAMA_MODEL_PATH")
	if modelPath == "" {
		modelPath = "/Users/yuda/.llama-cpp/models/llama-3.2-3b-instruct-q8_0.gguf"
	}

	// Skip if model doesn't exist
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Skipping integration test: model file not found at", modelPath)
	}

	t.Run("should stream response with llama model", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		prov, err := New(WithModelPath(modelPath))
		require.NoError(t, err)

		model, err := prov.LanguageModel(ctx, "")
		require.NoError(t, err)

		maxTokens := int64(50)
		stream, err := model.Stream(ctx, fantasy.Call{
			Prompt: fantasy.Prompt{
				fantasy.NewUserMessage("Count from 1 to 5."),
			},
			MaxOutputTokens: &maxTokens,
		})

		require.NoError(t, err)
		require.NotNil(t, stream)

		var textDeltas []string
		var gotFinish bool

		for part := range stream {
			switch part.Type {
			case fantasy.StreamPartTypeTextDelta:
				textDeltas = append(textDeltas, part.Delta)
				t.Logf("Delta: %q", part.Delta)
			case fantasy.StreamPartTypeFinish:
				gotFinish = true
				t.Logf("Finish reason: %s", part.FinishReason)
			case fantasy.StreamPartTypeError:
				t.Fatalf("Stream error: %v", part.Error)
			}
		}

		require.NotEmpty(t, textDeltas, "Expected text deltas")
		require.True(t, gotFinish, "Expected finish event")

		// Cleanup
		if p, ok := prov.(*provider); ok {
			p.Cleanup()
		}
	})
}
