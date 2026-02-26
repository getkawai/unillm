package llama

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/google/uuid"
	"github.com/getkawai/unillm"
	"github.com/getkawai/llamalib"
	"github.com/getkawai/llamalib/llama"
	"github.com/getkawai/llamalib/message"
	"github.com/getkawai/llamalib/template"
	"github.com/getkawai/tools"
)

// Default batch size for token processing
const defaultBatchSize = 2048

type languageModel struct {
	provider     string
	modelID      string
	service      *llamalib.Service
	toolRegistry *tools.ToolRegistry
}

func newLanguageModel(modelID string, provider string, service *llamalib.Service, toolRegistry *tools.ToolRegistry) *languageModel {
	return &languageModel{
		modelID:      modelID,
		provider:     provider,
		service:      service,
		toolRegistry: toolRegistry,
	}
}

// Model implements fantasy.LanguageModel.
func (l *languageModel) Model() string {
	return l.modelID
}

// Provider implements fantasy.LanguageModel.
func (l *languageModel) Provider() string {
	return l.provider
}

// Generate implements fantasy.LanguageModel.
func (l *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	// Convert fantasy.Call to prompt string using chat template
	prompt, err := l.preparePrompt(call)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare prompt: %w", err)
	}

	// Get max tokens from call or use default
	maxTokens := int32(32768)
	if call.MaxOutputTokens != nil {
		maxTokens = int32(*call.MaxOutputTokens)
	}

	// Generate response
	response, promptTokens, completionTokens, err := l.generateWithTokenCounts(ctx, prompt, maxTokens)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	// Parse tool calls from response
	toolCalls := tools.ParseToolCalls(response)

	// Build response content
	content := make([]fantasy.Content, 0)

	// Clean tool call tags and extract reasoning from response text
	reasoning := l.extractReasoning(&response)
	cleanedResponse := l.cleanToolCallTags(response)

	if reasoning != "" {
		content = append(content, fantasy.ReasoningContent{Text: reasoning})
	}

	if cleanedResponse != "" {
		content = append(content, fantasy.TextContent{Text: cleanedResponse})
	}

	// Add tool calls as content
	for _, tc := range toolCalls {
		content = append(content, fantasy.ToolCallContent{
			ToolCallID:       tc.ID,
			ToolName:         tc.Name,
			Input:            tc.Input,
			ProviderExecuted: false,
		})
	}

	finishReason := fantasy.FinishReasonStop
	if len(toolCalls) > 0 {
		finishReason = fantasy.FinishReasonToolCalls
	}

	return &fantasy.Response{
		Content:      content,
		FinishReason: finishReason,
		Usage: fantasy.Usage{
			InputTokens:  int64(promptTokens),
			OutputTokens: int64(completionTokens),
			TotalTokens:  int64(promptTokens + completionTokens),
		},
	}, nil
}

// Stream implements fantasy.LanguageModel.
func (l *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	prompt, err := l.preparePrompt(call)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare prompt: %w", err)
	}

	maxTokens := int32(32768)
	if call.MaxOutputTokens != nil {
		maxTokens = int32(*call.MaxOutputTokens)
	}

	return func(yield func(fantasy.StreamPart) bool) {
		promptTokens := len(prompt) / 4 // Approximate

		processor := newStreamProcessor(yield)

		// Generate with streaming
		completionTokens, err := l.generateStreaming(ctx, prompt, maxTokens, func(token string, done bool) {
			if done {
				return
			}
			processor.Process(token)
		})

		if err != nil {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: err,
			})
			return
		}

		processor.Flush()

		finishReason := fantasy.FinishReasonStop
		if processor.HasToolCalls() {
			finishReason = fantasy.FinishReasonToolCalls
		}

		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: finishReason,
			Usage: fantasy.Usage{
				InputTokens:  int64(promptTokens),
				OutputTokens: int64(completionTokens),
				TotalTokens:  int64(promptTokens + completionTokens),
			},
		})
	}, nil
}

// GenerateObject implements fantasy.LanguageModel.
func (l *languageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	// For now, use text-based object generation
	// TODO: Implement proper JSON mode if model supports it
	return nil, fmt.Errorf("GenerateObject not yet implemented for llama provider")
}

// StreamObject implements fantasy.LanguageModel.
func (l *languageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, fmt.Errorf("StreamObject not yet implemented for llama provider")
}

// preparePrompt converts fantasy.Call to a formatted prompt string
func (l *languageModel) preparePrompt(call fantasy.Call) (string, error) {
	// Start with original fantasy messages
	messages := call.Prompt

	// Enhance with tools if available and ToolChoice is not "none"
	// ToolChoiceNone disables all tools (useful for title/summary generation)
	shouldEnhanceTools := l.toolRegistry != nil && len(call.Tools) > 0
	if call.ToolChoice != nil && *call.ToolChoice == fantasy.ToolChoiceNone {
		shouldEnhanceTools = false
	}
	if shouldEnhanceTools {
		messages = l.enhanceWithTools(messages, call.Tools)
	}

	// Convert fantasy.Prompt to template-compatible messages
	templateMessages := l.convertToTemplateMessages(messages)

	// Get chat template
	chatTemplate := l.getChatTemplate()

	// Apply chat template
	prompt, err := template.Apply(chatTemplate, templateMessages, true)
	if err != nil {
		log.Printf("⚠️  Failed to apply model template: %v. Falling back to universal ChatML template.", err)
		// Fallback to universal ChatML template which is gonja-compatible
		prompt, err = template.Apply(llamalib.ChatMLToolTemplate, templateMessages, true)
		if err != nil {
			return "", fmt.Errorf("failed to apply both model and fallback templates: %w", err)
		}
	}

	return prompt, nil
}

// convertToTemplateMessages converts fantasy.Prompt to []message.Message for template.Apply
func (l *languageModel) convertToTemplateMessages(prompt fantasy.Prompt) []message.Message {
	result := make([]message.Message, 0, len(prompt))

	// Build a map of tool call IDs to tool names for resolving tool responses
	toolCallIDToName := make(map[string]string)

	for _, msg := range prompt {
		// Check if message contains tool calls (assistant with tool calls)
		var toolCalls []message.ToolCall
		var textContent string
		var toolResultName string
		var toolResultContent string

		for _, part := range msg.Content {
			switch p := part.(type) {
			case fantasy.TextPart:
				textContent += p.Text
			case fantasy.ToolCallPart:
				// Convert fantasy.ToolCallPart to message.ToolCall
				// Parse the Input JSON string to map[string]string
				args := map[string]string{
					"json": p.Input,
				}
				toolCalls = append(toolCalls, message.ToolCall{
					Type: "function",
					Function: message.ToolFunction{
						Name:      p.ToolName,
						Arguments: args,
					},
				})
				// Store mapping for later tool response resolution
				toolCallIDToName[p.ToolCallID] = p.ToolName
			case fantasy.ToolResultPart:
				// Extract tool result information
				if output, ok := p.Output.(fantasy.ToolResultOutputContentText); ok {
					toolResultContent = output.Text
				}
				// Look up the tool name from the tool call ID
				if name, ok := toolCallIDToName[p.ToolCallID]; ok {
					toolResultName = name
				}
			}
		}

		// Create appropriate message type based on content
		if len(toolCalls) > 0 {
			// Assistant message with tool calls
			// If there's also text content, emit it as a separate Chat message first
			if textContent != "" {
				result = append(result, message.Chat{
					Role:    string(msg.Role),
					Content: textContent,
				})
			}
			result = append(result, message.Tool{
				Role:      string(msg.Role),
				ToolCalls: toolCalls,
			})
		} else if msg.Role == fantasy.MessageRoleTool {
			// Tool response message
			result = append(result, message.ToolResponse{
				Role:    string(msg.Role),
				Name:    toolResultName,
				Content: toolResultContent,
			})
		} else {
			// Regular text message (system, user, assistant)
			result = append(result, message.Chat{
				Role:    string(msg.Role),
				Content: textContent,
			})
		}
	}

	return result
}

// enhanceWithTools adds tool definitions to the prompt
func (l *languageModel) enhanceWithTools(messages fantasy.Prompt, callTools []fantasy.Tool) fantasy.Prompt {
	if len(callTools) == 0 {
		return messages
	}

	// Build tool definitions JSON
	var toolDefs []map[string]any
	for _, tool := range callTools {
		if ft, ok := tool.(fantasy.FunctionTool); ok {
			toolDefs = append(toolDefs, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        ft.Name,
					"description": ft.Description,
					"parameters":  ft.InputSchema,
				},
			})
		}
	}

	if len(toolDefs) == 0 {
		return messages
	}

	// Format tools for prompt
	toolsJSON := tools.FormatToolsJSON(toolDefs)

	// Make a copy to avoid modifying original
	result := make(fantasy.Prompt, len(messages))
	copy(result, messages)

	// Find or create system message
	if len(result) > 0 && result[0].Role == fantasy.MessageRoleSystem {
		// Get existing system text
		var existingText string
		for _, part := range result[0].Content {
			if tp, ok := part.(fantasy.TextPart); ok {
				existingText = tp.Text
				break
			}
		}
		// Enhance existing system message
		enhancedContent := tools.BuildSystemPrompt(existingText, toolsJSON)
		result[0] = fantasy.NewSystemMessage(enhancedContent)
	} else {
		// Prepend new system message with tools
		systemContent := tools.BuildSystemPrompt("You are a helpful AI assistant.", toolsJSON)
		result = append(fantasy.Prompt{fantasy.NewSystemMessage(systemContent)}, result...)
	}

	return result
}

// getChatTemplate returns the appropriate chat template for the current model
func (l *languageModel) getChatTemplate() string {
	modelPath := l.service.GetLoadedChatModel()
	modelPathLower := strings.ToLower(modelPath)

	// 1. Check for specific known models that need custom templates
	if strings.Contains(modelPathLower, "llama-3.2") || strings.Contains(modelPathLower, "llama_3.2") || strings.Contains(modelPathLower, "llama3.2") {
		log.Printf("🔧 Using custom Llama 3.2 tool template")
		return llamalib.Llama32ToolTemplate
	}

	// 2. Models using ChatML format (Nemotron, Qwen, OpenThinker)
	if strings.Contains(modelPathLower, "nemotron") || strings.Contains(modelPathLower, "openthinker") || strings.Contains(modelPathLower, "qwen") {
		log.Printf("🔧 Using universal ChatML tool template for %s", modelPathLower)
		return llamalib.ChatMLToolTemplate
	}

	// 3. Fallback: Use embedded template from model metadata
	tmpl := llama.ModelChatTemplate(l.service.GetChatModel(), "")
	if tmpl == "" {
		log.Printf("⚠️  Model has no embedded chat template, using universal ChatML fallback")
		return llamalib.ChatMLToolTemplate
	}

	return tmpl
}

// generateWithTokenCounts generates response and returns token counts
func (l *languageModel) generateWithTokenCounts(ctx context.Context, prompt string, maxTokens int32) (string, int, int, error) {
	// Tokenize prompt
	vocab := l.service.GetChatVocab()
	tokens := llama.Tokenize(vocab, prompt, true, true)
	if len(tokens) == 0 {
		return "", 0, 0, fmt.Errorf("failed to tokenize prompt")
	}

	promptTokens := len(tokens)

	// Generate response
	response, err := l.service.Generate(prompt, maxTokens)
	if err != nil {
		return "", promptTokens, 0, err
	}

	// Estimate completion tokens
	completionTokens := len(response) / 4

	return response, promptTokens, completionTokens, nil
}

// generateStreaming performs token-by-token generation with callback
// Uses WithChatLock to ensure thread-safe access to chat model resources
func (l *languageModel) generateStreaming(ctx context.Context, prompt string, maxTokens int32, callback func(token string, done bool)) (int, error) {
	var completionTokens int
	var genErr error

	l.service.WithChatLock(func() {
		chatModel, chatContext, chatVocab, chatSampler := l.service.GetChatResourcesUnsafe()

		if chatModel == 0 || chatContext == 0 {
			genErr = fmt.Errorf("chat model not loaded")
			return
		}

		// Tokenize prompt
		tokens := llama.Tokenize(chatVocab, prompt, true, true)
		if len(tokens) == 0 {
			genErr = fmt.Errorf("failed to tokenize prompt")
			return
		}

		// Reset sampler state
		llama.SamplerReset(chatSampler)

		// Process prompt tokens in batches
		log.Printf("🔢 Processing %d prompt tokens in batches of %d", len(tokens), defaultBatchSize)

		for i := 0; i < len(tokens); i += defaultBatchSize {
			end := i + defaultBatchSize
			if end > len(tokens) {
				end = len(tokens)
			}

			batchTokens := tokens[i:end]
			batch := llama.BatchGetOne(batchTokens)

			errCode, err := llama.Decode(chatContext, batch)
			if err != nil || errCode != 0 {
				genErr = fmt.Errorf("failed to decode prompt batch %d-%d: %w", i, end, err)
				return
			}
		}

		// Generate tokens and stream
		buf := make([]byte, 256)
		for nGenerated := int32(0); nGenerated < maxTokens; nGenerated++ {
			// Check context cancellation
			select {
			case <-ctx.Done():
				callback("", true)
				genErr = ctx.Err()
				return
			default:
			}

			// Sample next token
			token := llama.SamplerSample(chatSampler, chatContext, -1)

			// Check for end of generation
			if llama.VocabIsEOG(chatVocab, token) {
				callback("", true)
				break
			}

			// Convert token to text
			length := llama.TokenToPiece(chatVocab, token, buf, 0, false)
			content := string(buf[:length])

			completionTokens++
			callback(content, false)

			// Accept token and prepare for next generation
			llama.SamplerAccept(chatSampler, token)

			// Decode the new token
			nextBatch := llama.BatchGetOne([]llama.Token{token})
			errCode, err := llama.Decode(chatContext, nextBatch)
			if err != nil || errCode != 0 {
				genErr = fmt.Errorf("failed to decode token: %w", err)
				return
			}
		}
	})

	return completionTokens, genErr
}

// cleanToolCallTags removes tool call and reasoning XML tags from response
func (l *languageModel) cleanToolCallTags(response string) string {
	// Use robust tool call stripping that handles nested JSON and edge cases
	cleanResponse := tools.StripToolCallTags(response)

	// Also strip reasoning tags using robust extraction
	cleanResponse = stripReasoningTags(cleanResponse)

	return strings.TrimSpace(cleanResponse)
}

// stripReasoningTags removes <think> and <thought> tags using brace-aware extraction
func stripReasoningTags(text string) string {
	result := text

	// Process each type of reasoning tag
	for _, tagPair := range [][]string{{"<think>", "</think>"}, {"<thought>", "</thought>"}} {
		openTag, closeTag := tagPair[0], tagPair[1]
		var output strings.Builder
		remaining := result

		for {
			openIdx := strings.Index(remaining, openTag)
			if openIdx == -1 {
				output.WriteString(remaining)
				break
			}

			// Add text before the tag
			output.WriteString(remaining[:openIdx])

			// Find the matching close tag
			afterOpen := remaining[openIdx+len(openTag):]
			closeIdx := strings.Index(afterOpen, closeTag)
			if closeIdx == -1 {
				// No closing tag found, keep the rest as-is
				output.WriteString(remaining[openIdx:])
				break
			}

			// Skip the content and closing tag
			remaining = afterOpen[closeIdx+len(closeTag):]
		}

		result = output.String()
	}

	return result
}

// extractReasoning extracts text from reasoning tags and removes them from input
func (l *languageModel) extractReasoning(response *string) string {
	var reasoning strings.Builder
	tags := [][]string{
		{"<think>", "</think>"},
		{"<thought>", "</thought>"},
	}

	for _, tagPair := range tags {
		for {
			start := strings.Index(*response, tagPair[0])
			end := strings.Index(*response, tagPair[1])
			if start != -1 && end != -1 && start < end {
				reasoning.WriteString((*response)[start+len(tagPair[0]) : end])
				*response = (*response)[:start] + (*response)[end+len(tagPair[1]):]
			} else {
				break
			}
		}
	}
	return strings.TrimSpace(reasoning.String())
}

// generateToolCallID generates a unique ID for a tool call
func generateToolCallID() string {
	return "call_" + uuid.NewString()[:8]
}
