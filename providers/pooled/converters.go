package pooled

import (
	"encoding/json"

	"github.com/kawai-network/veridium/pkg/cliproxy/sdk/cliproxy/executor"
	"github.com/getkawai/unillm"
)

// convertCallToRequest converts fantasy.Call to executor.Request.
func convertCallToRequest(call fantasy.Call) executor.Request {
	// Build a standard OpenAI-compatible payload
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	
	type ToolFunction struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	}
	
	type Tool struct {
		Type     string       `json:"type"`
		Function ToolFunction `json:"function"`
	}
	
	type Payload struct {
		Messages    []Message  `json:"messages"`
		MaxTokens   *int64     `json:"max_tokens,omitempty"`
		Temperature *float64   `json:"temperature,omitempty"`
		TopP        *float64   `json:"top_p,omitempty"`
		Tools       []Tool     `json:"tools,omitempty"`
		ToolChoice  any        `json:"tool_choice,omitempty"`
	}

	// Convert fantasy.Prompt ([]Message) to payload messages
	messages := make([]Message, 0, len(call.Prompt))
	for _, msg := range call.Prompt {
		// Extract text from message content (MessagePart)
		text := ""
		for _, part := range msg.Content {
			// Use type switch to handle different content types
			switch c := part.(type) {
			case fantasy.TextPart:
				text += c.Text
			}
		}
		
		messages = append(messages, Message{
			Role:    string(msg.Role),
			Content: text,
		})
	}

	// Convert tools to OpenAI format
	var tools []Tool
	if len(call.Tools) > 0 {
		tools = make([]Tool, 0, len(call.Tools))
		for _, tool := range call.Tools {
			if tool.GetType() == fantasy.ToolTypeFunction {
				if ft, ok := tool.(fantasy.FunctionTool); ok {
					tools = append(tools, Tool{
						Type: "function",
						Function: ToolFunction{
							Name:        ft.Name,
							Description: ft.Description,
							Parameters:  ft.InputSchema,
						},
					})
				}
			}
		}
	}

	payload := Payload{
		Messages:    messages,
		MaxTokens:   call.MaxOutputTokens,
		Temperature: call.Temperature,
		TopP:        call.TopP,
		Tools:       tools,
	}
	
	// Add tool choice if specified
	if call.ToolChoice != nil {
		switch *call.ToolChoice {
		case fantasy.ToolChoiceAuto:
			payload.ToolChoice = "auto"
		case fantasy.ToolChoiceNone:
			payload.ToolChoice = "none"
		default:
			// Specific tool choice
			payload.ToolChoice = map[string]any{
				"type": "function",
				"function": map[string]string{
					"name": string(*call.ToolChoice),
				},
			}
		}
	}

	// Serialize to JSON
	payloadJSON, _ := json.Marshal(payload)

	return executor.Request{
		Model:   "", // Model is handled by provider
		Payload: payloadJSON,
		Metadata: map[string]any{
			"fantasy_call": call, // Store original call for reference
		},
	}
}

// convertRequestToCall converts executor.Request to fantasy.Call.
func convertRequestToCall(req executor.Request) fantasy.Call {
	// Try to get original call from metadata
	if originalCall, ok := req.Metadata["fantasy_call"].(fantasy.Call); ok {
		return originalCall
	}

	// Otherwise parse from payload
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	
	type Payload struct {
		Messages    []Message `json:"messages"`
		MaxTokens   *int64    `json:"max_tokens,omitempty"`
		Temperature *float64  `json:"temperature,omitempty"`
		TopP        *float64  `json:"top_p,omitempty"`
	}

	var payload Payload
	if err := json.Unmarshal(req.Payload, &payload); err != nil {
		// Return empty call if we can't parse
		return fantasy.Call{}
	}

	// Convert messages to fantasy.Prompt
	prompt := make(fantasy.Prompt, 0, len(payload.Messages))
	for _, msg := range payload.Messages {
		prompt = append(prompt, fantasy.Message{
			Role: fantasy.MessageRole(msg.Role),
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: msg.Content},
			},
		})
	}

	return fantasy.Call{
		Prompt:          prompt,
		MaxOutputTokens: payload.MaxTokens,
		Temperature:     payload.Temperature,
		TopP:            payload.TopP,
	}
}

// convertResponseToFantasy converts executor.Response to fantasy.Response.
func convertResponseToFantasy(resp executor.Response) (*fantasy.Response, error) {
	content := make(fantasy.ResponseContent, 0)
	
	// Extract text content
	if contentStr, ok := resp.Metadata["content"].(string); ok && contentStr != "" {
		content = append(content, fantasy.TextContent{Text: contentStr})
	}
	
	// Extract tool calls
	if toolCallsRaw, ok := resp.Metadata["tool_calls"]; ok {
		// First, marshal and unmarshal to normalize the type
		toolCallsJSON, err := json.Marshal(toolCallsRaw)
		if err == nil {
			var toolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments any    `json:"arguments"` // Can be string or object
				} `json:"function"`
			}
			
			if err := json.Unmarshal(toolCallsJSON, &toolCalls); err == nil {
				for _, tc := range toolCalls {
					if tc.ID != "" && tc.Function.Name != "" {
						// Convert arguments to string if needed
						argsStr := ""
						switch args := tc.Function.Arguments.(type) {
						case string:
							argsStr = args
						default:
							// Marshal back to JSON string
							if argsJSON, err := json.Marshal(args); err == nil {
								argsStr = string(argsJSON)
							}
						}
						
						content = append(content, fantasy.ToolCallContent{
							ProviderExecuted: false,
							ToolCallID:       tc.ID,
							ToolName:         tc.Function.Name,
							Input:            argsStr,
						})
					}
				}
			}
		}
	}
	
	// If no content at all, return empty response
	if len(content) == 0 {
		content = fantasy.ResponseContent{
			fantasy.TextContent{Text: ""},
		}
	}

	finishReasonStr := ""
	if fr, ok := resp.Metadata["finish_reason"].(string); ok {
		finishReasonStr = fr
	}

	return &fantasy.Response{
		Content:      content,
		FinishReason: fantasy.FinishReason(finishReasonStr),
		Usage: fantasy.Usage{
			InputTokens:  getIntFromMetadata(resp.Metadata, "prompt_tokens"),
			OutputTokens: getIntFromMetadata(resp.Metadata, "completion_tokens"),
			TotalTokens:  getIntFromMetadata(resp.Metadata, "total_tokens"),
		},
	}, nil
}

// convertFantasyToResponse converts fantasy.Response to executor.Response.
func convertFantasyToResponse(resp *fantasy.Response) executor.Response {
	// Extract text from ResponseContent
	// Handle both text content and reasoning content
	contentText := ""
	var toolCalls []map[string]any
	
	for _, content := range resp.Content {
		switch c := content.(type) {
		case fantasy.TextContent:
			contentText += c.Text
		case fantasy.ReasoningContent:
			// For reasoning content, include it as well
			contentText += c.Text
		case fantasy.ToolCallContent:
			// Extract tool call
			toolCalls = append(toolCalls, map[string]any{
				"id":   c.ToolCallID,
				"type": "function",
				"function": map[string]any{
					"name":      c.ToolName,
					"arguments": c.Input,
				},
			})
		}
	}
	
	// Fallback: use the Text() method if nothing extracted
	if contentText == "" && len(toolCalls) == 0 {
		contentText = resp.Content.Text()
	}

	metadata := map[string]any{
		"content":           contentText,
		"finish_reason":     string(resp.FinishReason),
		"prompt_tokens":     resp.Usage.InputTokens,
		"completion_tokens": resp.Usage.OutputTokens,
		"total_tokens":      resp.Usage.TotalTokens,
	}
	
	// Add tool calls if present
	if len(toolCalls) > 0 {
		metadata["tool_calls"] = toolCalls
	}

	return executor.Response{
		Metadata: metadata,
	}
}

// convertStreamToFantasy converts executor stream to fantasy.StreamResponse.
func convertStreamToFantasy(chunks <-chan executor.StreamChunk) fantasy.StreamResponse {
	return func(yield func(fantasy.StreamPart) bool) {
		for chunk := range chunks {
			part := convertChunkToFantasyPart(chunk)
			if !yield(part) {
				return
			}
		}
	}
}

// convertChunkToFantasyPart converts executor.StreamChunk to fantasy.StreamPart.
func convertChunkToFantasyPart(chunk executor.StreamChunk) fantasy.StreamPart {
	if chunk.Err != nil {
		return fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeError,
			Error: chunk.Err,
		}
	}

	// Parse payload as delta text
	if len(chunk.Payload) > 0 {
		return fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeTextDelta,
			Delta: string(chunk.Payload),
		}
	}

	return fantasy.StreamPart{
		Type: fantasy.StreamPartTypeTextDelta,
	}
}

// convertFantasyPartToChunk converts fantasy.StreamPart to executor.StreamChunk.
func convertFantasyPartToChunk(part fantasy.StreamPart) executor.StreamChunk {
	if part.Error != nil {
		return executor.StreamChunk{
			Err: part.Error,
		}
	}

	// Convert delta to payload
	return executor.StreamChunk{
		Payload: []byte(part.Delta),
	}
}

// getIntFromMetadata safely extracts an int from metadata.
func getIntFromMetadata(metadata map[string]any, key string) int64 {
	if val, ok := metadata[key].(int); ok {
		return int64(val)
	}
	if val, ok := metadata[key].(int64); ok {
		return val
	}
	if val, ok := metadata[key].(float64); ok {
		return int64(val)
	}
	return 0
}

