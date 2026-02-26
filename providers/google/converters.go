package google

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/getkawai/unillm"
	"google.golang.org/genai"
)

// toGooglePrompt converts a fantasy.Prompt to Google's Content format.
func toGooglePrompt(prompt fantasy.Prompt) (*genai.Content, []*genai.Content, []fantasy.CallWarning) { //nolint: unparam
	var systemInstructions *genai.Content
	var content []*genai.Content
	var warnings []fantasy.CallWarning

	finishedSystemBlock := false
	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			if finishedSystemBlock {
				// skip multiple system messages that are separated by user/assistant messages
				// TODO: see if we need to send error here?
				continue
			}
			finishedSystemBlock = true

			var systemMessages []string
			for _, part := range msg.Content {
				text, ok := fantasy.AsMessagePart[fantasy.TextPart](part)
				if !ok || text.Text == "" {
					continue
				}
				systemMessages = append(systemMessages, text.Text)
			}
			if len(systemMessages) > 0 {
				systemInstructions = &genai.Content{
					Parts: []*genai.Part{
						{
							Text: strings.Join(systemMessages, "\n"),
						},
					},
				}
			}
		case fantasy.MessageRoleUser:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case fantasy.ContentTypeText:
					text, ok := fantasy.AsMessagePart[fantasy.TextPart](part)
					if !ok || text.Text == "" {
						continue
					}
					parts = append(parts, &genai.Part{
						Text: text.Text,
					})
				case fantasy.ContentTypeFile:
					file, ok := fantasy.AsMessagePart[fantasy.FilePart](part)
					if !ok {
						continue
					}
					parts = append(parts, &genai.Part{
						InlineData: &genai.Blob{
							Data:     file.Data,
							MIMEType: file.MediaType,
						},
					})
				}
			}
			if len(parts) > 0 {
				content = append(content, &genai.Content{
					Role:  genai.RoleUser,
					Parts: parts,
				})
			}
		case fantasy.MessageRoleAssistant:
			var parts []*genai.Part
			var currentReasoningMetadata *ReasoningMetadata
			for _, part := range msg.Content {
				switch part.GetType() {
				case fantasy.ContentTypeReasoning:
					reasoning, ok := fantasy.AsMessagePart[fantasy.ReasoningPart](part)
					if !ok {
						continue
					}

					metadata, ok := reasoning.ProviderOptions[Name]
					if !ok {
						continue
					}
					reasoningMetadata, ok := metadata.(*ReasoningMetadata)
					if !ok {
						continue
					}
					currentReasoningMetadata = reasoningMetadata
				case fantasy.ContentTypeText:
					text, ok := fantasy.AsMessagePart[fantasy.TextPart](part)
					if !ok || text.Text == "" {
						continue
					}
					geminiPart := &genai.Part{
						Text: text.Text,
					}
					if currentReasoningMetadata != nil {
						geminiPart.ThoughtSignature = []byte(currentReasoningMetadata.Signature)
						currentReasoningMetadata = nil
					}
					parts = append(parts, geminiPart)
				case fantasy.ContentTypeToolCall:
					toolCall, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part)
					if !ok {
						continue
					}

					var result map[string]any
					err := json.Unmarshal([]byte(toolCall.Input), &result)
					if err != nil {
						continue
					}
					geminiPart := &genai.Part{
						FunctionCall: &genai.FunctionCall{
							ID:   toolCall.ToolCallID,
							Name: toolCall.ToolName,
							Args: result,
						},
					}
					if currentReasoningMetadata != nil {
						geminiPart.ThoughtSignature = []byte(currentReasoningMetadata.Signature)
						currentReasoningMetadata = nil
					}
					parts = append(parts, geminiPart)
				}
			}
			if len(parts) > 0 {
				content = append(content, &genai.Content{
					Role:  genai.RoleModel,
					Parts: parts,
				})
			}
		case fantasy.MessageRoleTool:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case fantasy.ContentTypeToolResult:
					result, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part)
					if !ok {
						continue
					}
					var toolCall fantasy.ToolCallPart
					for _, m := range prompt {
						if m.Role == fantasy.MessageRoleAssistant {
							for _, content := range m.Content {
								tc, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](content)
								if !ok {
									continue
								}
								if tc.ToolCallID == result.ToolCallID {
									toolCall = tc
									break
								}
							}
						}
					}
					switch result.Output.GetType() {
					case fantasy.ToolResultContentTypeText:
						content, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](result.Output)
						if !ok {
							continue
						}
						response := map[string]any{"result": content.Text}
						parts = append(parts, &genai.Part{
							FunctionResponse: &genai.FunctionResponse{
								ID:       result.ToolCallID,
								Response: response,
								Name:     toolCall.ToolName,
							},
						})

					case fantasy.ToolResultContentTypeError:
						content, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](result.Output)
						if !ok {
							continue
						}
						response := map[string]any{"result": content.Error.Error()}
						parts = append(parts, &genai.Part{
							FunctionResponse: &genai.FunctionResponse{
								ID:       result.ToolCallID,
								Response: response,
								Name:     toolCall.ToolName,
							},
						})
					}
				}
			}
			if len(parts) > 0 {
				content = append(content, &genai.Content{
					Role:  genai.RoleUser,
					Parts: parts,
				})
			}
		default:
			// Skip unsupported message roles instead of panicking
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeOther,
				Message: fmt.Sprintf("unsupported message role '%s' - skipping", msg.Role),
			})
		}
	}
	return systemInstructions, content, warnings
}

// toGoogleTools converts fantasy.Tool to Google's FunctionDeclaration format.
func toGoogleTools(tools []fantasy.Tool, toolChoice *fantasy.ToolChoice) (googleTools []*genai.FunctionDeclaration, googleToolChoice *genai.ToolConfig, warnings []fantasy.CallWarning) {
	for _, tool := range tools {
		if tool.GetType() == fantasy.ToolTypeFunction {
			ft, ok := tool.(fantasy.FunctionTool)
			if !ok {
				continue
			}

			required := []string{}
			var properties map[string]any
			if props, ok := ft.InputSchema["properties"]; ok {
				properties, _ = props.(map[string]any)
			}
			if req, ok := ft.InputSchema["required"]; ok {
				if reqArr, ok := req.([]string); ok {
					required = reqArr
				}
			}
			declaration := &genai.FunctionDeclaration{
				Name:        ft.Name,
				Description: ft.Description,
				Parameters: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: convertSchemaProperties(properties),
					Required:   required,
				},
			}
			googleTools = append(googleTools, declaration)
			continue
		}
		// TODO: handle provider tool calls
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedTool,
			Tool:    tool,
			Message: "tool is not supported",
		})
	}
	if toolChoice == nil {
		return googleTools, googleToolChoice, warnings
	}
	switch *toolChoice {
	case fantasy.ToolChoiceAuto:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAuto,
			},
		}
	case fantasy.ToolChoiceRequired:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
			},
		}
	case fantasy.ToolChoiceNone:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeNone,
			},
		}
	default:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{
					string(*toolChoice),
				},
			},
		}
	}
	return googleTools, googleToolChoice, warnings
}

// convertSchemaProperties converts a map of parameters to Google's Schema format.
func convertSchemaProperties(parameters map[string]any) map[string]*genai.Schema {
	properties := make(map[string]*genai.Schema)

	for name, param := range parameters {
		properties[name] = convertToSchema(param)
	}

	return properties
}

// convertToSchema converts a parameter to Google's Schema format.
func convertToSchema(param any) *genai.Schema {
	schema := &genai.Schema{Type: genai.TypeString}

	paramMap, ok := param.(map[string]any)
	if !ok {
		return schema
	}

	if desc, ok := paramMap["description"].(string); ok {
		schema.Description = desc
	}

	typeVal, hasType := paramMap["type"]
	if !hasType {
		return schema
	}

	typeStr, ok := typeVal.(string)
	if !ok {
		return schema
	}

	schema.Type = mapJSONTypeToGoogle(typeStr)

	switch typeStr {
	case "array":
		schema.Items = processArrayItems(paramMap)
	case "object":
		if props, ok := paramMap["properties"].(map[string]any); ok {
			schema.Properties = convertSchemaProperties(props)
		}
	}

	return schema
}

// processArrayItems processes array items in a schema.
func processArrayItems(paramMap map[string]any) *genai.Schema {
	items, ok := paramMap["items"].(map[string]any)
	if !ok {
		return nil
	}

	return convertToSchema(items)
}

// mapJSONTypeToGoogle maps JSON schema types to Google's Type enum.
func mapJSONTypeToGoogle(jsonType string) genai.Type {
	switch jsonType {
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	case "object":
		return genai.TypeObject
	default:
		return genai.TypeString // Default to string for unknown types
	}
}

// mapFinishReason maps Google's FinishReason to fantasy.FinishReason.
func mapFinishReason(reason genai.FinishReason) fantasy.FinishReason {
	switch reason {
	case genai.FinishReasonStop:
		return fantasy.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return fantasy.FinishReasonLength
	case genai.FinishReasonSafety,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety:
		return fantasy.FinishReasonContentFilter
	case genai.FinishReasonRecitation,
		genai.FinishReasonLanguage,
		genai.FinishReasonMalformedFunctionCall:
		return fantasy.FinishReasonError
	case genai.FinishReasonOther:
		return fantasy.FinishReasonOther
	default:
		return fantasy.FinishReasonUnknown
	}
}

// mapUsage maps Google's usage metadata to fantasy.Usage.
func mapUsage(usage *genai.GenerateContentResponseUsageMetadata) fantasy.Usage {
	return fantasy.Usage{
		InputTokens:         int64(usage.PromptTokenCount),
		OutputTokens:        int64(usage.CandidatesTokenCount),
		TotalTokens:         int64(usage.TotalTokenCount),
		ReasoningTokens:     int64(usage.ThoughtsTokenCount),
		CacheCreationTokens: 0,
		CacheReadTokens:     int64(usage.CachedContentTokenCount),
	}
}

// GetReasoningMetadata extracts reasoning metadata from provider options for google models.
func GetReasoningMetadata(providerOptions fantasy.ProviderOptions) *ReasoningMetadata {
	if googleOptions, ok := providerOptions[Name]; ok {
		if reasoning, ok := googleOptions.(*ReasoningMetadata); ok {
			return reasoning
		}
	}
	return nil
}
