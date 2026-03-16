package vercel

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/getkawai/unillm"
	"github.com/getkawai/unillm/providers/google"
	openaipkg "github.com/getkawai/unillm/providers/openai"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

const reasoningStartedCtx = "reasoning_started"

type currentReasoningState struct {
	metadata       *openaipkg.ResponsesReasoningMetadata
	googleMetadata *google.ReasoningMetadata
	googleText     string
}

func languagePrepareModelCall(_ unillm.LanguageModel, params *openaisdk.ChatCompletionNewParams, call unillm.Call) ([]unillm.CallWarning, error) {
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, &unillm.Error{Title: "invalid argument", Message: "vercel provider options should be *vercel.ProviderOptions"}
		}
	}

	extraFields := make(map[string]any)

	// Handle reasoning options
	if providerOptions.Reasoning != nil {
		data, err := structToMapJSON(providerOptions.Reasoning)
		if err != nil {
			return nil, err
		}
		extraFields["reasoning"] = data
	}

	// Handle provider options for gateway routing
	if providerOptions.ProviderOptions != nil {
		data, err := structToMapJSON(providerOptions.ProviderOptions)
		if err != nil {
			return nil, err
		}
		extraFields["providerOptions"] = map[string]any{
			"gateway": data,
		}
	}

	// Handle BYOK (Bring Your Own Key)
	if providerOptions.BYOK != nil {
		data, err := structToMapJSON(providerOptions.BYOK)
		if err != nil {
			return nil, err
		}
		if gatewayOpts, ok := extraFields["providerOptions"].(map[string]any); ok {
			gatewayOpts["byok"] = data
		} else {
			extraFields["providerOptions"] = map[string]any{
				"gateway": map[string]any{
					"byok": data,
				},
			}
		}
	}

	// Handle standard OpenAI options
	if providerOptions.LogitBias != nil {
		params.LogitBias = providerOptions.LogitBias
	}
	if providerOptions.LogProbs != nil {
		params.Logprobs = param.NewOpt(*providerOptions.LogProbs)
	}
	if providerOptions.TopLogProbs != nil {
		params.TopLogprobs = param.NewOpt(*providerOptions.TopLogProbs)
	}
	if providerOptions.User != nil {
		params.User = param.NewOpt(*providerOptions.User)
	}
	if providerOptions.ParallelToolCalls != nil {
		params.ParallelToolCalls = param.NewOpt(*providerOptions.ParallelToolCalls)
	}

	// Handle model fallbacks - direct models field
	if providerOptions.ProviderOptions != nil && len(providerOptions.ProviderOptions.Models) > 0 {
		extraFields["models"] = providerOptions.ProviderOptions.Models
	}

	maps.Copy(extraFields, providerOptions.ExtraBody)
	params.SetExtraFields(extraFields)
	return nil, nil
}

func languageModelExtraContent(choice openaisdk.ChatCompletionChoice) []unillm.Content {
	content := make([]unillm.Content, 0)
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Message.RawJSON()), &reasoningData)
	if err != nil {
		return content
	}

	responsesReasoningBlocks := make([]openaipkg.ResponsesReasoningMetadata, 0)
	googleReasoningBlocks := make([]struct {
		text     string
		metadata *google.ReasoningMetadata
	}, 0)
	otherReasoning := make([]string, 0)

	for _, detail := range reasoningData.ReasoningDetails {
		if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
			var thinkingBlock openaipkg.ResponsesReasoningMetadata
			if len(responsesReasoningBlocks)-1 >= detail.Index {
				thinkingBlock = responsesReasoningBlocks[detail.Index]
			} else {
				thinkingBlock = openaipkg.ResponsesReasoningMetadata{}
				responsesReasoningBlocks = append(responsesReasoningBlocks, thinkingBlock)
			}

			switch detail.Type {
			case "reasoning.summary":
				thinkingBlock.Summary = append(thinkingBlock.Summary, detail.Summary)
			case "reasoning.encrypted":
				thinkingBlock.EncryptedContent = &detail.Data
			}
			if detail.ID != "" {
				thinkingBlock.ItemID = detail.ID
			}
			responsesReasoningBlocks[detail.Index] = thinkingBlock
			continue
		}

		if strings.HasPrefix(detail.Format, "google-gemini") {
			var thinkingBlock struct {
				text     string
				metadata *google.ReasoningMetadata
			}
			if len(googleReasoningBlocks)-1 >= detail.Index {
				thinkingBlock = googleReasoningBlocks[detail.Index]
			} else {
				thinkingBlock = struct {
					text     string
					metadata *google.ReasoningMetadata
				}{metadata: &google.ReasoningMetadata{}}
				googleReasoningBlocks = append(googleReasoningBlocks, thinkingBlock)
			}

			switch detail.Type {
			case "reasoning.text":
				thinkingBlock.text = detail.Text
			case "reasoning.encrypted":
				thinkingBlock.metadata.Signature = detail.Data
				thinkingBlock.metadata.ToolID = detail.ID
			}
			googleReasoningBlocks[detail.Index] = thinkingBlock
			continue
		}

		otherReasoning = append(otherReasoning, detail.Text)
	}

	// Fallback to simple reasoning field if no details
	if reasoningData.Reasoning != "" && len(reasoningData.ReasoningDetails) == 0 {
		otherReasoning = append(otherReasoning, reasoningData.Reasoning)
	}

	for _, block := range responsesReasoningBlocks {
		if len(block.Summary) == 0 {
			block.Summary = []string{""}
		}
		content = append(content, unillm.ReasoningContent{
			Text: strings.Join(block.Summary, "\n"),
			ProviderMetadata: unillm.ProviderMetadata{
				openaipkg.Name: &block,
			},
		})
	}

	for _, block := range googleReasoningBlocks {
		content = append(content, unillm.ReasoningContent{
			Text: block.text,
			ProviderMetadata: unillm.ProviderMetadata{
				google.Name: block.metadata,
			},
		})
	}

	for _, reasoning := range otherReasoning {
		if reasoning != "" {
			content = append(content, unillm.ReasoningContent{
				Text: reasoning,
			})
		}
	}

	return content
}

func extractReasoningContext(ctx map[string]any) *currentReasoningState {
	reasoningStarted, ok := ctx[reasoningStartedCtx]
	if !ok {
		return nil
	}
	state, ok := reasoningStarted.(*currentReasoningState)
	if !ok {
		return nil
	}
	return state
}

func languageModelStreamExtra(chunk openaisdk.ChatCompletionChunk, yield func(unillm.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
	if len(chunk.Choices) == 0 {
		return ctx, true
	}

	currentState := extractReasoningContext(ctx)

	inx := 0
	choice := chunk.Choices[inx]
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Delta.RawJSON()), &reasoningData)
	if err != nil {
		yield(unillm.StreamPart{
			Type:  unillm.StreamPartTypeError,
			Error: &unillm.Error{Title: "stream error", Message: "error unmarshalling delta", Cause: err},
		})
		return ctx, false
	}

	// Reasoning Start
	if currentState == nil {
		if len(reasoningData.ReasoningDetails) == 0 && reasoningData.Reasoning == "" {
			return ctx, true
		}

		var metadata unillm.ProviderMetadata
		currentState = &currentReasoningState{}

		if len(reasoningData.ReasoningDetails) > 0 {
			detail := reasoningData.ReasoningDetails[0]

			if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
				currentState.metadata = &openaipkg.ResponsesReasoningMetadata{
					Summary: []string{detail.Summary},
				}
				metadata = unillm.ProviderMetadata{
					openaipkg.Name: currentState.metadata,
				}
				if detail.Data != "" {
					shouldContinue := yield(unillm.StreamPart{
						Type:             unillm.StreamPartTypeReasoningStart,
						ID:               fmt.Sprintf("%d", inx),
						Delta:            detail.Summary,
						ProviderMetadata: metadata,
					})
					if !shouldContinue {
						return ctx, false
					}
					return ctx, yield(unillm.StreamPart{
						Type: unillm.StreamPartTypeReasoningEnd,
						ID:   fmt.Sprintf("%d", inx),
						ProviderMetadata: unillm.ProviderMetadata{
							openaipkg.Name: &openaipkg.ResponsesReasoningMetadata{
								Summary:          []string{detail.Summary},
								EncryptedContent: &detail.Data,
								ItemID:           detail.ID,
							},
						},
					})
				}
			}

			if strings.HasPrefix(detail.Format, "google-gemini") {
				if detail.Type == "reasoning.encrypted" {
					ctx[reasoningStartedCtx] = nil
					if !yield(unillm.StreamPart{
						Type: unillm.StreamPartTypeReasoningStart,
						ID:   fmt.Sprintf("%d", inx),
					}) {
						return ctx, false
					}
					return ctx, yield(unillm.StreamPart{
						Type: unillm.StreamPartTypeReasoningEnd,
						ID:   fmt.Sprintf("%d", inx),
						ProviderMetadata: unillm.ProviderMetadata{
							google.Name: &google.ReasoningMetadata{
								Signature: detail.Data,
								ToolID:    detail.ID,
							},
						},
					})
				}
				currentState.googleMetadata = &google.ReasoningMetadata{}
				currentState.googleText = detail.Text
				metadata = unillm.ProviderMetadata{
					google.Name: currentState.googleMetadata,
				}
			}
		}

		ctx[reasoningStartedCtx] = currentState
		delta := reasoningData.Reasoning
		if len(reasoningData.ReasoningDetails) > 0 {
			delta = reasoningData.ReasoningDetails[0].Summary
			if strings.HasPrefix(reasoningData.ReasoningDetails[0].Format, "google-gemini") {
				delta = reasoningData.ReasoningDetails[0].Text
			}
		}
		return ctx, yield(unillm.StreamPart{
			Type:             unillm.StreamPartTypeReasoningStart,
			ID:               fmt.Sprintf("%d", inx),
			Delta:            delta,
			ProviderMetadata: metadata,
		})
	}

	if len(reasoningData.ReasoningDetails) == 0 && reasoningData.Reasoning == "" {
		if choice.Delta.Content != "" || len(choice.Delta.ToolCalls) > 0 {
			ctx[reasoningStartedCtx] = nil
			return ctx, yield(unillm.StreamPart{
				Type: unillm.StreamPartTypeReasoningEnd,
				ID:   fmt.Sprintf("%d", inx),
			})
		}
		return ctx, true
	}

	if len(reasoningData.ReasoningDetails) > 0 {
		detail := reasoningData.ReasoningDetails[0]

		if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
			if detail.Data != "" {
				currentState.metadata.EncryptedContent = &detail.Data
				currentState.metadata.ItemID = detail.ID
				ctx[reasoningStartedCtx] = nil
				return ctx, yield(unillm.StreamPart{
					Type: unillm.StreamPartTypeReasoningEnd,
					ID:   fmt.Sprintf("%d", inx),
					ProviderMetadata: unillm.ProviderMetadata{
						openaipkg.Name: currentState.metadata,
					},
				})
			}
			var textDelta string
			if len(currentState.metadata.Summary)-1 >= detail.Index {
				currentState.metadata.Summary[detail.Index] += detail.Summary
				textDelta = detail.Summary
			} else {
				currentState.metadata.Summary = append(currentState.metadata.Summary, detail.Summary)
				textDelta = "\n" + detail.Summary
			}
			ctx[reasoningStartedCtx] = currentState
			return ctx, yield(unillm.StreamPart{
				Type:  unillm.StreamPartTypeReasoningDelta,
				ID:    fmt.Sprintf("%d", inx),
				Delta: textDelta,
				ProviderMetadata: unillm.ProviderMetadata{
					openaipkg.Name: currentState.metadata,
				},
			})
		}

		if strings.HasPrefix(detail.Format, "google-gemini") {
			if detail.Type == "reasoning.text" {
				currentState.googleText += detail.Text
				ctx[reasoningStartedCtx] = currentState
				return ctx, yield(unillm.StreamPart{
					Type:  unillm.StreamPartTypeReasoningDelta,
					ID:    fmt.Sprintf("%d", inx),
					Delta: detail.Text,
				})
			}
			if detail.Type == "reasoning.encrypted" {
				currentState.googleMetadata.Signature = detail.Data
				currentState.googleMetadata.ToolID = detail.ID
				metadata := unillm.ProviderMetadata{
					google.Name: currentState.googleMetadata,
				}
				ctx[reasoningStartedCtx] = nil
				return ctx, yield(unillm.StreamPart{
					Type:             unillm.StreamPartTypeReasoningEnd,
					ID:               fmt.Sprintf("%d", inx),
					ProviderMetadata: metadata,
				})
			}
		}

		return ctx, yield(unillm.StreamPart{
			Type:  unillm.StreamPartTypeReasoningDelta,
			ID:    fmt.Sprintf("%d", inx),
			Delta: detail.Text,
		})
	}

	if reasoningData.Reasoning != "" {
		return ctx, yield(unillm.StreamPart{
			Type:  unillm.StreamPartTypeReasoningDelta,
			ID:    fmt.Sprintf("%d", inx),
			Delta: reasoningData.Reasoning,
		})
	}

	return ctx, true
}

func languageModelUsage(response openaisdk.ChatCompletion) (unillm.Usage, unillm.ProviderOptionsData) {
	if len(response.Choices) == 0 {
		return unillm.Usage{}, nil
	}

	usage := response.Usage
	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails

	var provider string
	if p, ok := response.JSON.ExtraFields["provider"]; ok {
		provider = p.Raw()
	}

	providerMetadata := &ProviderMetadata{
		Provider: provider,
	}

	return unillm.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}, providerMetadata
}

func languageModelStreamUsage(chunk openaisdk.ChatCompletionChunk, _ map[string]any, metadata unillm.ProviderMetadata) (unillm.Usage, unillm.ProviderMetadata) {
	usage := chunk.Usage
	if usage.TotalTokens == 0 {
		return unillm.Usage{}, nil
	}

	streamProviderMetadata := &ProviderMetadata{}
	if metadata != nil {
		if providerMetadata, ok := metadata[Name]; ok {
			converted, ok := providerMetadata.(*ProviderMetadata)
			if ok {
				streamProviderMetadata = converted
			}
		}
	}

	if p, ok := chunk.JSON.ExtraFields["provider"]; ok {
		streamProviderMetadata.Provider = p.Raw()
	}

	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails
	aiUsage := unillm.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}

	return aiUsage, unillm.ProviderMetadata{
		Name: streamProviderMetadata,
	}
}

func languageModelToPrompt(prompt unillm.Prompt, _, model string) ([]openaisdk.ChatCompletionMessageParamUnion, []unillm.CallWarning) {
	var messages []openaisdk.ChatCompletionMessageParamUnion
	var warnings []unillm.CallWarning

	for _, msg := range prompt {
		switch msg.Role {
		case unillm.MessageRoleSystem:
			var systemPromptParts []string
			for _, c := range msg.Content {
				if c.GetType() != unillm.ContentTypeText {
					warnings = append(warnings, unillm.CallWarning{
						Type:    unillm.CallWarningTypeOther,
						Message: "system prompt can only have text content",
					})
					continue
				}
				textPart, ok := unillm.AsContentType[unillm.TextPart](c)
				if !ok {
					warnings = append(warnings, unillm.CallWarning{
						Type:    unillm.CallWarningTypeOther,
						Message: "system prompt text part does not have the right type",
					})
					continue
				}
				text := textPart.Text
				if strings.TrimSpace(text) != "" {
					systemPromptParts = append(systemPromptParts, textPart.Text)
				}
			}
			if len(systemPromptParts) == 0 {
				warnings = append(warnings, unillm.CallWarning{
					Type:    unillm.CallWarningTypeOther,
					Message: "system prompt has no text parts",
				})
				continue
			}
			systemMsg := openaisdk.SystemMessage(strings.Join(systemPromptParts, "\n"))
			messages = append(messages, systemMsg)

		case unillm.MessageRoleUser:
			if len(msg.Content) == 1 && msg.Content[0].GetType() == unillm.ContentTypeText {
				textPart, ok := unillm.AsContentType[unillm.TextPart](msg.Content[0])
				if !ok {
					warnings = append(warnings, unillm.CallWarning{
						Type:    unillm.CallWarningTypeOther,
						Message: "user message text part does not have the right type",
					})
					continue
				}
				userMsg := openaisdk.UserMessage(textPart.Text)
				messages = append(messages, userMsg)
				continue
			}

			var content []openaisdk.ChatCompletionContentPartUnionParam
			for _, c := range msg.Content {
				switch c.GetType() {
				case unillm.ContentTypeText:
					textPart, ok := unillm.AsContentType[unillm.TextPart](c)
					if !ok {
						warnings = append(warnings, unillm.CallWarning{
							Type:    unillm.CallWarningTypeOther,
							Message: "user message text part does not have the right type",
						})
						continue
					}
					part := openaisdk.ChatCompletionContentPartUnionParam{
						OfText: &openaisdk.ChatCompletionContentPartTextParam{
							Text: textPart.Text,
						},
					}
					content = append(content, part)
				case unillm.ContentTypeFile:
					filePart, ok := unillm.AsContentType[unillm.FilePart](c)
					if !ok {
						warnings = append(warnings, unillm.CallWarning{
							Type:    unillm.CallWarningTypeOther,
							Message: "user message file part does not have the right type",
						})
						continue
					}
					switch {
					case strings.HasPrefix(filePart.MediaType, "image/"):
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						data := "data:" + filePart.MediaType + ";base64," + base64Encoded
						imageURL := openaisdk.ChatCompletionContentPartImageImageURLParam{URL: data}
						if providerOptions, ok := filePart.ProviderOptions[openaipkg.Name]; ok {
							if detail, ok := providerOptions.(*openaipkg.ProviderFileOptions); ok {
								imageURL.Detail = detail.ImageDetail
							}
						}
						imageBlock := openaisdk.ChatCompletionContentPartImageParam{ImageURL: imageURL}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock})

					case filePart.MediaType == "audio/wav":
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						audioBlock := openaisdk.ChatCompletionContentPartInputAudioParam{
							InputAudio: openaisdk.ChatCompletionContentPartInputAudioInputAudioParam{
								Data:   base64Encoded,
								Format: "wav",
							},
						}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock})

					case filePart.MediaType == "audio/mpeg" || filePart.MediaType == "audio/mp3":
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						audioBlock := openaisdk.ChatCompletionContentPartInputAudioParam{
							InputAudio: openaisdk.ChatCompletionContentPartInputAudioInputAudioParam{
								Data:   base64Encoded,
								Format: "mp3",
							},
						}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock})

					case filePart.MediaType == "application/pdf":
						dataStr := string(filePart.Data)
						if strings.HasPrefix(dataStr, "file-") {
							fileBlock := openaisdk.ChatCompletionContentPartFileParam{
								File: openaisdk.ChatCompletionContentPartFileFileParam{
									FileID: param.NewOpt(dataStr),
								},
							}
							content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfFile: &fileBlock})
						} else {
							base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
							data := "data:application/pdf;base64," + base64Encoded
							filename := filePart.Filename
							if filename == "" {
								filename = fmt.Sprintf("part-%d.pdf", len(content))
							}
							fileBlock := openaisdk.ChatCompletionContentPartFileParam{
								File: openaisdk.ChatCompletionContentPartFileFileParam{
									Filename: param.NewOpt(filename),
									FileData: param.NewOpt(data),
								},
							}
							content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfFile: &fileBlock})
						}

					default:
						warnings = append(warnings, unillm.CallWarning{
							Type:    unillm.CallWarningTypeOther,
							Message: fmt.Sprintf("file part media type %s not supported", filePart.MediaType),
						})
					}
				}
			}
			if !hasVisibleUserContent(content) {
				warnings = append(warnings, unillm.CallWarning{
					Type:    unillm.CallWarningTypeOther,
					Message: "dropping empty user message (contains neither user-facing content nor tool results)",
				})
				continue
			}
			messages = append(messages, openaisdk.UserMessage(content))

		case unillm.MessageRoleAssistant:
			if len(msg.Content) == 1 && msg.Content[0].GetType() == unillm.ContentTypeText {
				textPart, ok := unillm.AsContentType[unillm.TextPart](msg.Content[0])
				if !ok {
					warnings = append(warnings, unillm.CallWarning{
						Type:    unillm.CallWarningTypeOther,
						Message: "assistant message text part does not have the right type",
					})
					continue
				}
				assistantMsg := openaisdk.AssistantMessage(textPart.Text)
				messages = append(messages, assistantMsg)
				continue
			}

			assistantMsg := openaisdk.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}
			for _, c := range msg.Content {
				switch c.GetType() {
				case unillm.ContentTypeText:
					textPart, ok := unillm.AsContentType[unillm.TextPart](c)
					if !ok {
						warnings = append(warnings, unillm.CallWarning{
							Type:    unillm.CallWarningTypeOther,
							Message: "assistant message text part does not have the right type",
						})
						continue
					}
					if assistantMsg.Content.OfString.Valid() {
						textPart.Text = assistantMsg.Content.OfString.Value + "\n" + textPart.Text
					}
					assistantMsg.Content = openaisdk.ChatCompletionAssistantMessageParamContentUnion{
						OfString: param.NewOpt(textPart.Text),
					}
				case unillm.ContentTypeReasoning:
					reasoningPart, ok := unillm.AsContentType[unillm.ReasoningPart](c)
					if !ok {
						warnings = append(warnings, unillm.CallWarning{
							Type:    unillm.CallWarningTypeOther,
							Message: "assistant message reasoning part does not have the right type",
						})
						continue
					}
					var reasoningDetails []ReasoningDetail
					metadata := openaipkg.GetReasoningMetadata(reasoningPart.Options())
					if metadata != nil {
						for inx, summary := range metadata.Summary {
							if summary == "" {
								continue
							}
							reasoningDetails = append(reasoningDetails, ReasoningDetail{
								Type:    "reasoning.summary",
								Format:  "openai-responses-v1",
								Summary: summary,
								Index:   inx,
							})
						}
						if metadata.EncryptedContent != nil {
							reasoningDetails = append(reasoningDetails, ReasoningDetail{
								Type:   "reasoning.encrypted",
								Format: "openai-responses-v1",
								Data:   *metadata.EncryptedContent,
								ID:     metadata.ItemID,
							})
						}
					} else {
						reasoningDetails = append(reasoningDetails, ReasoningDetail{
							Type:   "reasoning.text",
							Text:   reasoningPart.Text,
							Format: "unknown",
						})
					}
					data, _ := json.Marshal(reasoningDetails)
					var reasoningDetailsMap []map[string]any
					_ = json.Unmarshal(data, &reasoningDetailsMap)
					assistantMsg.SetExtraFields(map[string]any{
						"reasoning_details": reasoningDetailsMap,
					})
				case unillm.ContentTypeToolCall:
					toolCallPart, ok := unillm.AsContentType[unillm.ToolCallPart](c)
					if ok {
						assistantMsg.ToolCalls = append(assistantMsg.ToolCalls,
							openaisdk.ChatCompletionMessageToolCallUnionParam{
								OfFunction: &openaisdk.ChatCompletionMessageFunctionToolCallParam{
									ID:   toolCallPart.ToolCallID,
									Type: "function",
									Function: openaisdk.ChatCompletionMessageFunctionToolCallFunctionParam{
										Name:      toolCallPart.ToolName,
										Arguments: toolCallPart.Input,
									},
								},
							})
					}
				}
			}
			messages = append(messages, openaisdk.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			})

		case unillm.MessageRoleTool:
			for _, c := range msg.Content {
				if c.GetType() != unillm.ContentTypeToolResult {
					continue
				}
				toolResultPart, ok := unillm.AsContentType[unillm.ToolResultPart](c)
				if !ok {
					continue
				}
				switch toolResultPart.Output.GetType() {
				case unillm.ToolResultContentTypeText:
					output, ok := unillm.AsToolResultOutputType[unillm.ToolResultOutputContentText](toolResultPart.Output)
					if ok {
						messages = append(messages, openaisdk.ToolMessage(output.Text, toolResultPart.ToolCallID))
					}
				case unillm.ToolResultContentTypeError:
					output, ok := unillm.AsToolResultOutputType[unillm.ToolResultOutputContentError](toolResultPart.Output)
					if ok {
						messages = append(messages, openaisdk.ToolMessage(output.Error.Error(), toolResultPart.ToolCallID))
					}
				}
			}
		}
	}
	return messages, warnings
}

func hasVisibleUserContent(parts []openaisdk.ChatCompletionContentPartUnionParam) bool {
	for _, part := range parts {
		if part.OfText != nil && strings.TrimSpace(part.OfText.Text) != "" {
			return true
		}
		if part.OfImageURL != nil || part.OfInputAudio != nil || part.OfFile != nil {
			return true
		}
	}
	return false
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
