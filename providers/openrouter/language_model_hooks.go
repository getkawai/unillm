package openrouter

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/getkawai/unillm"
	"github.com/getkawai/unillm/providers/openai"
	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

const reasoningStartedCtx = "reasoning_started"

func languagePrepareModelCall(_ fantasy.LanguageModel, params *openaisdk.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, &fantasy.Error{Title: "invalid argument", Message: "openrouter provider options should be *openrouter.ProviderOptions"}
		}
	}

	extraFields := make(map[string]any)

	if providerOptions.Provider != nil {
		data, err := structToMapJSON(providerOptions.Provider)
		if err != nil {
			return nil, err
		}
		extraFields["provider"] = data
	}

	if providerOptions.Reasoning != nil {
		data, err := structToMapJSON(providerOptions.Reasoning)
		if err != nil {
			return nil, err
		}
		extraFields["reasoning"] = data
	}

	if providerOptions.IncludeUsage != nil {
		extraFields["usage"] = map[string]any{
			"include": *providerOptions.IncludeUsage,
		}
	} else {
		extraFields["usage"] = map[string]any{
			"include": true,
		}
	}
	if providerOptions.LogitBias != nil {
		params.LogitBias = providerOptions.LogitBias
	}
	if providerOptions.LogProbs != nil {
		params.Logprobs = param.NewOpt(*providerOptions.LogProbs)
	}
	if providerOptions.User != nil {
		params.User = param.NewOpt(*providerOptions.User)
	}
	if providerOptions.ParallelToolCalls != nil {
		params.ParallelToolCalls = param.NewOpt(*providerOptions.ParallelToolCalls)
	}

	maps.Copy(extraFields, providerOptions.ExtraBody)
	params.SetExtraFields(extraFields)
	return nil, nil
}

func languageModelExtraContent(choice openaisdk.ChatCompletionChoice) []fantasy.Content {
	content := make([]fantasy.Content, 0)
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Message.RawJSON()), &reasoningData)
	if err != nil {
		return content
	}

	responsesReasoningBlocks := make([]openai.ResponsesReasoningMetadata, 0)
	otherReasoning := make([]string, 0)

	for _, detail := range reasoningData.ReasoningDetails {
		if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
			var thinkingBlock openai.ResponsesReasoningMetadata
			if len(responsesReasoningBlocks)-1 >= detail.Index {
				thinkingBlock = responsesReasoningBlocks[detail.Index]
			} else {
				thinkingBlock = openai.ResponsesReasoningMetadata{}
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

		otherReasoning = append(otherReasoning, detail.Text)
	}

	for _, block := range responsesReasoningBlocks {
		if len(block.Summary) == 0 {
			block.Summary = []string{""}
		}
		content = append(content, fantasy.ReasoningContent{
			Text: strings.Join(block.Summary, "\n"),
			ProviderMetadata: fantasy.ProviderMetadata{
				openai.Name: &block,
			},
		})
	}

	for _, reasoning := range otherReasoning {
		content = append(content, fantasy.ReasoningContent{
			Text: reasoning,
		})
	}
	return content
}

type currentReasoningState struct {
	metadata *openai.ResponsesReasoningMetadata
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

func languageModelStreamExtra(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
	if len(chunk.Choices) == 0 {
		return ctx, true
	}

	currentState := extractReasoningContext(ctx)

	inx := 0
	choice := chunk.Choices[inx]
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Delta.RawJSON()), &reasoningData)
	if err != nil {
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeError,
			Error: &fantasy.Error{Title: "stream error", Message: "error unmarshalling delta", Cause: err},
		})
		return ctx, false
	}

	if currentState == nil {
		if len(reasoningData.ReasoningDetails) == 0 {
			return ctx, true
		}

		var metadata fantasy.ProviderMetadata
		currentState = &currentReasoningState{}

		detail := reasoningData.ReasoningDetails[0]
		if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
			currentState.metadata = &openai.ResponsesReasoningMetadata{
				Summary: []string{detail.Summary},
			}
			metadata = fantasy.ProviderMetadata{
				openai.Name: currentState.metadata,
			}
			if detail.Data != "" {
				shouldContinue := yield(fantasy.StreamPart{
					Type:             fantasy.StreamPartTypeReasoningStart,
					ID:               fmt.Sprintf("%d", inx),
					Delta:            detail.Summary,
					ProviderMetadata: metadata,
				})
				if !shouldContinue {
					return ctx, false
				}
				return ctx, yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningEnd,
					ID:   fmt.Sprintf("%d", inx),
					ProviderMetadata: fantasy.ProviderMetadata{
						openai.Name: &openai.ResponsesReasoningMetadata{
							Summary:          []string{detail.Summary},
							EncryptedContent: &detail.Data,
							ItemID:           detail.ID,
						},
					},
				})
			}
		}

		ctx[reasoningStartedCtx] = currentState
		return ctx, yield(fantasy.StreamPart{
			Type:             fantasy.StreamPartTypeReasoningStart,
			ID:               fmt.Sprintf("%d", inx),
			Delta:            detail.Summary,
			ProviderMetadata: metadata,
		})
	}

	if len(reasoningData.ReasoningDetails) == 0 {
		if choice.Delta.Content != "" || len(choice.Delta.ToolCalls) > 0 {
			ctx[reasoningStartedCtx] = nil
			return ctx, yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeReasoningEnd,
				ID:   fmt.Sprintf("%d", inx),
			})
		}
		return ctx, true
	}

	detail := reasoningData.ReasoningDetails[0]
	if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
		if detail.Data != "" {
			currentState.metadata.EncryptedContent = &detail.Data
			currentState.metadata.ItemID = detail.ID
			ctx[reasoningStartedCtx] = nil
			return ctx, yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeReasoningEnd,
				ID:   fmt.Sprintf("%d", inx),
				ProviderMetadata: fantasy.ProviderMetadata{
					openai.Name: currentState.metadata,
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
		return ctx, yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeReasoningDelta,
			ID:    fmt.Sprintf("%d", inx),
			Delta: textDelta,
			ProviderMetadata: fantasy.ProviderMetadata{
				openai.Name: currentState.metadata,
			},
		})
	}

	return ctx, yield(fantasy.StreamPart{
		Type:  fantasy.StreamPartTypeReasoningDelta,
		ID:    fmt.Sprintf("%d", inx),
		Delta: detail.Text,
	})
}

func languageModelUsage(response openaisdk.ChatCompletion) (fantasy.Usage, fantasy.ProviderOptionsData) {
	if len(response.Choices) == 0 {
		return fantasy.Usage{}, nil
	}
	openrouterUsage := UsageAccounting{}
	usage := response.Usage

	_ = json.Unmarshal([]byte(usage.RawJSON()), &openrouterUsage)

	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails

	var provider string
	if p, ok := response.JSON.ExtraFields["provider"]; ok {
		provider = p.Raw()
	}

	providerMetadata := &ProviderMetadata{
		Provider: provider,
		Usage:    openrouterUsage,
	}

	return fantasy.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}, providerMetadata
}

func languageModelStreamUsage(chunk openaisdk.ChatCompletionChunk, _ map[string]any, metadata fantasy.ProviderMetadata) (fantasy.Usage, fantasy.ProviderMetadata) {
	usage := chunk.Usage
	if usage.TotalTokens == 0 {
		return fantasy.Usage{}, nil
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
	openrouterUsage := UsageAccounting{}
	_ = json.Unmarshal([]byte(usage.RawJSON()), &openrouterUsage)
	streamProviderMetadata.Usage = openrouterUsage

	if p, ok := chunk.JSON.ExtraFields["provider"]; ok {
		streamProviderMetadata.Provider = p.Raw()
	}

	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails
	aiUsage := fantasy.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}

	return aiUsage, fantasy.ProviderMetadata{
		Name: streamProviderMetadata,
	}
}

func languageModelToPrompt(prompt fantasy.Prompt, _, model string) ([]openaisdk.ChatCompletionMessageParamUnion, []fantasy.CallWarning) {
	var messages []openaisdk.ChatCompletionMessageParamUnion
	var warnings []fantasy.CallWarning

	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			var systemPromptParts []string
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeText {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "system prompt can only have text content",
					})
					continue
				}
				textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
				if !ok {
					continue
				}
				if strings.TrimSpace(textPart.Text) != "" {
					systemPromptParts = append(systemPromptParts, textPart.Text)
				}
			}
			if len(systemPromptParts) == 0 {
				continue
			}
			messages = append(messages, openaisdk.SystemMessage(strings.Join(systemPromptParts, "\n")))

		case fantasy.MessageRoleUser:
			if len(msg.Content) == 1 && msg.Content[0].GetType() == fantasy.ContentTypeText {
				textPart, ok := fantasy.AsContentType[fantasy.TextPart](msg.Content[0])
				if ok {
					messages = append(messages, openaisdk.UserMessage(textPart.Text))
					continue
				}
			}
			var content []openaisdk.ChatCompletionContentPartUnionParam
			for _, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
					if ok {
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{
							OfText: &openaisdk.ChatCompletionContentPartTextParam{
								Text: textPart.Text,
							},
						})
					}
				case fantasy.ContentTypeFile:
					filePart, ok := fantasy.AsContentType[fantasy.FilePart](c)
					if !ok {
						continue
					}
					if strings.HasPrefix(filePart.MediaType, "image/") {
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						data := "data:" + filePart.MediaType + ";base64," + base64Encoded
						imageURL := openaisdk.ChatCompletionContentPartImageImageURLParam{URL: data}
						if providerOptions, ok := filePart.ProviderOptions[Name]; ok {
							if detail, ok := providerOptions.(*openai.ProviderFileOptions); ok {
								imageURL.Detail = detail.ImageDetail
							}
						}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{
							OfImageURL: &openaisdk.ChatCompletionContentPartImageParam{ImageURL: imageURL},
						})
					} else if filePart.MediaType == "audio/wav" || filePart.MediaType == "audio/mpeg" || filePart.MediaType == "audio/mp3" {
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						format := "wav"
						if filePart.MediaType == "audio/mpeg" || filePart.MediaType == "audio/mp3" {
							format = "mp3"
						}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{
							OfInputAudio: &openaisdk.ChatCompletionContentPartInputAudioParam{
								InputAudio: openaisdk.ChatCompletionContentPartInputAudioInputAudioParam{
									Data:   base64Encoded,
									Format: format,
								},
							},
						})
					} else if filePart.MediaType == "application/pdf" {
						dataStr := string(filePart.Data)
						if strings.HasPrefix(dataStr, "file-") {
							content = append(content, openaisdk.ChatCompletionContentPartUnionParam{
								OfFile: &openaisdk.ChatCompletionContentPartFileParam{
									File: openaisdk.ChatCompletionContentPartFileFileParam{
										FileID: param.NewOpt(dataStr),
									},
								},
							})
						} else {
							base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
							data := "data:application/pdf;base64," + base64Encoded
							filename := filePart.Filename
							if filename == "" {
								filename = fmt.Sprintf("part-%d.pdf", len(content))
							}
							content = append(content, openaisdk.ChatCompletionContentPartUnionParam{
								OfFile: &openaisdk.ChatCompletionContentPartFileParam{
									File: openaisdk.ChatCompletionContentPartFileFileParam{
										Filename: param.NewOpt(filename),
										FileData: param.NewOpt(data),
									},
								},
							})
						}
					}
				}
			}
			messages = append(messages, openaisdk.UserMessage(content))

		case fantasy.MessageRoleAssistant:
			if len(msg.Content) == 1 && msg.Content[0].GetType() == fantasy.ContentTypeText {
				textPart, ok := fantasy.AsContentType[fantasy.TextPart](msg.Content[0])
				if ok {
					messages = append(messages, openaisdk.AssistantMessage(textPart.Text))
					continue
				}
			}
			assistantMsg := openaisdk.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}
			for _, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
					if ok {
						assistantMsg.Content = openaisdk.ChatCompletionAssistantMessageParamContentUnion{
							OfString: param.NewOpt(textPart.Text),
						}
					}
				case fantasy.ContentTypeReasoning:
					reasoningPart, ok := fantasy.AsContentType[fantasy.ReasoningPart](c)
					if !ok {
						continue
					}
					var reasoningDetails []ReasoningDetail
					metadata := openai.GetReasoningMetadata(reasoningPart.Options())
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
				case fantasy.ContentTypeToolCall:
					toolCallPart, ok := fantasy.AsContentType[fantasy.ToolCallPart](c)
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

		case fantasy.MessageRoleTool:
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeToolResult {
					continue
				}
				toolResultPart, ok := fantasy.AsContentType[fantasy.ToolResultPart](c)
				if !ok {
					continue
				}
				switch toolResultPart.Output.GetType() {
				case fantasy.ToolResultContentTypeText:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](toolResultPart.Output)
					if ok {
						messages = append(messages, openaisdk.ToolMessage(output.Text, toolResultPart.ToolCallID))
					}
				case fantasy.ToolResultContentTypeError:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](toolResultPart.Output)
					if ok {
						messages = append(messages, openaisdk.ToolMessage(output.Error.Error(), toolResultPart.ToolCallID))
					}
				}
			}
		}
	}
	return messages, warnings
}
