package google

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/charmbracelet/x/exp/slice"
	"github.com/getkawai/unillm"
	"github.com/getkawai/unillm/object"
	"github.com/getkawai/unillm/schema"
	"google.golang.org/genai"
)

// languageModel implements unillm.LanguageModel for Google's language models.
type languageModel struct {
	provider        string
	modelID         string
	client          *genai.Client
	providerOptions options
	objectMode      unillm.ObjectMode
}

// Model implements unillm.LanguageModel.
func (g *languageModel) Model() string {
	return g.modelID
}

// Provider implements unillm.LanguageModel.
func (g *languageModel) Provider() string {
	return g.provider
}

func (g languageModel) prepareParams(call unillm.Call) (*genai.GenerateContentConfig, []*genai.Content, []unillm.CallWarning, error) {
	config := &genai.GenerateContentConfig{}

	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, nil, nil, &unillm.Error{Title: "invalid argument", Message: "google provider options should be *google.ProviderOptions"}
		}
	}

	systemInstructions, content, warnings := toGooglePrompt(call.Prompt)

	if providerOptions.ThinkingConfig != nil {
		if providerOptions.ThinkingConfig.IncludeThoughts != nil &&
			*providerOptions.ThinkingConfig.IncludeThoughts &&
			g.providerOptions.backend != genai.BackendVertexAI {
			warnings = append(warnings, unillm.CallWarning{
				Type: unillm.CallWarningTypeOther,
				Message: "The 'includeThoughts' option is only supported with the Google Vertex AI backend " +
					"and might not be supported or could behave unexpectedly with the current backend " +
					fmt.Sprintf("(backend: %v)", g.providerOptions.backend),
			})
		}

		if providerOptions.ThinkingConfig.ThinkingBudget != nil &&
			*providerOptions.ThinkingConfig.ThinkingBudget < 128 {
			warnings = append(warnings, unillm.CallWarning{
				Type:    unillm.CallWarningTypeOther,
				Message: "The 'thinking_budget' option can not be under 128 and will be set to 128 by default",
			})
			providerOptions.ThinkingConfig.ThinkingBudget = unillm.Opt(int64(128))
		}
	}

	isGemmaModel := strings.HasPrefix(strings.ToLower(g.modelID), "gemma-")

	if isGemmaModel && systemInstructions != nil && len(systemInstructions.Parts) > 0 {
		if len(content) > 0 && content[0].Role == genai.RoleUser {
			systemParts := []string{}
			for _, sp := range systemInstructions.Parts {
				systemParts = append(systemParts, sp.Text)
			}
			systemMsg := strings.Join(systemParts, "\n")
			content[0].Parts = append([]*genai.Part{
				{
					Text: systemMsg + "\n\n",
				},
			}, content[0].Parts...)
			systemInstructions = nil
		}
	}

	config.SystemInstruction = systemInstructions

	if call.MaxOutputTokens != nil {
		config.MaxOutputTokens = int32(*call.MaxOutputTokens) //nolint: gosec
	}

	if call.Temperature != nil {
		tmp := float32(*call.Temperature)
		config.Temperature = &tmp
	}
	if call.TopK != nil {
		tmp := float32(*call.TopK)
		config.TopK = &tmp
	}
	if call.TopP != nil {
		tmp := float32(*call.TopP)
		config.TopP = &tmp
	}
	if call.FrequencyPenalty != nil {
		tmp := float32(*call.FrequencyPenalty)
		config.FrequencyPenalty = &tmp
	}
	if call.PresencePenalty != nil {
		tmp := float32(*call.PresencePenalty)
		config.PresencePenalty = &tmp
	}

	if providerOptions.ThinkingConfig != nil {
		config.ThinkingConfig = &genai.ThinkingConfig{}
		if providerOptions.ThinkingConfig.IncludeThoughts != nil {
			config.ThinkingConfig.IncludeThoughts = *providerOptions.ThinkingConfig.IncludeThoughts
		}
		if providerOptions.ThinkingConfig.ThinkingBudget != nil {
			tmp := int32(*providerOptions.ThinkingConfig.ThinkingBudget) //nolint: gosec
			config.ThinkingConfig.ThinkingBudget = &tmp
		}
	}
	for _, safetySetting := range providerOptions.SafetySettings {
		config.SafetySettings = append(config.SafetySettings, &genai.SafetySetting{
			Category:  genai.HarmCategory(safetySetting.Category),
			Threshold: genai.HarmBlockThreshold(safetySetting.Threshold),
		})
	}
	if providerOptions.CachedContent != "" {
		config.CachedContent = providerOptions.CachedContent
	}

	if len(call.Tools) > 0 {
		tools, toolChoice, toolWarnings := toGoogleTools(call.Tools, call.ToolChoice)
		config.ToolConfig = toolChoice
		config.Tools = append(config.Tools, &genai.Tool{
			FunctionDeclarations: tools,
		})
		warnings = append(warnings, toolWarnings...)
	}

	return config, content, warnings, nil
}
func (g *languageModel) Generate(ctx context.Context, call unillm.Call) (*unillm.Response, error) {
	config, contents, warnings, err := g.prepareParams(call)
	if err != nil {
		return nil, err
	}

	lastMessage, history, ok := slice.Pop(contents)
	if !ok {
		return nil, errors.New("no messages to send")
	}

	chat, err := g.client.Chats.Create(ctx, g.modelID, config, history)
	if err != nil {
		return nil, err
	}

	response, err := chat.SendMessage(ctx, depointerSlice(lastMessage.Parts)...)
	if err != nil {
		return nil, toProviderErr(err)
	}

	return g.mapResponse(response, warnings)
}

func (g *languageModel) Stream(ctx context.Context, call unillm.Call) (unillm.StreamResponse, error) {
	config, contents, warnings, err := g.prepareParams(call)
	if err != nil {
		return nil, err
	}

	lastMessage, history, ok := slice.Pop(contents)
	if !ok {
		return nil, errors.New("no messages to send")
	}

	chat, err := g.client.Chats.Create(ctx, g.modelID, config, history)
	if err != nil {
		return nil, err
	}

	return func(yield func(unillm.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(unillm.StreamPart{
				Type:     unillm.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		var currentContent string
		var toolCalls []unillm.ToolCallContent
		var isActiveText bool
		var isActiveReasoning bool
		var blockCounter int
		var currentTextBlockID string
		var currentReasoningBlockID string
		var usage *unillm.Usage
		var lastFinishReason unillm.FinishReason

		for resp, err := range chat.SendMessageStream(ctx, depointerSlice(lastMessage.Parts)...) {
			if err != nil {
				yield(unillm.StreamPart{
					Type:  unillm.StreamPartTypeError,
					Error: toProviderErr(err),
				})
				return
			}

			if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				for _, part := range resp.Candidates[0].Content.Parts {
					switch {
					case part.Text != "":
						delta := part.Text
						if delta != "" {
							// Check if this is a reasoning/thought part
							if part.Thought {
								// End any active text block before starting reasoning
								if isActiveText {
									isActiveText = false
									if !yield(unillm.StreamPart{
										Type: unillm.StreamPartTypeTextEnd,
										ID:   currentTextBlockID,
									}) {
										return
									}
								}

								// Start new reasoning block if not already active
								if !isActiveReasoning {
									isActiveReasoning = true
									currentReasoningBlockID = fmt.Sprintf("%d", blockCounter)
									blockCounter++
									if !yield(unillm.StreamPart{
										Type: unillm.StreamPartTypeReasoningStart,
										ID:   currentReasoningBlockID,
									}) {
										return
									}
								}

								if !yield(unillm.StreamPart{
									Type:  unillm.StreamPartTypeReasoningDelta,
									ID:    currentReasoningBlockID,
									Delta: delta,
								}) {
									return
								}
							} else {
								// Start new text block if not already active
								if !isActiveText {
									isActiveText = true
									currentTextBlockID = fmt.Sprintf("%d", blockCounter)
									blockCounter++
									if !yield(unillm.StreamPart{
										Type: unillm.StreamPartTypeTextStart,
										ID:   currentTextBlockID,
									}) {
										return
									}
								}
								// End any active reasoning block before starting text
								if isActiveReasoning {
									isActiveReasoning = false
									metadata := &ReasoningMetadata{
										Signature: string(part.ThoughtSignature),
									}
									if !yield(unillm.StreamPart{
										Type: unillm.StreamPartTypeReasoningEnd,
										ID:   currentReasoningBlockID,
										ProviderMetadata: unillm.ProviderMetadata{
											Name: metadata,
										},
									}) {
										return
									}
								} else if part.ThoughtSignature != nil {
									metadata := &ReasoningMetadata{
										Signature: string(part.ThoughtSignature),
									}

									if !yield(unillm.StreamPart{
										Type: unillm.StreamPartTypeReasoningStart,
										ID:   currentReasoningBlockID,
									}) {
										return
									}
									if !yield(unillm.StreamPart{
										Type: unillm.StreamPartTypeReasoningEnd,
										ID:   currentReasoningBlockID,
										ProviderMetadata: unillm.ProviderMetadata{
											Name: metadata,
										},
									}) {
										return
									}
								}

								if !yield(unillm.StreamPart{
									Type:  unillm.StreamPartTypeTextDelta,
									ID:    currentTextBlockID,
									Delta: delta,
								}) {
									return
								}
								currentContent += delta
							}
						}
					case part.FunctionCall != nil:
						// End any active text or reasoning blocks
						if isActiveText {
							isActiveText = false
							if !yield(unillm.StreamPart{
								Type: unillm.StreamPartTypeTextEnd,
								ID:   currentTextBlockID,
							}) {
								return
							}
						}
						toolCallID := cmp.Or(part.FunctionCall.ID, g.providerOptions.toolCallIDFunc())
						// End any active reasoning block before starting text
						if isActiveReasoning {
							isActiveReasoning = false
							metadata := &ReasoningMetadata{
								Signature: string(part.ThoughtSignature),
								ToolID:    toolCallID,
							}
							if !yield(unillm.StreamPart{
								Type: unillm.StreamPartTypeReasoningEnd,
								ID:   currentReasoningBlockID,
								ProviderMetadata: unillm.ProviderMetadata{
									Name: metadata,
								},
							}) {
								return
							}
						} else if part.ThoughtSignature != nil {
							metadata := &ReasoningMetadata{
								Signature: string(part.ThoughtSignature),
								ToolID:    toolCallID,
							}

							if !yield(unillm.StreamPart{
								Type: unillm.StreamPartTypeReasoningStart,
								ID:   currentReasoningBlockID,
							}) {
								return
							}
							if !yield(unillm.StreamPart{
								Type: unillm.StreamPartTypeReasoningEnd,
								ID:   currentReasoningBlockID,
								ProviderMetadata: unillm.ProviderMetadata{
									Name: metadata,
								},
							}) {
								return
							}
						}
						args, err := json.Marshal(part.FunctionCall.Args)
						if err != nil {
							yield(unillm.StreamPart{
								Type:  unillm.StreamPartTypeError,
								Error: err,
							})
							return
						}

						if !yield(unillm.StreamPart{
							Type:         unillm.StreamPartTypeToolInputStart,
							ID:           toolCallID,
							ToolCallName: part.FunctionCall.Name,
						}) {
							return
						}

						if !yield(unillm.StreamPart{
							Type:  unillm.StreamPartTypeToolInputDelta,
							ID:    toolCallID,
							Delta: string(args),
						}) {
							return
						}

						if !yield(unillm.StreamPart{
							Type: unillm.StreamPartTypeToolInputEnd,
							ID:   toolCallID,
						}) {
							return
						}

						if !yield(unillm.StreamPart{
							Type:             unillm.StreamPartTypeToolCall,
							ID:               toolCallID,
							ToolCallName:     part.FunctionCall.Name,
							ToolCallInput:    string(args),
							ProviderExecuted: false,
						}) {
							return
						}

						toolCalls = append(toolCalls, unillm.ToolCallContent{
							ToolCallID:       toolCallID,
							ToolName:         part.FunctionCall.Name,
							Input:            string(args),
							ProviderExecuted: false,
						})
					}
				}
			}

			// we need to make sure that there is actual tokendata
			if resp.UsageMetadata != nil && resp.UsageMetadata.TotalTokenCount != 0 {
				currentUsage := mapUsage(resp.UsageMetadata)
				// if first usage chunk
				if usage == nil {
					usage = &currentUsage
				} else {
					usage.OutputTokens += currentUsage.OutputTokens
					usage.ReasoningTokens += currentUsage.ReasoningTokens
					usage.CacheReadTokens += currentUsage.CacheReadTokens
				}
			}

			if len(resp.Candidates) > 0 && resp.Candidates[0].FinishReason != "" {
				lastFinishReason = mapFinishReason(resp.Candidates[0].FinishReason)
			}
		}

		// Close any open blocks before finishing
		if isActiveText {
			if !yield(unillm.StreamPart{
				Type: unillm.StreamPartTypeTextEnd,
				ID:   currentTextBlockID,
			}) {
				return
			}
		}
		if isActiveReasoning {
			if !yield(unillm.StreamPart{
				Type: unillm.StreamPartTypeReasoningEnd,
				ID:   currentReasoningBlockID,
			}) {
				return
			}
		}

		finishReason := lastFinishReason
		if len(toolCalls) > 0 {
			finishReason = unillm.FinishReasonToolCalls
		} else if finishReason == "" {
			finishReason = unillm.FinishReasonStop
		}

		yield(unillm.StreamPart{
			Type:         unillm.StreamPartTypeFinish,
			Usage:        *usage,
			FinishReason: finishReason,
		})
	}, nil
}

func (g *languageModel) GenerateObject(ctx context.Context, call unillm.ObjectCall) (*unillm.ObjectResponse, error) {
	switch g.objectMode {
	case unillm.ObjectModeText:
		return object.GenerateWithText(ctx, g, call)
	case unillm.ObjectModeTool:
		return object.GenerateWithTool(ctx, g, call)
	default:
		return g.generateObjectWithJSONMode(ctx, call)
	}
}

// StreamObject implements unillm.LanguageModel.
func (g *languageModel) StreamObject(ctx context.Context, call unillm.ObjectCall) (unillm.ObjectStreamResponse, error) {
	switch g.objectMode {
	case unillm.ObjectModeTool:
		return object.StreamWithTool(ctx, g, call)
	case unillm.ObjectModeText:
		return object.StreamWithText(ctx, g, call)
	default:
		return g.streamObjectWithJSONMode(ctx, call)
	}
}

func (g *languageModel) generateObjectWithJSONMode(ctx context.Context, call unillm.ObjectCall) (*unillm.ObjectResponse, error) {
	// Convert our Schema to Google's JSON Schema format
	jsonSchemaMap := schema.ToMap(call.Schema)

	// Build request using prepareParams
	fantasyCall := unillm.Call{
		Prompt:           call.Prompt,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		TopK:             call.TopK,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	}

	config, contents, warnings, err := g.prepareParams(fantasyCall)
	if err != nil {
		return nil, err
	}

	// Set ResponseMIMEType and ResponseJsonSchema for structured output
	config.ResponseMIMEType = "application/json"
	config.ResponseJsonSchema = jsonSchemaMap

	lastMessage, history, ok := slice.Pop(contents)
	if !ok {
		return nil, errors.New("no messages to send")
	}

	chat, err := g.client.Chats.Create(ctx, g.modelID, config, history)
	if err != nil {
		return nil, err
	}

	response, err := chat.SendMessage(ctx, depointerSlice(lastMessage.Parts)...)
	if err != nil {
		return nil, toProviderErr(err)
	}

	mappedResponse, err := g.mapResponse(response, warnings)
	if err != nil {
		return nil, err
	}

	jsonText := mappedResponse.Content.Text()
	if jsonText == "" {
		return nil, &unillm.NoObjectGeneratedError{
			RawText:      "",
			ParseError:   fmt.Errorf("no text content in response"),
			Usage:        mappedResponse.Usage,
			FinishReason: mappedResponse.FinishReason,
		}
	}

	// Parse and validate
	var obj any
	if call.RepairText != nil {
		obj, err = schema.ParseAndValidateWithRepair(ctx, jsonText, call.Schema, call.RepairText)
	} else {
		obj, err = schema.ParseAndValidate(jsonText, call.Schema)
	}

	if err != nil {
		// Add usage info to error
		if nogErr, ok := err.(*unillm.NoObjectGeneratedError); ok {
			nogErr.Usage = mappedResponse.Usage
			nogErr.FinishReason = mappedResponse.FinishReason
		}
		return nil, err
	}

	return &unillm.ObjectResponse{
		Object:           obj,
		RawText:          jsonText,
		Usage:            mappedResponse.Usage,
		FinishReason:     mappedResponse.FinishReason,
		Warnings:         warnings,
		ProviderMetadata: mappedResponse.ProviderMetadata,
	}, nil
}

func (g *languageModel) streamObjectWithJSONMode(ctx context.Context, call unillm.ObjectCall) (unillm.ObjectStreamResponse, error) {
	// Convert our Schema to Google's JSON Schema format
	jsonSchemaMap := schema.ToMap(call.Schema)

	// Build request using prepareParams
	fantasyCall := unillm.Call{
		Prompt:           call.Prompt,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		TopK:             call.TopK,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	}

	config, contents, warnings, err := g.prepareParams(fantasyCall)
	if err != nil {
		return nil, err
	}

	// Set ResponseMIMEType and ResponseJsonSchema for structured output
	config.ResponseMIMEType = "application/json"
	config.ResponseJsonSchema = jsonSchemaMap

	lastMessage, history, ok := slice.Pop(contents)
	if !ok {
		return nil, errors.New("no messages to send")
	}

	chat, err := g.client.Chats.Create(ctx, g.modelID, config, history)
	if err != nil {
		return nil, err
	}

	return func(yield func(unillm.ObjectStreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(unillm.ObjectStreamPart{
				Type:     unillm.ObjectStreamPartTypeObject,
				Warnings: warnings,
			}) {
				return
			}
		}

		var accumulated string
		var lastParsedObject any
		var usage *unillm.Usage
		var lastFinishReason unillm.FinishReason
		var streamErr error

		for resp, err := range chat.SendMessageStream(ctx, depointerSlice(lastMessage.Parts)...) {
			if err != nil {
				streamErr = toProviderErr(err)
				yield(unillm.ObjectStreamPart{
					Type:  unillm.ObjectStreamPartTypeError,
					Error: streamErr,
				})
				return
			}

			if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				for _, part := range resp.Candidates[0].Content.Parts {
					if part.Text != "" && !part.Thought {
						accumulated += part.Text

						// Try to parse the accumulated text
						obj, state, parseErr := schema.ParsePartialJSON(accumulated)

						// If we successfully parsed, validate and emit
						if state == schema.ParseStateSuccessful || state == schema.ParseStateRepaired {
							if err := schema.ValidateAgainstSchema(obj, call.Schema); err == nil {
								// Only emit if object is different from last
								if !reflect.DeepEqual(obj, lastParsedObject) {
									if !yield(unillm.ObjectStreamPart{
										Type:   unillm.ObjectStreamPartTypeObject,
										Object: obj,
									}) {
										return
									}
									lastParsedObject = obj
								}
							}
						}

						// If parsing failed and we have a repair function, try it
						if state == schema.ParseStateFailed && call.RepairText != nil {
							repairedText, repairErr := call.RepairText(ctx, accumulated, parseErr)
							if repairErr == nil {
								obj2, state2, _ := schema.ParsePartialJSON(repairedText)
								if (state2 == schema.ParseStateSuccessful || state2 == schema.ParseStateRepaired) &&
									schema.ValidateAgainstSchema(obj2, call.Schema) == nil {
									if !reflect.DeepEqual(obj2, lastParsedObject) {
										if !yield(unillm.ObjectStreamPart{
											Type:   unillm.ObjectStreamPartTypeObject,
											Object: obj2,
										}) {
											return
										}
										lastParsedObject = obj2
									}
								}
							}
						}
					}
				}
			}

			// we need to make sure that there is actual tokendata
			if resp.UsageMetadata != nil && resp.UsageMetadata.TotalTokenCount != 0 {
				currentUsage := mapUsage(resp.UsageMetadata)
				if usage == nil {
					usage = &currentUsage
				} else {
					usage.OutputTokens += currentUsage.OutputTokens
					usage.ReasoningTokens += currentUsage.ReasoningTokens
					usage.CacheReadTokens += currentUsage.CacheReadTokens
				}
			}

			if len(resp.Candidates) > 0 && resp.Candidates[0].FinishReason != "" {
				lastFinishReason = mapFinishReason(resp.Candidates[0].FinishReason)
			}
		}

		// Final validation and emit
		if streamErr == nil && lastParsedObject != nil {
			finishReason := lastFinishReason
			if finishReason == "" {
				finishReason = unillm.FinishReasonStop
			}

			yield(unillm.ObjectStreamPart{
				Type:         unillm.ObjectStreamPartTypeFinish,
				Usage:        *usage,
				FinishReason: finishReason,
			})
		} else if streamErr == nil && lastParsedObject == nil {
			// No object was generated
			finalUsage := unillm.Usage{}
			if usage != nil {
				finalUsage = *usage
			}
			yield(unillm.ObjectStreamPart{
				Type: unillm.ObjectStreamPartTypeError,
				Error: &unillm.NoObjectGeneratedError{
					RawText:      accumulated,
					ParseError:   fmt.Errorf("no valid object generated in stream"),
					Usage:        finalUsage,
					FinishReason: lastFinishReason,
				},
			})
		}
	}, nil
}
func (g languageModel) mapResponse(response *genai.GenerateContentResponse, warnings []unillm.CallWarning) (*unillm.Response, error) {
	if len(response.Candidates) == 0 || response.Candidates[0].Content == nil {
		return nil, errors.New("no response from model")
	}

	var (
		content      []unillm.Content
		finishReason unillm.FinishReason
		hasToolCalls bool
		candidate    = response.Candidates[0]
	)

	for _, part := range candidate.Content.Parts {
		switch {
		case part.Text != "":
			if part.Thought {
				reasoningContent := unillm.ReasoningContent{Text: part.Text}
				if part.ThoughtSignature != nil {
					metadata := &ReasoningMetadata{
						Signature: string(part.ThoughtSignature),
					}
					reasoningContent.ProviderMetadata = unillm.ProviderMetadata{
						Name: metadata,
					}
				}
				content = append(content, reasoningContent)
			} else {
				foundReasoning := false
				if part.ThoughtSignature != nil {
					metadata := &ReasoningMetadata{
						Signature: string(part.ThoughtSignature),
					}
					// find the last reasoning content and add the signature
					for i := len(content) - 1; i >= 0; i-- {
						c := content[i]
						if c.GetType() == unillm.ContentTypeReasoning {
							reasoningContent, ok := unillm.AsContentType[unillm.ReasoningContent](c)
							if !ok {
								continue
							}
							reasoningContent.ProviderMetadata = unillm.ProviderMetadata{
								Name: metadata,
							}
							content[i] = reasoningContent
							foundReasoning = true
							break
						}
					}
					if !foundReasoning {
						content = append(content, unillm.ReasoningContent{
							ProviderMetadata: unillm.ProviderMetadata{
								Name: metadata,
							},
						})
					}
				}
				content = append(content, unillm.TextContent{Text: part.Text})
			}
		case part.FunctionCall != nil:
			input, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, err
			}
			toolCallID := cmp.Or(part.FunctionCall.ID, g.providerOptions.toolCallIDFunc())
			foundReasoning := false
			if part.ThoughtSignature != nil {
				metadata := &ReasoningMetadata{
					Signature: string(part.ThoughtSignature),
					ToolID:    toolCallID,
				}
				// find the last reasoning content and add the signature
				for i := len(content) - 1; i >= 0; i-- {
					c := content[i]
					if c.GetType() == unillm.ContentTypeReasoning {
						reasoningContent, ok := unillm.AsContentType[unillm.ReasoningContent](c)
						if !ok {
							continue
						}
						reasoningContent.ProviderMetadata = unillm.ProviderMetadata{
							Name: metadata,
						}
						content[i] = reasoningContent
						foundReasoning = true
						break
					}
				}
				if !foundReasoning {
					content = append(content, unillm.ReasoningContent{
						ProviderMetadata: unillm.ProviderMetadata{
							Name: metadata,
						},
					})
				}
			}
			content = append(content, unillm.ToolCallContent{
				ToolCallID:       toolCallID,
				ToolName:         part.FunctionCall.Name,
				Input:            string(input),
				ProviderExecuted: false,
			})
			hasToolCalls = true
		default:
			// Silently skip unknown part types instead of erroring
			// This allows for forward compatibility with new part types
		}
	}

	if hasToolCalls {
		finishReason = unillm.FinishReasonToolCalls
	} else {
		finishReason = mapFinishReason(candidate.FinishReason)
	}

	return &unillm.Response{
		Content:      content,
		Usage:        mapUsage(response.UsageMetadata),
		FinishReason: finishReason,
		Warnings:     warnings,
	}, nil
}
