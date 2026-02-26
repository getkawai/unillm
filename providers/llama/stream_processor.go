package llama

import (
	"strings"

	"github.com/getkawai/tools"
	"github.com/getkawai/unillm"
	"github.com/google/uuid"
)

type streamState int

const (
	stateText streamState = iota
	stateReasoning
	stateToolCall
)

type streamProcessor struct {
	yield func(unillm.StreamPart) bool
	state streamState

	textStarted      bool
	reasoningStarted bool
	toolInputStarted bool
	hasToolCalls     bool

	buffer        strings.Builder
	currentToolID string
}

func newStreamProcessor(yield func(unillm.StreamPart) bool) *streamProcessor {
	return &streamProcessor{
		yield: yield,
		state: stateText,
	}
}

func (p *streamProcessor) Process(token string) bool {
	p.buffer.WriteString(token)
	fullText := p.buffer.String()

	for {
		switch p.state {
		case stateText:
			// Check for reasoning start
			if idx := strings.Index(fullText, "<think>"); idx != -1 {
				p.emitText(fullText[:idx])
				p.state = stateReasoning
				p.buffer.Reset()
				p.buffer.WriteString(fullText[idx+len("<think>"):])
				fullText = p.buffer.String()
				continue
			}
			if idx := strings.Index(fullText, "<thought>"); idx != -1 {
				p.emitText(fullText[:idx])
				p.state = stateReasoning
				p.buffer.Reset()
				p.buffer.WriteString(fullText[idx+len("<thought>"):])
				fullText = p.buffer.String()
				continue
			}

			// Check for tool call start
			if idx := strings.Index(fullText, "<tool_call>"); idx != -1 {
				p.emitText(fullText[:idx])
				p.state = stateToolCall
				p.buffer.Reset()
				p.buffer.WriteString(fullText[idx+len("<tool_call>"):])
				fullText = p.buffer.String()
				continue
			}

			// If no tags found, but buffer is getting long, emit some text
			// but keep some back in case it's a partial tag
			if len(fullText) > 20 {
				emitLen := len(fullText) - 15 // Keep 15 chars for tag detection
				p.emitText(fullText[:emitLen])
				p.buffer.Reset()
				p.buffer.WriteString(fullText[emitLen:])
				return true
			}
			return true

		case stateReasoning:
			// Check for reasoning end
			endTag := ""
			if strings.Contains(fullText, "</think>") {
				endTag = "</think>"
			} else if strings.Contains(fullText, "</thought>") {
				endTag = "</thought>"
			}

			if endTag != "" {
				idx := strings.Index(fullText, endTag)
				p.emitReasoning(fullText[:idx])
				p.finishReasoning()
				p.state = stateText
				p.buffer.Reset()
				p.buffer.WriteString(fullText[idx+len(endTag):])
				fullText = p.buffer.String()
				continue
			}

			// Emit reasoning deltas
			if len(fullText) > 20 {
				emitLen := len(fullText) - 15
				p.emitReasoning(fullText[:emitLen])
				p.buffer.Reset()
				p.buffer.WriteString(fullText[emitLen:])
				return true
			}
			return true

		case stateToolCall:
			// Check for tool call end
			if idx := strings.Index(fullText, "</tool_call>"); idx != -1 {
				toolJSON := fullText[:idx]
				p.handleToolCall(toolJSON)
				p.state = stateText
				p.buffer.Reset()
				p.buffer.WriteString(fullText[idx+len("</tool_call>"):])
				fullText = p.buffer.String()
				continue
			}

			// For tool calls, we accumulate until the end because we need the full JSON for interactive components
			// However, we could stream it if we wanted to show progress.
			// Let's at least stream it as tool_input_delta.
			if len(fullText) > 20 {
				emitLen := len(fullText) - 15
				p.emitToolInput(fullText[:emitLen])
				p.buffer.Reset()
				p.buffer.WriteString(fullText[emitLen:])
				return true
			}

			return true
		}
	}
}

func (p *streamProcessor) Flush() {
	remaining := p.buffer.String()
	if remaining == "" {
		return
	}

	switch p.state {
	case stateText:
		p.emitText(remaining)
	case stateReasoning:
		p.emitReasoning(remaining)
		p.finishReasoning()
	case stateToolCall:
		p.handleToolCall(remaining)
	}
}

func (p *streamProcessor) emitText(text string) {
	if text == "" {
		return
	}
	if !p.textStarted {
		p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeTextStart, ID: "0"})
		p.textStarted = true
	}
	p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeTextDelta, ID: "0", Delta: text})
}

func (p *streamProcessor) emitReasoning(text string) {
	if text == "" {
		return
	}
	if !p.reasoningStarted {
		p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeReasoningStart, ID: "0"})
		p.reasoningStarted = true
	}
	p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeReasoningDelta, ID: "0", Delta: text})
}

func (p *streamProcessor) finishReasoning() {
	if p.reasoningStarted {
		p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeReasoningEnd, ID: "0"})
		p.reasoningStarted = false
	}
}

func (p *streamProcessor) emitToolInput(text string) {
	if text == "" {
		return
	}
	if !p.toolInputStarted {
		p.currentToolID = uuid.NewString()
		p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeToolInputStart, ID: p.currentToolID})
		p.toolInputStarted = true
	}
	p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeToolInputDelta, ID: p.currentToolID, Delta: text})
}

func (p *streamProcessor) handleToolCall(jsonStr string) {
	p.hasToolCalls = true
	// Try to parse the tool call
	calls := tools.ParseToolCalls("<tool_call>" + jsonStr + "</tool_call>")
	if len(calls) > 0 {
		tc := calls[0]
		if !p.toolInputStarted {
			p.currentToolID = tc.ID
			p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeToolInputStart, ID: p.currentToolID, ToolCallName: tc.Name})
		} else {
			// If we already started tool input, we might need to send a final delta if there's any left
			// but ParseToolCalls already got the whole thing.
		}

		p.yield(unillm.StreamPart{Type: unillm.StreamPartTypeToolInputEnd, ID: p.currentToolID})
		p.yield(unillm.StreamPart{
			Type:          unillm.StreamPartTypeToolCall,
			ID:            p.currentToolID,
			ToolCallName:  tc.Name,
			ToolCallInput: tc.Input,
		})
	}
	p.toolInputStarted = false
}

func (p *streamProcessor) HasToolCalls() bool {
	return p.hasToolCalls
}
