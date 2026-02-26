package pooled

import (
	"context"
	"fmt"
	"log"

	"github.com/kawai-network/veridium/pkg/cliproxy/sdk/cliproxy/auth"
	"github.com/kawai-network/veridium/pkg/cliproxy/sdk/cliproxy/executor"
	"github.com/getkawai/unillm"
)

// Identifier implements auth.ProviderExecutor.
func (e *PooledExecutor) Identifier() string {
	return e.providerName
}

// Execute implements auth.ProviderExecutor.
func (e *PooledExecutor) Execute(ctx context.Context, authEntry *auth.Auth, req executor.Request, opts executor.Options) (executor.Response, error) {
	// Extract API key from auth metadata
	apiKey, ok := authEntry.Metadata["api_key"].(string)
	if !ok {
		return executor.Response{}, fmt.Errorf("api_key not found in auth metadata")
	}

	// Create client with this API key
	client, err := e.createClient(apiKey)
	if err != nil {
		return executor.Response{}, fmt.Errorf("failed to create client: %w", err)
	}

	// Convert executor.Request to fantasy.Call
	call := convertRequestToCall(req)

	// Execute the call
	resp, err := client.Generate(ctx, call)
	if err != nil {
		return executor.Response{}, err
	}

	// Debug: log raw fantasy response with content types
	contentTypes := make([]string, len(resp.Content))
	toolCallCount := 0
	for i, c := range resp.Content {
		contentTypes[i] = string(c.GetType())
		if c.GetType() == fantasy.ContentTypeToolCall {
			toolCallCount++
		}
	}
	log.Printf("[PooledExecutor:%s] Raw fantasy.Response: Content=%d parts %v (tool_calls=%d), Text=%q, Usage=%+v",
		e.providerName, len(resp.Content), contentTypes, toolCallCount, resp.Content.Text(), resp.Usage)

	// Convert fantasy.Response to executor.Response
	execResp := convertFantasyToResponse(resp)

	// Debug: log converted response
	if content, ok := execResp.Metadata["content"].(string); ok {
		log.Printf("[PooledExecutor:%s] Converted content length: %d, content: %q",
			e.providerName, len(content), content)
		if content == "" {
			log.Printf("⚠️  Warning: Empty response from provider %s", e.providerName)
		}
	}

	// Debug: log tool calls
	if toolCalls, ok := execResp.Metadata["tool_calls"]; ok {
		log.Printf("[PooledExecutor:%s] Tool calls detected: %+v", e.providerName, toolCalls)
	} else {
		log.Printf("[PooledExecutor:%s] No tool calls in metadata", e.providerName)
	}

	return execResp, nil
}

// ExecuteStream implements auth.ProviderExecutor.
func (e *PooledExecutor) ExecuteStream(ctx context.Context, authEntry *auth.Auth, req executor.Request, opts executor.Options) (<-chan executor.StreamChunk, error) {
	// Extract API key from auth metadata
	apiKey, ok := authEntry.Metadata["api_key"].(string)
	if !ok {
		return nil, fmt.Errorf("api_key not found in auth metadata")
	}

	// Create client with this API key
	client, err := e.createClient(apiKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create client: %w", err)
	}

	// Convert executor.Request to fantasy.Call
	call := convertRequestToCall(req)

	// Execute the stream
	fantasyStream, err := client.Stream(ctx, call)
	if err != nil {
		return nil, err
	}

	// Convert fantasy.StreamResponse to executor stream
	out := make(chan executor.StreamChunk)
	go func() {
		defer close(out)
		for part := range fantasyStream {
			chunk := convertFantasyPartToChunk(part)
			out <- chunk
		}
	}()

	return out, nil
}

// Refresh implements auth.ProviderExecutor (not needed for API keys).
func (e *PooledExecutor) Refresh(ctx context.Context, authEntry *auth.Auth) (*auth.Auth, error) {
	// API keys don't need refresh
	return authEntry, nil
}

// CountTokens implements auth.ProviderExecutor (optional).
func (e *PooledExecutor) CountTokens(ctx context.Context, authEntry *auth.Auth, req executor.Request, opts executor.Options) (executor.Response, error) {
	// Not implemented for now
	return executor.Response{}, fmt.Errorf("CountTokens not implemented")
}
