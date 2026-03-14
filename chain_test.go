package unillm

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

type chainStubModel struct {
	provider string
	model    string

	generate       func(context.Context, Call) (*Response, error)
	stream         func(context.Context, Call) (StreamResponse, error)
	generateObject func(context.Context, ObjectCall) (*ObjectResponse, error)
	streamObject   func(context.Context, ObjectCall) (ObjectStreamResponse, error)
}

func (m *chainStubModel) Generate(ctx context.Context, call Call) (*Response, error) {
	if m.generate != nil {
		return m.generate(ctx, call)
	}
	return nil, errors.New("generate not implemented")
}

func (m *chainStubModel) Stream(ctx context.Context, call Call) (StreamResponse, error) {
	if m.stream != nil {
		return m.stream(ctx, call)
	}
	return nil, errors.New("stream not implemented")
}

func (m *chainStubModel) GenerateObject(ctx context.Context, call ObjectCall) (*ObjectResponse, error) {
	if m.generateObject != nil {
		return m.generateObject(ctx, call)
	}
	return nil, errors.New("generate object not implemented")
}

func (m *chainStubModel) StreamObject(ctx context.Context, call ObjectCall) (ObjectStreamResponse, error) {
	if m.streamObject != nil {
		return m.streamObject(ctx, call)
	}
	return nil, errors.New("stream object not implemented")
}

func (m *chainStubModel) Provider() string { return m.provider }
func (m *chainStubModel) Model() string    { return m.model }

func TestChainGenerateFallsBackToNextModel(t *testing.T) {
	t.Parallel()

	firstCalls := 0
	secondCalls := 0

	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		generate: func(context.Context, Call) (*Response, error) {
			firstCalls++
			return nil, errors.New("upstream timeout")
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		generate: func(context.Context, Call) (*Response, error) {
			secondCalls++
			return &Response{
				Content: []Content{TextContent{Text: "ok"}},
			}, nil
		},
	}

	chain, err := NewChain([]LanguageModel{first, second})
	require.NoError(t, err)

	resp, err := chain.Generate(context.Background(), Call{})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Equal(t, "ok", resp.Content.Text())
	require.Equal(t, 1, firstCalls)
	require.Equal(t, 1, secondCalls)
}

func TestChainGenerateStopsWhenContextEnded(t *testing.T) {
	t.Parallel()

	firstCalls := 0
	secondCalls := 0

	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		generate: func(context.Context, Call) (*Response, error) {
			firstCalls++
			return nil, errors.New("failed")
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		generate: func(context.Context, Call) (*Response, error) {
			secondCalls++
			return &Response{}, nil
		},
	}

	chain, err := NewChain([]LanguageModel{first, second})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = chain.Generate(ctx, Call{})
	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
	require.ErrorContains(t, err, "context ended")
	require.Equal(t, 1, firstCalls)
	require.Equal(t, 0, secondCalls)
}

func TestChainCircuitBreakerNotPoisonedByCanceledContext(t *testing.T) {
	t.Parallel()

	firstCalls := 0
	secondCalls := 0

	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		generate: func(ctx context.Context, _ Call) (*Response, error) {
			firstCalls++
			if ctx.Err() != nil {
				return nil, errors.New("failed while context ended")
			}
			return &Response{
				Content: []Content{TextContent{Text: "ok"}},
			}, nil
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		generate: func(context.Context, Call) (*Response, error) {
			secondCalls++
			return &Response{}, nil
		},
	}

	chain, err := NewChain([]LanguageModel{first, second}, WithCircuitBreaker(1, 0))
	require.NoError(t, err)

	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = chain.Generate(cancelledCtx, Call{})
	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
	require.ErrorContains(t, err, "context ended")

	resp, err := chain.Generate(context.Background(), Call{})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Equal(t, "ok", resp.Content.Text())
	require.Equal(t, 2, firstCalls)
	require.Equal(t, 0, secondCalls)
}

func TestChainAllCircuitsOpenReturnsUsefulError(t *testing.T) {
	t.Parallel()

	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		generate: func(context.Context, Call) (*Response, error) {
			return nil, errors.New("boom-1")
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		generate: func(context.Context, Call) (*Response, error) {
			return nil, errors.New("boom-2")
		},
	}

	chain, err := NewChain([]LanguageModel{first, second}, WithCircuitBreaker(1, 0))
	require.NoError(t, err)

	_, err = chain.Generate(context.Background(), Call{})
	require.Error(t, err)
	require.ErrorContains(t, err, "all models failed")

	_, err = chain.Generate(context.Background(), Call{})
	require.Error(t, err)
	require.ErrorContains(t, err, "no available models (all circuits open)")
}

func TestChainStreamAndStreamObjectAllCircuitsOpen(t *testing.T) {
	t.Parallel()

	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		generate: func(context.Context, Call) (*Response, error) {
			return nil, errors.New("boom-1")
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		generate: func(context.Context, Call) (*Response, error) {
			return nil, errors.New("boom-2")
		},
	}

	chain, err := NewChain([]LanguageModel{first, second}, WithCircuitBreaker(1, 0))
	require.NoError(t, err)

	// Open circuits first.
	_, err = chain.Generate(context.Background(), Call{})
	require.Error(t, err)

	stream, err := chain.Stream(context.Background(), Call{})
	require.NoError(t, err)

	var streamErr error
	for part := range stream {
		if part.Error != nil {
			streamErr = part.Error
		}
	}
	require.Error(t, streamErr)
	require.ErrorContains(t, streamErr, "no available models (all circuits open)")

	objectStream, err := chain.StreamObject(context.Background(), ObjectCall{})
	require.NoError(t, err)

	var objectStreamErr error
	for part := range objectStream {
		if part.Error != nil {
			objectStreamErr = part.Error
		}
	}
	require.Error(t, objectStreamErr)
	require.ErrorContains(t, objectStreamErr, "no available models (all circuits open)")
}

func TestChainStreamObjectStopsWhenContextEnded(t *testing.T) {
	t.Parallel()

	firstCalls := 0
	secondCalls := 0

	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		streamObject: func(context.Context, ObjectCall) (ObjectStreamResponse, error) {
			firstCalls++
			return nil, errors.New("setup failed")
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		streamObject: func(context.Context, ObjectCall) (ObjectStreamResponse, error) {
			secondCalls++
			return nil, errors.New("should not be called")
		},
	}

	chain, err := NewChain([]LanguageModel{first, second})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stream, err := chain.StreamObject(ctx, ObjectCall{})
	require.NoError(t, err)

	var streamErr error
	for part := range stream {
		if part.Error != nil {
			streamErr = part.Error
		}
	}

	require.Error(t, streamErr)
	require.ErrorIs(t, streamErr, context.Canceled)
	require.ErrorContains(t, streamErr, "context ended")
	require.Equal(t, 1, firstCalls)
	require.Equal(t, 0, secondCalls)
}

func TestChainStreamObjectStopsFallbackOnMidStreamContextEnd(t *testing.T) {
	t.Parallel()

	firstCalls := 0
	secondCalls := 0

	ctx, cancel := context.WithCancel(context.Background())

	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		streamObject: func(context.Context, ObjectCall) (ObjectStreamResponse, error) {
			firstCalls++
			return func(yield func(ObjectStreamPart) bool) {
				cancel()
				yield(ObjectStreamPart{
					Type:  ObjectStreamPartTypeError,
					Error: context.Canceled,
				})
			}, nil
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		streamObject: func(context.Context, ObjectCall) (ObjectStreamResponse, error) {
			secondCalls++
			return nil, errors.New("should not be called")
		},
	}

	chain, err := NewChain([]LanguageModel{first, second})
	require.NoError(t, err)

	stream, err := chain.StreamObject(ctx, ObjectCall{})
	require.NoError(t, err)

	var streamErr error
	for part := range stream {
		if part.Error != nil {
			streamErr = part.Error
		}
	}

	require.Error(t, streamErr)
	require.ErrorIs(t, streamErr, context.Canceled)
	require.ErrorContains(t, streamErr, "context ended")
	require.Equal(t, 1, firstCalls)
	require.Equal(t, 0, secondCalls)
}

// Benchmark tests for Chain functionality

func BenchmarkChain_Generate_SingleModel(b *testing.B) {
	model := &chainStubModel{
		provider: "test",
		model:    "test-model",
		generate: func(context.Context, Call) (*Response, error) {
			return &Response{
				Content: []Content{TextContent{Text: "ok"}},
				Usage:   Usage{InputTokens: 5, OutputTokens: 10, TotalTokens: 15},
			}, nil
		},
	}
	chain, _ := NewChain([]LanguageModel{model})
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = chain.Generate(ctx, Call{})
	}
}

func BenchmarkChain_Generate_Fallback(b *testing.B) {
	firstCalls := 0
	first := &chainStubModel{
		provider: "p1",
		model:    "m1",
		generate: func(context.Context, Call) (*Response, error) {
			firstCalls++
			return nil, errors.New("upstream timeout")
		},
	}
	second := &chainStubModel{
		provider: "p2",
		model:    "m2",
		generate: func(context.Context, Call) (*Response, error) {
			return &Response{
				Content: []Content{TextContent{Text: "ok"}},
				Usage:   Usage{InputTokens: 5, OutputTokens: 10, TotalTokens: 15},
			}, nil
		},
	}
	chain, _ := NewChain([]LanguageModel{first, second})
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		firstCalls = 0
		_, _ = chain.Generate(ctx, Call{})
	}
}

func BenchmarkChain_Stream_SingleModel(b *testing.B) {
	model := &chainStubModel{
		provider: "test",
		model:    "test-model",
		stream: func(context.Context, Call) (StreamResponse, error) {
			return func(yield func(StreamPart) bool) {
				yield(StreamPart{Type: StreamPartTypeTextStart, ID: "1"})
				yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "1", Delta: "Hello"})
				yield(StreamPart{Type: StreamPartTypeTextEnd, ID: "1"})
				yield(StreamPart{Type: StreamPartTypeFinish, Usage: Usage{InputTokens: 5, OutputTokens: 10, TotalTokens: 15}})
			}, nil
		},
	}
	chain, _ := NewChain([]LanguageModel{model})
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream, _ := chain.Stream(ctx, Call{})
		for range stream {
		}
	}
}
