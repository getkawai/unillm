package llamaembed

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/getkawai/llamalib/llama"
)

const defaultContextSize = 2048

// LlamaConfig holds configuration for Llama embedding
type LlamaConfig struct {
	// ModelPath specifies the path to the embedding model GGUF file (required)
	ModelPath string

	// ContextSize specifies the maximum context size for tokenization (default: 2048)
	ContextSize int
}

// LlamaEmbedder implements Embedder interface using llama.cpp via pkg/yzma/llama
type LlamaEmbedder struct {
	config     *LlamaConfig
	model      llama.Model
	context    llama.Context
	vocab      llama.Vocab
	dimensions int32
	mutex      sync.Mutex
}

// NewLlamaEmbedder creates a new Llama embedder.
// Assumes llama.cpp library is already loaded via llama.Load() and llama.Init().
func NewLlamaEmbedder(config *LlamaConfig) (*LlamaEmbedder, error) {
	if config == nil {
		return nil, errors.New("config must not be nil")
	}

	if config.ModelPath == "" {
		return nil, errors.New("model_path is required")
	}

	if config.ContextSize <= 0 {
		config.ContextSize = defaultContextSize
	}

	// Expand home directory in model path
	modelPath := expandPath(config.ModelPath)

	// Validate model file exists
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model file not found: %s (%w)", modelPath, err)
	}

	// Load model
	model, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil || model == 0 {
		return nil, fmt.Errorf("failed to load embedding model from: %s (%w)", modelPath, err)
	}

	// Get embedding dimensions
	dimensions := llama.ModelNEmbd(model)
	log.Printf("Embedding model loaded: %s (dimensions: %d)", filepath.Base(modelPath), dimensions)

	// Create context for embeddings
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(config.ContextSize)
	ctxParams.NBatch = uint32(config.ContextSize)
	ctxParams.NUbatch = uint32(config.ContextSize)
	ctxParams.Embeddings = 1

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil || ctx == 0 {
		llama.ModelFree(model)
		return nil, fmt.Errorf("failed to create context for embeddings: %w", err)
	}

	// Get vocabulary
	vocab := llama.ModelGetVocab(model)

	return &LlamaEmbedder{
		config:     config,
		model:      model,
		context:    ctx,
		vocab:      vocab,
		dimensions: dimensions,
	}, nil
}

// Embed generates embeddings for the given texts.
func (e *LlamaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, errors.New("no texts provided")
	}

	result := make([][]float32, len(texts))
	for i, text := range texts {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		emb, err := e.generateEmbedding(text)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for text %d: %w", i, err)
		}
		result[i] = emb
	}

	return result, nil
}

// Dimensions returns the embedding dimension size.
func (e *LlamaEmbedder) Dimensions() int {
	return int(e.dimensions)
}

// Close releases resources held by the embedder.
func (e *LlamaEmbedder) Close() error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if e.context != 0 {
		llama.Free(e.context)
		e.context = 0
		log.Println("Embedding context freed")
	}

	if e.model != 0 {
		llama.ModelFree(e.model)
		e.model = 0
		log.Println("Embedding model freed")
	}

	return nil
}

// generateEmbedding generates a single embedding.
func (e *LlamaEmbedder) generateEmbedding(text string) ([]float32, error) {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if text == "" {
		return nil, errors.New("empty text provided")
	}

	// Tokenize text
	tokens := llama.Tokenize(e.vocab, text, true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("failed to tokenize text: %q", text)
	}

	// Truncate tokens if they exceed context size
	maxTokens := e.config.ContextSize
	if len(tokens) > maxTokens {
		log.Printf("Text has %d tokens, truncating to %d tokens", len(tokens), maxTokens)
		tokens = tokens[:maxTokens]
	}

	// Get model's native context size
	nCtxTrain := int(llama.ModelNCtxTrain(e.model))
	if nCtxTrain <= 0 {
		nCtxTrain = defaultContextSize
	}

	// Strict truncation to context size
	if len(tokens) > nCtxTrain {
		log.Printf("Input text too long (%d tokens), truncating to model context size (%d)", len(tokens), nCtxTrain)
		tokens = tokens[:nCtxTrain]
	}

	// Create batch
	batch := llama.BatchGetOne(tokens)

	// Decode tokens
	errCode, err := llama.Decode(e.context, batch)
	if err != nil || errCode != 0 {
		return nil, fmt.Errorf("failed to decode tokens: error code %d (%w)", errCode, err)
	}

	// Get embeddings
	vec, err := llama.GetEmbeddingsSeq(e.context, 0, e.dimensions)
	if err != nil {
		return nil, fmt.Errorf("failed to get embeddings from context: %w", err)
	}

	// Normalize embeddings using L2 norm
	return normalizeVector(vec), nil
}

// expandPath expands ~ to home directory in paths.
func expandPath(path string) string {
	if len(path) > 0 && path[0] == '~' {
		homeDir, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(homeDir, path[1:])
		}
	}
	return path
}

// normalizeVector normalizes a vector using L2 norm.
func normalizeVector(v []float32) []float32 {
	var sum float64
	for _, val := range v {
		sum += float64(val) * float64(val)
	}
	if sum == 0 {
		return v
	}

	norm := float32(1.0 / math.Sqrt(sum))
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = val * norm
	}
	return result
}

// Ensure LlamaEmbedder implements Embedder interface
var _ Embedder = (*LlamaEmbedder)(nil)
