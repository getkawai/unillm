package llamaembed

import "context"

// Embedder is a simple interface for text embedding generation.
// This replaces the Eino embedding.Embedder interface with a simpler, custom interface.
type Embedder interface {
	// Embed generates embeddings for the given texts.
	// Returns a slice of embeddings, one for each input text.
	// Each embedding is a []float32 vector.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the embedding dimension size.
	Dimensions() int

	// Close releases any resources held by the embedder.
	Close() error
}
