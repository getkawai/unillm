package llama

import (
	"encoding/json"

	"github.com/getkawai/unillm"
)

const (
	// TypeProviderOptions is the type identifier for llama provider options.
	TypeProviderOptions = "llama.options"
)

func init() {
	unillm.RegisterProviderType(TypeProviderOptions, func(data []byte) (unillm.ProviderOptionsData, error) {
		var opts ProviderOptions
		if err := json.Unmarshal(data, &opts); err != nil {
			return nil, err
		}
		return &opts, nil
	})
}

// ProviderOptions contains llama-specific provider options.
type ProviderOptions struct {
	// Temperature controls randomness in sampling (0.0 = deterministic, 1.0 = very random)
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP controls nucleus sampling threshold
	TopP *float64 `json:"top_p,omitempty"`

	// TopK limits sampling to top K tokens
	TopK *int64 `json:"top_k,omitempty"`

	// RepetitionPenalty penalizes repeated tokens
	RepetitionPenalty *float64 `json:"repetition_penalty,omitempty"`

	// Seed for reproducible sampling (-1 for random)
	Seed *int64 `json:"seed,omitempty"`

	// UseReasoningMode enables thinking/reasoning mode for compatible models
	UseReasoningMode bool `json:"use_reasoning_mode,omitempty"`
}

// Options implements unillm.ProviderOptionsData.
func (*ProviderOptions) Options() {}

// MarshalJSON implements json.Marshaler.
func (p ProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderOptions
	return unillm.MarshalProviderType(TypeProviderOptions, plain(p))
}

// UnmarshalJSON implements json.Unmarshaler.
func (p *ProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderOptions
	var pp plain
	if err := unillm.UnmarshalProviderType(data, &pp); err != nil {
		return err
	}
	*p = ProviderOptions(pp)
	return nil
}

// ProviderMetadata contains llama-specific response metadata.
type ProviderMetadata struct {
	// ModelPath is the path to the loaded model file
	ModelPath string `json:"model_path,omitempty"`

	// ContextSize is the context window size used
	ContextSize int64 `json:"context_size,omitempty"`

	// BatchSize is the batch size used for processing
	BatchSize int64 `json:"batch_size,omitempty"`
}
