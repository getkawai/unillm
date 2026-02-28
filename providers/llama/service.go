package llama

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/getkawai/llamalib"
	llamaapi "github.com/getkawai/llamalib/llama"
)

var runtimeState struct {
	mu    sync.Mutex
	ready bool
}

// Service wraps low-level llama.cpp runtime/model resources.
type Service struct {
	mu sync.Mutex

	installer *llamalib.LlamaCppInstaller

	chatModel       llamaapi.Model
	chatContext     llamaapi.Context
	chatVocab       llamaapi.Vocab
	chatSampler     llamaapi.Sampler
	loadedChatModel string
}

// NewService creates a new runtime service.
func NewService() *Service {
	return &Service{
		installer: llamalib.NewLlamaCppInstaller(),
	}
}

func (s *Service) ensureRuntime() error {
	runtimeState.mu.Lock()
	defer runtimeState.mu.Unlock()

	if runtimeState.ready {
		return nil
	}

	if !s.installer.IsLlamaCppInstalled() {
		if err := s.installer.InstallLlamaCpp(); err != nil {
			return fmt.Errorf("install llama.cpp runtime: %w", err)
		}
	}

	if err := llamaapi.Load(s.installer.GetLibraryPath()); err != nil {
		return fmt.Errorf("load llama runtime libraries: %w", err)
	}
	llamaapi.Init()

	runtimeState.ready = true
	return nil
}

// WaitForInitialization ensures llama runtime is available.
func (s *Service) WaitForInitialization(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	return s.ensureRuntime()
}

// IsChatModelLoaded reports whether chat resources are available.
func (s *Service) IsChatModelLoaded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.chatModel != 0 && s.chatContext != 0 && s.chatVocab != 0
}

// GetLoadedChatModel returns the currently loaded model path.
func (s *Service) GetLoadedChatModel() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loadedChatModel
}

func (s *Service) resolveChatModelPath(modelPath string) (string, error) {
	if modelPath != "" {
		return expandUserPath(modelPath), nil
	}

	available, err := s.installer.GetAvailableTextModels()
	if err != nil {
		return "", fmt.Errorf("list available text models: %w", err)
	}

	if len(available) == 0 {
		if err := s.installer.AutoDownloadRecommendedTextModel(); err != nil {
			return "", fmt.Errorf("auto-download text model: %w", err)
		}
		available, err = s.installer.GetAvailableTextModels()
		if err != nil {
			return "", fmt.Errorf("list text models after download: %w", err)
		}
	}

	if len(available) == 0 {
		return "", fmt.Errorf("no chat model available")
	}

	sort.Strings(available)
	return available[0], nil
}

func (s *Service) releaseChatLocked() {
	if s.chatSampler != 0 {
		llamaapi.SamplerFree(s.chatSampler)
		s.chatSampler = 0
	}
	if s.chatContext != 0 {
		_ = llamaapi.Free(s.chatContext)
		s.chatContext = 0
	}
	if s.chatModel != 0 {
		_ = llamaapi.ModelFree(s.chatModel)
		s.chatModel = 0
	}

	s.chatVocab = 0
	s.loadedChatModel = ""
}

// LoadChatModel loads a chat model and initializes generation resources.
func (s *Service) LoadChatModel(modelPath string) error {
	if err := s.ensureRuntime(); err != nil {
		return err
	}

	resolvedPath, err := s.resolveChatModelPath(modelPath)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.loadedChatModel == resolvedPath && s.chatModel != 0 && s.chatContext != 0 {
		return nil
	}

	s.releaseChatLocked()

	modelParams := llamaapi.ModelDefaultParams()
	chatModel, err := llamaapi.ModelLoadFromFile(resolvedPath, modelParams)
	if err != nil || chatModel == 0 {
		return fmt.Errorf("load model %q: %w", resolvedPath, err)
	}

	ctxParams := llamaapi.ContextDefaultParams()
	chatCtx, err := llamaapi.InitFromModel(chatModel, ctxParams)
	if err != nil || chatCtx == 0 {
		_ = llamaapi.ModelFree(chatModel)
		return fmt.Errorf("create context for %q: %w", resolvedPath, err)
	}

	samplerParams := llamaapi.DefaultSamplerParams()
	chatSampler := llamaapi.NewSampler(chatModel, llamaapi.DefaultSamplers, samplerParams)
	if chatSampler == 0 {
		_ = llamaapi.Free(chatCtx)
		_ = llamaapi.ModelFree(chatModel)
		return fmt.Errorf("create sampler for %q", resolvedPath)
	}

	s.chatModel = chatModel
	s.chatContext = chatCtx
	s.chatVocab = llamaapi.ModelGetVocab(chatModel)
	s.chatSampler = chatSampler
	s.loadedChatModel = resolvedPath
	return nil
}

// GetChatModel returns the loaded model handle.
func (s *Service) GetChatModel() llamaapi.Model {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.chatModel
}

// GetChatVocab returns the loaded vocab handle.
func (s *Service) GetChatVocab() llamaapi.Vocab {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.chatVocab
}

// WithChatLock executes fn while holding the chat resource lock.
func (s *Service) WithChatLock(fn func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fn()
}

// GetChatResourcesUnsafe returns chat resources without additional synchronization.
// Caller must hold lock via WithChatLock.
func (s *Service) GetChatResourcesUnsafe() (llamaapi.Model, llamaapi.Context, llamaapi.Vocab, llamaapi.Sampler) {
	return s.chatModel, s.chatContext, s.chatVocab, s.chatSampler
}

// Cleanup releases loaded chat resources for this service.
func (s *Service) Cleanup() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseChatLocked()
}

func expandUserPath(path string) string {
	if path == "~" {
		if home, err := os.UserHomeDir(); err == nil {
			return home
		}
		return path
	}
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}
