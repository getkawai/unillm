package llamautil

import (
	"fmt"
	"sync"

	llamaapi "github.com/getkawai/llamalib/llama"
)

// RuntimeInstaller abstracts the minimal installer/runtime surface needed
// to bootstrap llama.cpp shared libraries.
type RuntimeInstaller interface {
	IsLlamaCppInstalled() bool
	InstallLlamaCpp() error
	GetLibraryPath() string
}

var runtimeState struct {
	mu    sync.Mutex
	ready bool
}

// EnsureLlamaRuntime performs one-time llama.cpp runtime initialization
// shared across providers.
func EnsureLlamaRuntime(installer RuntimeInstaller) error {
	runtimeState.mu.Lock()
	defer runtimeState.mu.Unlock()

	if runtimeState.ready {
		return nil
	}

	if !installer.IsLlamaCppInstalled() {
		if err := installer.InstallLlamaCpp(); err != nil {
			return fmt.Errorf("install llama.cpp runtime: %w", err)
		}
	}

	if err := llamaapi.Load(installer.GetLibraryPath()); err != nil {
		return fmt.Errorf("load llama runtime libraries: %w", err)
	}
	llamaapi.Init()

	runtimeState.ready = true
	return nil
}
