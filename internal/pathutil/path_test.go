package pathutil

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestExpandUserPath(t *testing.T) {
	t.Parallel()

	home, err := os.UserHomeDir()
	require.NoError(t, err)

	require.Equal(t, home, ExpandUserPath("~"))
	require.Equal(t, filepath.Join(home, "models", "x.gguf"), ExpandUserPath("~/models/x.gguf"))
	require.Equal(t, "/tmp/model.gguf", ExpandUserPath("/tmp/model.gguf"))
	require.Equal(t, "relative/model.gguf", ExpandUserPath("relative/model.gguf"))
}
