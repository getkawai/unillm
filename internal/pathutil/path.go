package pathutil

import (
	"os"
	"path/filepath"
	"strings"
)

// ExpandUserPath expands "~" and "~/" prefixes to the user home directory.
// If home resolution fails, the original path is returned unchanged.
func ExpandUserPath(path string) string {
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
