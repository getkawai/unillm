package google

import (
	"cmp"
	"errors"

	"github.com/getkawai/unillm"
	"google.golang.org/genai"
)

func toProviderErr(err error) error {
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	return &unillm.ProviderError{
		Message:      apiErr.Message,
		Title:        cmp.Or(unillm.ErrorTitleForStatusCode(apiErr.Code), "provider request failed"),
		Cause:        err,
		StatusCode:   apiErr.Code,
		ResponseBody: []byte(apiErr.Message),
	}
}
