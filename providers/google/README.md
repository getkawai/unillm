# Google Provider for Fantasy AI SDK

Provider ini mengimplementasikan interface Fantasy AI SDK untuk model bahasa Google (Gemini dan Vertex AI).

## Fitur

- ✅ Dukungan Gemini API dan Vertex AI
- ✅ Text generation dan streaming
- ✅ Tool calling (function calling)
- ✅ Object generation dengan JSON mode
- ✅ Reasoning/thinking capabilities
- ✅ Multi-modal support (text dan image)
- ✅ Context caching
- ✅ Safety settings

## Instalasi

```bash
go get google.golang.org/genai
go get cloud.google.com/go/auth
```

## Penggunaan

### 1. Menggunakan Gemini API

```go
import (
    "context"
    "github.com/getkawai/unillm"
    "github.com/getkawai/unillm/providers/google"
)

// Buat provider dengan Gemini API key
provider, err := google.New(
    google.WithGeminiAPIKey("YOUR_API_KEY"),
)
if err != nil {
    log.Fatal(err)
}

// Dapatkan language model
ctx := context.Background()
model, err := provider.LanguageModel(ctx, "gemini-1.5-flash")
if err != nil {
    log.Fatal(err)
}

// Generate text
response, err := model.Generate(ctx, fantasy.Call{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "Hello, how are you?"},
        }},
    },
})
```

### 2. Menggunakan Vertex AI

```go
provider, err := google.New(
    google.WithVertex("your-project-id", "us-central1"),
)
if err != nil {
    log.Fatal(err)
}

model, err := provider.LanguageModel(ctx, "gemini-1.5-pro")
```

### 3. Streaming Response

```go
stream, err := model.Stream(ctx, fantasy.Call{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "Tell me a story"},
        }},
    },
})
if err != nil {
    log.Fatal(err)
}

for part := range stream {
    switch part.Type {
    case fantasy.StreamPartTypeTextDelta:
        fmt.Print(part.Delta)
    case fantasy.StreamPartTypeError:
        log.Printf("Error: %v", part.Error)
    }
}
```

### 4. Tool Calling

```go
tools := []fantasy.Tool{
    fantasy.FunctionTool{
        Name:        "get_weather",
        Description: "Get weather information",
        InputSchema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "location": map[string]any{
                    "type":        "string",
                    "description": "City name",
                },
            },
            "required": []string{"location"},
        },
    },
}

response, err := model.Generate(ctx, fantasy.Call{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "What's the weather in Jakarta?"},
        }},
    },
    Tools: tools,
})
```

### 5. Object Generation (Structured Output)

```go
type Person struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

objResponse, err := model.GenerateObject(ctx, fantasy.ObjectCall{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "Extract person info: John is 30 years old"},
        }},
    },
    Schema: schema.FromStruct(Person{}),
})

person := objResponse.Object.(*Person)
fmt.Printf("Name: %s, Age: %d\n", person.Name, person.Age)
```

### 6. Thinking/Reasoning Mode

```go
providerOpts := &google.ProviderOptions{
    ThinkingConfig: &google.ThinkingConfig{
        IncludeThoughts: fantasy.Opt(true),
        ThinkingBudget:  fantasy.Opt(int64(1024)),
    },
}

response, err := model.Generate(ctx, fantasy.Call{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "Solve this complex problem..."},
        }},
    },
    ProviderOptions: fantasy.ProviderOptions{
        google.Name: providerOpts,
    },
})

// Akses reasoning content
for _, content := range response.Content {
    if content.GetType() == fantasy.ContentTypeReasoning {
        reasoning := content.(fantasy.ReasoningContent)
        fmt.Println("Reasoning:", reasoning.Text)
    }
}
```

### 7. Multi-modal (Image + Text)

```go
imageData, _ := os.ReadFile("image.jpg")

response, err := model.Generate(ctx, fantasy.Call{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "What's in this image?"},
            fantasy.FilePart{
                Data:      imageData,
                MediaType: "image/jpeg",
            },
        }},
    },
})
```

### 8. Context Caching

```go
providerOpts := &google.ProviderOptions{
    CachedContent: "cachedContents/your-cached-content-id",
}

response, err := model.Generate(ctx, fantasy.Call{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "Continue from cached context..."},
        }},
    },
    ProviderOptions: fantasy.ProviderOptions{
        google.Name: providerOpts,
    },
})
```

### 9. Safety Settings

```go
providerOpts := &google.ProviderOptions{
    SafetySettings: []google.SafetySetting{
        {
            Category:  "HARM_CATEGORY_HATE_SPEECH",
            Threshold: "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            Category:  "HARM_CATEGORY_DANGEROUS_CONTENT",
            Threshold: "BLOCK_MEDIUM_AND_ABOVE",
        },
    },
}

response, err := model.Generate(ctx, fantasy.Call{
    Prompt: fantasy.Prompt{
        {Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
            fantasy.TextPart{Text: "Your prompt here"},
        }},
    },
    ProviderOptions: fantasy.ProviderOptions{
        google.Name: providerOpts,
    },
})
```

## Opsi Konfigurasi

### Provider Options

- `WithGeminiAPIKey(apiKey string)` - Set Gemini API key
- `WithVertex(project, location string)` - Konfigurasi Vertex AI
- `WithBaseURL(baseURL string)` - Set custom base URL
- `WithHeaders(headers map[string]string)` - Set custom headers
- `WithHTTPClient(client *http.Client)` - Set custom HTTP client
- `WithSkipAuth(skipAuth bool)` - Skip authentication (untuk testing)
- `WithName(name string)` - Set provider name
- `WithToolCallIDFunc(f ToolCallIDFunc)` - Set custom tool call ID generator
- `WithObjectMode(om fantasy.ObjectMode)` - Set object generation mode

### Call Options

- `MaxOutputTokens` - Maximum tokens untuk output
- `Temperature` - Kontrol randomness (0.0 - 2.0)
- `TopP` - Nucleus sampling parameter
- `TopK` - Top-k sampling parameter
- `FrequencyPenalty` - Penalti untuk token yang sering muncul
- `PresencePenalty` - Penalti untuk token yang sudah ada

## Model yang Didukung

### Gemini API
- `gemini-1.5-flash`
- `gemini-1.5-flash-8b`
- `gemini-1.5-pro`
- `gemini-2.0-flash-exp`
- `gemini-exp-1206`

### Vertex AI
- Semua model Gemini yang tersedia di Vertex AI
- Model Gemma (dengan beberapa limitasi)

## Error Handling

Provider ini mengkonversi error dari Google API menjadi `fantasy.ProviderError`:

```go
response, err := model.Generate(ctx, call)
if err != nil {
    if providerErr, ok := err.(*fantasy.ProviderError); ok {
        fmt.Printf("Status: %d\n", providerErr.StatusCode)
        fmt.Printf("Message: %s\n", providerErr.Message)
    }
}
```

## Limitasi

1. Model Gemma tidak mendukung system instructions secara native - akan dikonversi ke user message
2. Beberapa fitur hanya tersedia di Vertex AI (seperti thinking mode)
3. Context caching memerlukan setup terpisah di Google Cloud

## Lisensi

Lihat LICENSE file di root repository.
