# Pooled Provider - CLIProxyAPI Integration

This package integrates CLIProxyAPI's robust fallback mechanism into Veridium's fantasy provider system.

## Features

### ✅ **3-Level Fallback Hierarchy**

1. **Account Failover** - Rotates through multiple API keys for the same provider
2. **Provider Failover** - Falls back to different providers (OpenRouter → Pollinations → ZAI → Local)
3. **Global Retry** - Retries the entire chain with exponential backoff

### ✅ **Smart Error Handling**

| HTTP Status | Action | Duration | Reason |
|-------------|--------|----------|--------|
| **401** | Suspend Account | 30 min | Invalid API Key |
| **402/403** | Suspend Account | 30 min | Payment required |
| **404** | Suspend Model | 12 hours | Model not available |
| **429** | Cooldown + Backoff | Dynamic | Quota exceeded |
| **5xx** | Transient | 1 min | Server error |

### ✅ **Round-Robin Load Balancing**

Distributes requests evenly across all API keys to maximize quota utilization.

### ✅ **State Persistence**

Tracks cooldown state per account and per model (in-memory for desktop app).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PooledProvider (fantasy.LanguageModel)                 │
│  - Wraps multiple API keys                              │
│  - Implements fantasy interface                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  auth.Manager (CLIProxyAPI)                             │
│  - 3-level fallback logic                               │
│  - Round-robin selection                                │
│  - Smart error handling                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  PooledExecutor (auth.ProviderExecutor)                 │
│  - Converts between fantasy and executor types          │
│  - Creates clients with selected API key                │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```go
import (
    "context"
    "github.com/getkawai/unillm/providers/pooled"
    "github.com/getkawai/unillm/providers/openrouter"
)

// Create pooled provider with multiple API keys
pooledProvider, err := pooled.New(pooled.Config{
    ProviderName: "openrouter",
    BaseURL:      "https://openrouter.ai/api/v1",
    ModelName:    "auto",
    APIKeys:      []string{"key1", "key2", "key3"},
    CreateClient: func(apiKey string) (fantasy.LanguageModel, error) {
        provider, err := openrouter.New(
            openrouter.WithAPIKey(apiKey),
        )
        if err != nil {
            return nil, err
        }
        return provider.LanguageModel(context.Background(), "")
    },
})

// Use like any other fantasy.LanguageModel
response, err := pooledProvider.Generate(ctx, fantasy.Call{
    Messages: []fantasy.Message{
        {Role: "user", Content: "Hello!"},
    },
})
```

### Integration with Veridium Context

```go
// In internal/app/context.go

func (ctx *Context) InitLanguageModels() {
    // Use the new V2 version with pooling
    ctx.InitLanguageModelsV2()
}
```

This will automatically use the pooled providers with all configured API keys.

## How It Works

### 1. Request Flow

```
User Request
    ↓
fantasy.Call → executor.Request (conversion)
    ↓
auth.Manager.Execute() [Level 3: Global Retry]
    ↓
executeProvidersOnce() [Level 2: Provider Rotation]
    ↓
executeWithProvider() [Level 1: Account Failover]
    ↓
pickNext() → RoundRobinSelector.Pick()
    ↓
PooledExecutor.Execute()
    ↓
Create client with selected API key
    ↓
Execute fantasy.Call
    ↓
executor.Response → fantasy.Response (conversion)
    ↓
Return to user
```

### 2. Error Handling Flow

```
API Call Fails
    ↓
MarkResult() - Smart error handling
    ↓
Update Auth State:
  - Set NextRetryAfter
  - Set Quota.Exceeded
  - Calculate backoff level
    ↓
Persist state to store
    ↓
Try next account (Level 1)
    ↓
If all accounts fail → Try next provider (Level 2)
    ↓
If all providers fail → Wait & retry (Level 3)
```

### 3. Selection Flow

```
pickNext() called
    ↓
Filter candidates:
  - Same provider
  - Not disabled
  - Not already tried
  - Supports model
    ↓
RoundRobinSelector.Pick()
    ↓
getAvailableAuths()
    ↓
Filter by cooldown:
  - Check NextRetryAfter
  - Check Quota.NextRecoverAt
  - Skip if in cooldown
    ↓
Round-robin selection:
  cursor = (cursor + 1) % len(available)
    ↓
Return selected account
```

## Configuration

### Retry Settings

```go
// Configure global retry behavior
manager.SetRetryConfig(
    3,              // Max retry attempts
    5*time.Minute,  // Max wait between retries
)
```

### Circuit Breaker

```go
// In fantasy chain
circuitBreaker := fantasy.WithCircuitBreaker(
    1,  // Failure threshold
    0,  // Reset timeout (0 = never reset until restart)
)
```

## Benefits vs Current Implementation

| Feature | Current | With Pooled Provider |
|---------|---------|---------------------|
| **API Key Rotation** | ❌ Single key | ✅ Multiple keys |
| **Account Failover** | ❌ No | ✅ Yes |
| **Smart Error Handling** | ❌ Generic | ✅ Per HTTP status |
| **State Persistence** | ❌ Lost on restart | ✅ Persisted |
| **Retry Logic** | ❌ Simple | ✅ Exponential backoff |
| **Retry-After Support** | ❌ No | ✅ Yes |
| **Load Balancing** | ❌ No | ✅ Round-robin |
| **Per-Model State** | ❌ No | ✅ Yes |

## Testing

### Manual Test

```go
// test/pooled_provider_test.go
func TestPooledProvider(t *testing.T) {
    provider, err := pooled.New(pooled.Config{
        ProviderName: "test",
        APIKeys:      []string{"key1", "key2", "key3"},
        CreateClient: mockClientCreator,
    })
    
    // Simulate rate limit on key1
    // Should automatically fallback to key2
    
    resp, err := provider.Generate(ctx, call)
    assert.NoError(t, err)
}
```

## Monitoring

### Get Account Status

```go
// Get underlying manager
manager := pooledProvider.GetManager()

// List all accounts
accounts := manager.List()

for _, account := range accounts {
    fmt.Printf("Account: %s\n", account.Label)
    fmt.Printf("  Status: %s\n", account.Status)
    fmt.Printf("  Unavailable: %v\n", account.Unavailable)
    fmt.Printf("  Quota Exceeded: %v\n", account.Quota.Exceeded)
    if !account.NextRetryAfter.IsZero() {
        fmt.Printf("  Next Retry: %s\n", account.NextRetryAfter)
    }
}
```

## Troubleshooting

### All accounts in cooldown

```
Error: All credentials for model gpt-4 are cooling down
Retry-After: 120 seconds
```

**Solution:** Wait for cooldown to expire, or add more API keys.

### No accounts available

```
Error: no auth available
```

**Solution:** Check that API keys are registered correctly.

## Future Enhancements

1. **Persistent Store** - Save state to disk for cross-restart persistence
2. **Metrics** - Track success rate per account
3. **Dynamic Key Management** - Add/remove keys at runtime
4. **Priority-based Selection** - Prefer certain accounts over others
5. **Cost Tracking** - Monitor usage per account

## References

- [CLIProxyAPI FALLBACK.md](../../../CLIProxyAPI/FALLBACK.md)
- [fantasy.LanguageModel Interface](../../language_model.go)
- [auth.Manager Implementation](../../../cliproxy/sdk/cliproxy/auth/conductor.go)


