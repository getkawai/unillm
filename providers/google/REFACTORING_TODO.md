# Google Provider Refactoring TODO

## Current Structure Issue

The Google provider currently has all code in a single `google.go` file (1279 lines), which differs from other providers that separate concerns into multiple files.

## Comparison with Other Providers

### OpenAI Provider Structure:
```
openai/
├── openai.go                    # Provider setup & initialization
├── language_model.go            # LanguageModel implementation
├── language_model_hooks.go      # Hooks for customization
├── responses_language_model.go  # Response handling
├── responses_options.go         # Response options
├── error.go                     # Error handling
└── provider_options.go          # Provider options
```

### Llama Provider Structure:
```
llama/
├── llama.go                     # Provider setup & initialization
├── language_model.go            # LanguageModel implementation
├── stream_processor.go          # Streaming logic
└── provider_options.go          # Provider options
```

### Google Provider (Current):
```
google/
├── google.go                    # EVERYTHING (1279 lines!)
├── auth.go                      # Authentication helpers
├── error.go                     # Error handling
├── provider_options.go          # Provider options
└── slice.go                     # Utility functions
```

## Proposed Refactoring

### Step 1: Create `language_model.go`
Move the following from `google.go`:
- `type languageModel struct`
- All `languageModel` methods:
  - `Model()`
  - `Provider()`
  - `Generate()`
  - `Stream()`
  - `GenerateObject()`
  - `StreamObject()`
  - `prepareParams()`
  - `generateObjectWithJSONMode()`
  - `streamObjectWithJSONMode()`
  - `mapResponse()`

### Step 2: Create `converters.go`
Move helper functions:
- `toGooglePrompt()`
- `toGoogleTools()`
- `convertSchemaProperties()`
- `convertToSchema()`
- `processArrayItems()`
- `mapJSONTypeToGoogle()`
- `mapFinishReason()`
- `mapUsage()`

### Step 3: Keep in `google.go`
- `type provider struct`
- `type options struct`
- `New()` function
- All `With*()` option functions
- `(*provider).Name()` method
- `(*provider).LanguageModel()` method

## Benefits of Refactoring

1. **Consistency** - Matches structure of other providers
2. **Maintainability** - Smaller files are easier to navigate and modify
3. **Separation of Concerns** - Clear boundaries between different responsibilities
4. **Readability** - Easier to find specific functionality
5. **Testing** - Can test components in isolation

## Implementation Notes

- This is a **non-breaking change** - only internal file organization
- All public APIs remain the same
- Tests should continue to pass without modification
- Can be done incrementally (one file at a time)

## Priority

**Medium** - Not urgent as current code works fine, but should be done before:
- Adding more features
- Significant modifications to the provider
- When other developers start contributing

## Estimated Effort

- 2-3 hours for refactoring
- 1 hour for testing and verification
- Total: ~4 hours

## Related Issues

- Consistency with other providers
- Code organization best practices
- Future maintainability

---

**Created**: 2026-01-04
**Status**: Pending
**Assigned**: TBD
