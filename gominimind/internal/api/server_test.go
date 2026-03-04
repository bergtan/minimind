package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"minimind/pkg/config"
	"minimind/pkg/model"
	"minimind/pkg/tokenizer"
	"minimind/pkg/types"
)

// TestServerInitialization tests server initialization
func TestServerInitialization(t *testing.T) {
	// Create test configuration
	testConfig := &config.ServerConfig{
		Host:           "localhost",
		Port:           8080,
		APIKey:         "test-key",
		MaxTokens:      2048,
		Temperature:    0.7,
		TopP:           0.9,
		Workers:        4,
		LogLevel:       "info",
		ModelPath:      "./test-model",
		UseGPU:         false,
		MaxBatchSize:   32,
		CacheEnabled:   true,
		RateLimit:      100,
		RateLimitBurst: 10,
	}

	// Test server creation
	server, err := NewServer(testConfig)
	require.NoError(t, err)
	require.NotNil(t, server)

	// Test server components
	assert.NotNil(t, server.config)
	assert.NotNil(t, server.router)
	assert.NotNil(t, server.modelManager)
	assert.NotNil(t, server.tokenizer)

	// Test configuration values
	assert.Equal(t, testConfig.Host, server.config.Host)
	assert.Equal(t, testConfig.Port, server.config.Port)
	assert.Equal(t, testConfig.APIKey, server.config.APIKey)
}

// TestOpenAICompatibility tests OpenAI API compatibility
func TestOpenAICompatibility(t *testing.T) {
	// Create test server
	server := createTestServer(t)

	tests := []struct {
		name           string
		endpoint       string
		method         string
		requestBody    interface{}
		expectedStatus int
		expectedFields []string
	}{
		{
			name:     "Chat completions endpoint",
			endpoint: "/v1/chat/completions",
			method:   "POST",
			requestBody: types.ChatCompletionRequest{
				Model: "minimind",
				Messages: []types.Message{
					{Role: "user", Content: "Hello, world!"},
				},
				MaxTokens:   100,
				Temperature: 0.7,
			},
			expectedStatus: http.StatusOK,
			expectedFields: []string{"id", "object", "created", "model", "choices", "usage"},
		},
		{
			name:     "Completions endpoint",
			endpoint: "/v1/completions",
			method:   "POST",
			requestBody: types.CompletionRequest{
				Model:     "minimind",
				Prompt:    "Once upon a time",
				MaxTokens: 50,
			},
			expectedStatus: http.StatusOK,
			expectedFields: []string{"id", "object", "created", "model", "choices", "usage"},
		},
		{
			name:     "Embeddings endpoint",
			endpoint: "/v1/embeddings",
			method:   "POST",
			requestBody: types.EmbeddingRequest{
				Model: "minimind",
				Input: []string{"Hello world", "Test embedding"},
			},
			expectedStatus: http.StatusOK,
			expectedFields: []string{"object", "data", "model", "usage"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create request
			body, err := json.Marshal(tt.requestBody)
			require.NoError(t, err)

			req, err := http.NewRequest(tt.method, tt.endpoint, bytes.NewBuffer(body))
			require.NoError(t, err)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer test-key")

			// Create response recorder
			w := httptest.NewRecorder()

			// Serve request
			server.router.ServeHTTP(w, req)

			// Check response
			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus == http.StatusOK {
				var response map[string]interface{}
				err = json.Unmarshal(w.Body.Bytes(), &response)
				require.NoError(t, err)

				// Check required fields
				for _, field := range tt.expectedFields {
					assert.Contains(t, response, field)
				}
			}
		})
	}
}

// TestCustomAPIEndpoints tests custom API endpoints
func TestCustomAPIEndpoints(t *testing.T) {
	server := createTestServer(t)

	tests := []struct {
		name           string
		endpoint       string
		method         string
		expectedStatus int
		expectedFields []string
	}{
		{
			name:           "Health check endpoint",
			endpoint:       "/health",
			method:         "GET",
			expectedStatus: http.StatusOK,
			expectedFields: []string{"status", "timestamp", "version", "model_loaded"},
		},
		{
			name:           "Detailed health check",
			endpoint:       "/health/detailed",
			method:         "GET",
			expectedStatus: http.StatusOK,
			expectedFields: []string{"status", "timestamp", "version", "model_loaded", "gpu_available", "gpu_memory_usage"},
		},
		{
			name:           "Models list endpoint",
			endpoint:       "/api/v1/models",
			method:         "GET",
			expectedStatus: http.StatusOK,
			expectedFields: []string{"models"},
		},
		{
			name:           "Model details endpoint",
			endpoint:       "/api/v1/models/minimind",
			method:         "GET",
			expectedStatus: http.StatusOK,
			expectedFields: []string{"id", "name", "description", "version", "context_length", "parameters"},
		},
		{
			name:           "Metrics endpoint",
			endpoint:       "/metrics",
			method:         "GET",
			expectedStatus: http.StatusOK,
			expectedFields: nil, // Prometheus metrics format
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := http.NewRequest(tt.method, tt.endpoint, nil)
			require.NoError(t, err)
			req.Header.Set("Authorization", "Bearer test-key")

			w := httptest.NewRecorder()
			server.router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus == http.StatusOK && len(tt.expectedFields) > 0 {
				var response map[string]interface{}
				err = json.Unmarshal(w.Body.Bytes(), &response)
				require.NoError(t, err)

				for _, field := range tt.expectedFields {
					assert.Contains(t, response, field)
				}
			}
		})
	}
}

// TestAuthentication tests API authentication
func TestAuthentication(t *testing.T) {
	server := createTestServer(t)

	tests := []struct {
		name           string
		authHeader     string
		expectedStatus int
		errorCode      string
	}{
		{
			name:           "Valid API key",
			authHeader:     "Bearer test-key",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Invalid API key",
			authHeader:     "Bearer wrong-key",
			expectedStatus: http.StatusUnauthorized,
			errorCode:      "invalid_api_key",
		},
		{
			name:           "Missing API key",
			authHeader:     "",
			expectedStatus: http.StatusUnauthorized,
			errorCode:      "missing_api_key",
		},
		{
			name:           "Malformed auth header",
			authHeader:     "InvalidFormat",
			expectedStatus: http.StatusUnauthorized,
			errorCode:      "invalid_auth_header",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create chat completion request
			requestBody := types.ChatCompletionRequest{
				Model: "minimind",
				Messages: []types.Message{
					{Role: "user", Content: "Test message"},
				},
				MaxTokens: 10,
			}

			body, err := json.Marshal(requestBody)
			require.NoError(t, err)

			req, err := http.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
			require.NoError(t, err)
			req.Header.Set("Content-Type", "application/json")
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}

			w := httptest.NewRecorder()
			server.router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus != http.StatusOK {
				var errorResponse types.ErrorResponse
				err = json.Unmarshal(w.Body.Bytes(), &errorResponse)
				require.NoError(t, err)
				assert.Equal(t, tt.errorCode, errorResponse.Error.Code)
			}
		})
	}
}

// TestRateLimiting tests rate limiting functionality
func TestRateLimiting(t *testing.T) {
	server := createTestServer(t)

	// Create test request
	requestBody := types.ChatCompletionRequest{
		Model: "minimind",
		Messages: []types.Message{
			{Role: "user", Content: "Test message"},
		},
		MaxTokens: 10,
	}

	body, err := json.Marshal(requestBody)
	require.NoError(t, err)

	// Send multiple requests quickly to trigger rate limiting
	for i := 0; i < 15; i++ {
		req, err := http.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer test-key")

		w := httptest.NewRecorder()
		server.router.ServeHTTP(w, req)

		if i >= 10 {
			// After 10 requests, should get rate limited
			assert.Equal(t, http.StatusTooManyRequests, w.Code)

			var errorResponse types.ErrorResponse
			err = json.Unmarshal(w.Body.Bytes(), &errorResponse)
			require.NoError(t, err)
			assert.Equal(t, "rate_limit_exceeded", errorResponse.Error.Code)
		} else {
			assert.Equal(t, http.StatusOK, w.Code)
		}

		// Small delay between requests
		time.Sleep(10 * time.Millisecond)
	}
}

// TestErrorHandling tests error handling scenarios
func TestErrorHandling(t *testing.T) {
	server := createTestServer(t)

	tests := []struct {
		name           string
		endpoint       string
		method         string
		requestBody    interface{}
		expectedStatus int
		errorCode      string
		errorMessage   string
	}{
		{
			name:           "Invalid JSON",
			endpoint:       "/v1/chat/completions",
			method:         "POST",
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			errorCode:      "invalid_json",
			errorMessage:   "invalid character",
		},
		{
			name:           "Missing required field",
			endpoint:       "/v1/chat/completions",
			method:         "POST",
			requestBody:    map[string]interface{}{},
			expectedStatus: http.StatusBadRequest,
			errorCode:      "missing_required_field",
			errorMessage:   "messages",
		},
		{
			name:     "Invalid model name",
			endpoint: "/v1/chat/completions",
			method:   "POST",
			requestBody: types.ChatCompletionRequest{
				Model: "invalid-model",
				Messages: []types.Message{
					{Role: "user", Content: "Test"},
				},
			},
			expectedStatus: http.StatusBadRequest,
			errorCode:      "invalid_model",
			errorMessage:   "model not found",
		},
		{
			name:     "Invalid parameter value",
			endpoint: "/v1/chat/completions",
			method:   "POST",
			requestBody: types.ChatCompletionRequest{
				Model: "minimind",
				Messages: []types.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: -1.0,
			},
			expectedStatus: http.StatusBadRequest,
			errorCode:      "invalid_parameter",
			errorMessage:   "temperature",
		},
		{
			name:           "Non-existent endpoint",
			endpoint:       "/v1/nonexistent",
			method:         "POST",
			requestBody:    nil,
			expectedStatus: http.StatusNotFound,
			errorCode:      "endpoint_not_found",
			errorMessage:   "endpoint not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var bodyBytes []byte
			if tt.requestBody != nil {
				switch v := tt.requestBody.(type) {
				case string:
					bodyBytes = []byte(v)
				default:
					var err error
					bodyBytes, err = json.Marshal(v)
					require.NoError(t, err)
				}
			}

			req, err := http.NewRequest(tt.method, tt.endpoint, bytes.NewBuffer(bodyBytes))
			require.NoError(t, err)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer test-key")

			w := httptest.NewRecorder()
			server.router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			if tt.expectedStatus != http.StatusOK {
				var errorResponse types.ErrorResponse
				err = json.Unmarshal(w.Body.Bytes(), &errorResponse)
				require.NoError(t, err)

				assert.Equal(t, tt.errorCode, errorResponse.Error.Code)
				if tt.errorMessage != "" {
					assert.Contains(t, errorResponse.Error.Message, tt.errorMessage)
				}
			}
		})
	}
}

// TestStreamingResponse tests streaming response functionality
func TestStreamingResponse(t *testing.T) {
	server := createTestServer(t)

	requestBody := types.ChatCompletionRequest{
		Model: "minimind",
		Messages: []types.Message{
			{Role: "user", Content: "Tell me a short story"},
		},
		MaxTokens:   50,
		Stream:      true,
		Temperature: 0.7,
	}

	body, err := json.Marshal(requestBody)
	require.NoError(t, err)

	req, err := http.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w := httptest.NewRecorder()
	server.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	// Check streaming response format
	responseBody := w.Body.String()
	assert.Contains(t, responseBody, "data:")
	assert.Contains(t, responseBody, "[DONE]")

	// Parse streaming response
	lines := bytes.Split(w.Body.Bytes(), []byte("\n"))

	var dataLines int
	for _, line := range lines {
		if bytes.HasPrefix(line, []byte("data:")) && !bytes.Contains(line, []byte("[DONE]")) {
			dataLines++

			// Parse JSON from data line
			jsonStr := bytes.TrimPrefix(line, []byte("data:"))
			var chunk map[string]interface{}
			err := json.Unmarshal(jsonStr, &chunk)
			assert.NoError(t, err)

			// Check chunk structure
			assert.Contains(t, chunk, "id")
			assert.Contains(t, chunk, "object")
			assert.Contains(t, chunk, "choices")
		}
	}

	assert.True(t, dataLines > 0, "Should have at least one data chunk")
}

// TestBatchProcessing tests batch processing functionality
func TestBatchProcessing(t *testing.T) {
	server := createTestServer(t)

	batchRequest := types.BatchChatRequest{
		Requests: []types.ChatCompletionRequest{
			{
				Model: "minimind",
				Messages: []types.Message{
					{Role: "user", Content: "First message"},
				},
				MaxTokens: 20,
			},
			{
				Model: "minimind",
				Messages: []types.Message{
					{Role: "user", Content: "Second message"},
				},
				MaxTokens: 30,
			},
		},
		Parallel: 2,
	}

	body, err := json.Marshal(batchRequest)
	require.NoError(t, err)

	req, err := http.NewRequest("POST", "/api/v1/batch/chat", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w := httptest.NewRecorder()
	server.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response types.BatchChatResponse
	err = json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.Len(t, response.Responses, 2)

	for i, resp := range response.Responses {
		assert.NotNil(t, resp.ID)
		assert.Equal(t, "chat.completion", resp.Object)
		assert.Len(t, resp.Choices, 1)
		assert.NotEmpty(t, resp.Choices[0].Message.Content)
		assert.Equal(t, batchRequest.Requests[i].MaxTokens, resp.Usage.CompletionTokens)
	}
}

// TestMiddlewareIntegration tests middleware functionality
func TestMiddlewareIntegration(t *testing.T) {
	server := createTestServer(t)

	// Test CORS middleware
	req, err := http.NewRequest("OPTIONS", "/v1/chat/completions", nil)
	require.NoError(t, err)
	req.Header.Set("Origin", "http://localhost:3000")
	req.Header.Set("Access-Control-Request-Method", "POST")

	w := httptest.NewRecorder()
	server.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Header().Get("Access-Control-Allow-Origin"), "*")

	// Test logging middleware
	requestBody := types.ChatCompletionRequest{
		Model: "minimind",
		Messages: []types.Message{
			{Role: "user", Content: "Test logging"},
		},
		MaxTokens: 10,
	}

	body, err := json.Marshal(requestBody)
	require.NoError(t, err)

	req, err = http.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w = httptest.NewRecorder()
	server.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	// Logging middleware should not affect response
}

// TestPerformanceMonitoring tests performance monitoring
func TestPerformanceMonitoring(t *testing.T) {
	server := createTestServer(t)

	// Test metrics endpoint
	req, err := http.NewRequest("GET", "/metrics", nil)
	require.NoError(t, err)

	w := httptest.NewRecorder()
	server.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	// Metrics should be in Prometheus format
	metricsBody := w.Body.String()
	assert.Contains(t, metricsBody, "minimind_requests_total")
	assert.Contains(t, metricsBody, "minimind_request_duration_seconds")

	// Make a request to increment metrics
	requestBody := types.ChatCompletionRequest{
		Model: "minimind",
		Messages: []types.Message{
			{Role: "user", Content: "Test metrics"},
		},
		MaxTokens: 5,
	}

	body, err := json.Marshal(requestBody)
	require.NoError(t, err)

	req, err = http.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w = httptest.NewRecorder()
	server.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

// BenchmarkAPIPerformance benchmarks API performance
func BenchmarkAPIPerformance(b *testing.B) {
	server := createTestServer(b)

	requestBody := types.ChatCompletionRequest{
		Model: "minimind",
		Messages: []types.Message{
			{Role: "user", Content: "Benchmark test message"},
		},
		MaxTokens: 10,
	}

	body, err := json.Marshal(requestBody)
	require.NoError(b, err)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req, err := http.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
		require.NoError(b, err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer test-key")

		w := httptest.NewRecorder()
		server.router.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			b.Fatalf("Request failed with status: %d", w.Code)
		}
	}
}

// Helper function to create test server
func createTestServer(t testing.TB) *Server {
	// Create test configuration
	testConfig := &config.ServerConfig{
		Host:           "localhost",
		Port:           8080,
		APIKey:         "test-key",
		MaxTokens:      2048,
		Temperature:    0.7,
		TopP:           0.9,
		Workers:        4,
		LogLevel:       "info",
		ModelPath:      "./test-model",
		UseGPU:         false,
		MaxBatchSize:   32,
		CacheEnabled:   true,
		RateLimit:      10, // Low limit for testing
		RateLimitBurst: 2,
	}

	// Create mock model manager
	modelManager := &model.MockModelManager{}

	// Create mock tokenizer
	tokenizer, err := tokenizer.NewTokenizer(6400)
	require.NoError(t, err)

	// Create server with test components
	server := &Server{
		config:       testConfig,
		router:       gin.New(),
		modelManager: modelManager,
		tokenizer:    tokenizer,
	}

	// Setup routes
	setupRoutes(server)

	return server
}

// Mock model manager for testing
type MockModelManager struct{}

func (m *MockModelManager) LoadModel(path string) error { return nil }
func (m *MockModelManager) UnloadModel() error          { return nil }
func (m *MockModelManager) IsLoaded() bool              { return true }
func (m *MockModelManager) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":               "minimind",
		"version":            "1.0.0",
		"context_length":     32768,
		"parameters":         "25.8M",
		"supported_features": []string{"chat", "completion", "embedding"},
	}
}
func (m *MockModelManager) ChatCompletion(request *types.ChatCompletionRequest) (*types.ChatCompletionResponse, error) {
	return &types.ChatCompletionResponse{
		ID:      "chatcmpl-test",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "minimind",
		Choices: []types.Choice{
			{
				Index: 0,
				Message: types.Message{
					Role:    "assistant",
					Content: "This is a test response from the mock model.",
				},
				FinishReason: "stop",
			},
		},
		Usage: types.Usage{
			PromptTokens:     5,
			CompletionTokens: 10,
			TotalTokens:      15,
		},
	}, nil
}
func (m *MockModelManager) Completion(request *types.CompletionRequest) (*types.CompletionResponse, error) {
	return &types.CompletionResponse{
		ID:      "cmpl-test",
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   "minimind",
		Choices: []types.CompletionChoice{
			{
				Text:         "This is a test completion response.",
				Index:        0,
				LogProbs:     nil,
				FinishReason: "length",
			},
		},
		Usage: types.Usage{
			PromptTokens:     3,
			CompletionTokens: 7,
			TotalTokens:      10,
		},
	}, nil
}
func (m *MockModelManager) Embedding(request *types.EmbeddingRequest) (*types.EmbeddingResponse, error) {
	return &types.EmbeddingResponse{
		Object: "list",
		Data: []types.EmbeddingData{
			{
				Object:    "embedding",
				Embedding: []float32{0.1, 0.2, 0.3},
				Index:     0,
			},
		},
		Model: "minimind",
		Usage: types.Usage{
			PromptTokens: 5,
			TotalTokens:  5,
		},
	}, nil
}
func (m *MockModelManager) BatchChat(request *types.BatchChatRequest) (*types.BatchChatResponse, error) {
	responses := make([]types.ChatCompletionResponse, len(request.Requests))
	for i := range request.Requests {
		responses[i] = types.ChatCompletionResponse{
			ID:      "chatcmpl-batch-" + string(rune(i)),
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "minimind",
			Choices: []types.Choice{
				{
					Index: 0,
					Message: types.Message{
						Role:    "assistant",
						Content: "Batch response " + string(rune(i)),
					},
					FinishReason: "stop",
				},
			},
			Usage: types.Usage{
				PromptTokens:     5,
				CompletionTokens: uint32(request.Requests[i].MaxTokens),
				TotalTokens:      5 + uint32(request.Requests[i].MaxTokens),
			},
		}
	}

	return &types.BatchChatResponse{
		Responses: responses,
	}, nil
}
