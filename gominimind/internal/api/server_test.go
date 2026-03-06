package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gominimind/pkg/config"
	"gominimind/pkg/types"
)

// testServer 测试服务器包装器
type testServer struct {
	*Server
	router *gin.Engine
}

// TestServerInitialization tests server initialization
func TestServerInitialization(t *testing.T) {
	testConfig := createTestCfg()

	server, err := NewServer(testConfig, nil)
	require.NoError(t, err)
	require.NotNil(t, server)

	assert.NotNil(t, server.Config)
	assert.NotNil(t, server.Logger)

	assert.Equal(t, testConfig.Server.Host, server.Config.Server.Host)
	assert.Equal(t, testConfig.Server.Port, server.Config.Server.Port)
	assert.Equal(t, testConfig.Server.APIKey, server.Config.Server.APIKey)
}

// TestHealthCheck tests health check endpoint
func TestHealthCheck(t *testing.T) {
	ts := newTestServer(t)

	req, err := http.NewRequest("GET", "/health", nil)
	require.NoError(t, err)

	w := httptest.NewRecorder()
	ts.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Contains(t, response, "status")
}

// TestChatCompletionEndpoint tests chat completion endpoint
func TestChatCompletionEndpoint(t *testing.T) {
	ts := newTestServer(t)

	requestBody := types.ChatRequest{
		Model: "minimind",
		Messages: []types.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   10,
		Temperature: 0.7,
	}

	body, err := json.Marshal(requestBody)
	require.NoError(t, err)

	req, err := http.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w := httptest.NewRecorder()
	ts.router.ServeHTTP(w, req)

	// 因为没有实际模型，可能返回500或200
	// 只验证路由存在并能处理请求
	assert.NotEqual(t, http.StatusNotFound, w.Code)
}

// TestCompletionEndpoint tests completion endpoint
func TestCompletionEndpoint(t *testing.T) {
	ts := newTestServer(t)

	requestBody := types.CompletionRequest{
		Model:     "minimind",
		Prompt:    "Once upon a time",
		MaxTokens: 50,
	}

	body, err := json.Marshal(requestBody)
	require.NoError(t, err)

	req, err := http.NewRequest("POST", "/v1/completions", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w := httptest.NewRecorder()
	ts.router.ServeHTTP(w, req)

	assert.NotEqual(t, http.StatusNotFound, w.Code)
}

// TestEmbeddingEndpoint tests embedding endpoint
func TestEmbeddingEndpoint(t *testing.T) {
	ts := newTestServer(t)

	requestBody := types.EmbeddingRequest{
		Model: "minimind",
		Input: []string{"Hello world"},
	}

	body, err := json.Marshal(requestBody)
	require.NoError(t, err)

	req, err := http.NewRequest("POST", "/v1/embeddings", bytes.NewBuffer(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w := httptest.NewRecorder()
	ts.router.ServeHTTP(w, req)

	assert.NotEqual(t, http.StatusNotFound, w.Code)
}

// TestModelsEndpoint tests models list endpoint
func TestModelsEndpoint(t *testing.T) {
	ts := newTestServer(t)

	req, err := http.NewRequest("GET", "/v1/models", nil)
	require.NoError(t, err)
	req.Header.Set("Authorization", "Bearer test-key")

	w := httptest.NewRecorder()
	ts.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

// TestMetricsEndpoint tests metrics endpoint
func TestMetricsEndpoint(t *testing.T) {
	ts := newTestServer(t)

	req, err := http.NewRequest("GET", "/metrics", nil)
	require.NoError(t, err)

	w := httptest.NewRecorder()
	ts.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

// TestInvalidJSON tests invalid JSON handling
func TestInvalidJSON(t *testing.T) {
	ts := newTestServer(t)

	req, err := http.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString("invalid json"))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	w := httptest.NewRecorder()
	ts.router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

// BenchmarkHealthCheck benchmarks health check endpoint
func BenchmarkHealthCheck(b *testing.B) {
	ts := newTestServer(b)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req, _ := http.NewRequest("GET", "/health", nil)
		w := httptest.NewRecorder()
		ts.router.ServeHTTP(w, req)
	}
}

// ========== 辅助函数 ==========

func createTestCfg() *config.Config {
	return &config.Config{
		Server: config.ServerConfig{
			Host:        "localhost",
			Port:        8080,
			APIKey:      "test-key",
			MaxTokens:   2048,
			Temperature: 0.7,
			TopP:        0.9,
			Workers:     4,
			LogLevel:    "info",
			UseCache:    false,
			RateLimit:   100,
		},
		Cache: config.CacheConfig{
			Expiration: 60,
		},
	}
}

func newTestServer(t testing.TB) *testServer {
	gin.SetMode(gin.TestMode)

	testConfig := createTestCfg()

	server, err := NewServer(testConfig, nil)
	require.NoError(t, err)

	router := server.SetupRouter()

	return &testServer{
		Server: server,
		router: router,
	}
}
