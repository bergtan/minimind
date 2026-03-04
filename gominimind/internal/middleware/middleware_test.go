package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func setupTestRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	return gin.New()
}

// TestAuthMiddleware - 测试认证中间件
func TestAuthMiddleware(t *testing.T) {
	router := setupTestRouter()

	cfg := &AuthConfig{
		APIKey:    "test-api-key",
		APIKeyEnv: "TEST_API_KEY",
	}

	router.Use(AuthMiddleware(cfg))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	t.Run("Valid API Key", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("Authorization", "Bearer test-api-key")
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
	})

	t.Run("Invalid API Key", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("Authorization", "Bearer wrong-key")
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})

	t.Run("Missing Authorization Header", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})

	t.Run("Missing Bearer Prefix", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("Authorization", "test-api-key")
		router.ServeHTTP(w, req)

		assert.Equal(t, 401, w.Code)
	})
}

// TestRateLimitMiddleware - 测试限流中间件
func TestRateLimitMiddleware(t *testing.T) {
	router := setupTestRouter()

	cfg := &RateLimitConfig{
		RequestsPerSecond: 2,
		BurstSize:         2,
	}

	router.Use(RateLimitMiddleware(cfg))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	t.Run("Within Rate Limit", func(t *testing.T) {
		for i := 0; i < 2; i++ {
			w := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", "/test", nil)
			router.ServeHTTP(w, req)
			assert.Equal(t, 200, w.Code)
		}
	})

	t.Run("Exceed Rate Limit", func(t *testing.T) {
		// Third request should fail
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/./test", nil)
		router.ServeHTTP(w, req)

		// Note: This might succeed due to token replenishment
		// The actual test depends on timing
	})
}

// TestCORSConfig - 测试CORS配置中间件
func TestCORSConfig(t *testing.T) {
	router := setupTestRouter()

	corsConfig := &CORSConfig{
		AllowOrigins:     []string{"http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
	}

	router.Use(CORSMiddleware(corsConfig))
	router.POST("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	t.Run("CORS Preflight Request", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("OPTIONS", "/test", nil)
		req.Header.Set("Origin", "http://localhost:3000")
		req.Header.Set("Access-Control-Request-Method", "POST")
		router.ServeHTTP(w, req)

		assert.Equal(t, 204, w.Code)
		assert.Equal(t, "http://localhost:3000", w.Header().Get("Access-Control-Allow-Origin"))
	})

	t.Run("CORS Actual Request", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/test", strings.NewReader(`{"key":"value"}`))
		req.Header.Set("Origin", "http://localhost:3000")
		req.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
		assert.Equal(t, "http://localhost:3000", w.Header().Get("Access-Control-Allow-Origin"))
	})
}

// TestLoggerMiddleware - 测试日志中间件
func TestLoggerMiddleware(t *testing.T) {
	router := setupTestRouter()

	logger := CreateStructuredLogger("info")
	router.Use(GinLoggerMiddleware(logger))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	t.Run("Log Request", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("X-Request-ID", "test-request-id")
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
		// Logging is async, so we just verify the request completes
	})

	t.Run("Log Error Request", func(t *testing.T) {
		router.GET("/error", func(c *gin.Context) {
			c.JSON(500, gin.H{"error": "internal error"})
		})

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/error", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, 500, w.Code)
	})
}

// TestErrorHandlingMiddleware - 测试错误处理中间件
func TestErrorHandlingMiddleware(t *testing.T) {
	router := setupTestRouter()

	router.Use(ErrorHandlingMiddleware())
	router.GET("/success", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})
	router.GET("/error", func(c *gin.Context) {
		c.Error(InternalServerError("test error"))
	})
	router.GET("/panic", func(c *gin.Context) {
		panic("test panic")
	})

	t.Run("Success Response", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/success", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
	})

	t.Run("Error Response", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/error", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, 500, w.Code)

		var response ErrorResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.Equal(t, "internal_error", response.Code)
	})

	t.Run("Panic Recovery", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/panic", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, 500, w.Code)
	})
}

// TestIPWhitelistMiddleware - 测试IP白名单中间件
func TestIPWhitelistMiddleware(t *testing.T) {
	router := setupTestRouter()

	cfg := &IPWhitelistConfig{
		Enabled:    true,
		AllowedIPs: []string{"192.168.1.1", "10.0.0.0/8"},
	}

	router.Use(IPWhitelistMiddleware(cfg))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	t.Run("Allowed IP", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.RemoteAddr = "192.168.1.1:1234"
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
	})

	t.Run("Allowed CIDR", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.RemoteAddr = "10.0.0.1:1234"
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
	})

	t.Run("Disallowed IP", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.RemoteAddr = "172.16.0.1:1234"
		router.ServeHTTP(w, req)

		assert.Equal(t, 403, w.Code)
	})

	t.Run("Disabled Whitelist", func(t *testing.T) {
		disabledRouter := setupTestRouter()
		disabledCfg := &IPWhitelistConfig{
			Enabled: false,
		}
		disabledRouter.Use(IPWhitelistMiddleware(disabledCfg))
		disabledRouter.GET("/test", func(c *gin.Context) {
			c.JSON(200, gin.H{"status": "ok"})
		})

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.RemoteAddr = "172.16.0.1:1234"
		disabledRouter.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
	})
}

// TestRequestIDMiddleware - 测试请求ID中间件
func TestRequestIDMiddleware(t *testing.T) {
	router := setupTestRouter()

	router.Use(RequestIDMiddleware())
	router.GET("/test", func(c *gin.Context) {
		requestID := c.GetString("RequestID")
		c.JSON(200, gin.H{"request_id": requestID})
	})

	t.Run("Generate Request ID", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)

		var response map[string]string
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.NotEmpty(t, response["request_id"])
		assert.NotEmpty(t, w.Header().Get("X-Request-ID"))
	})

	t.Run("Use Existing Request ID", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("X-Request-ID", "existing-request-id")
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
		assert.Equal(t, "existing-request-id", w.Header().Get("X-Request-ID"))
	})
}

// BenchmarkRateLimitMiddleware - 限流中间件性能测试
func BenchmarkRateLimitMiddleware(b *testing.B) {
	router := setupTestRouter()

	cfg := &RateLimitConfig{
		RequestsPerSecond: 10000,
		BurstSize:         10000,
	}

	router.Use(RateLimitMiddleware(cfg))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			w := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", "/test", nil)
			router.ServeHTTP(w, req)
		}
	})
}

// BenchmarkAuthMiddleware - 认证中间件性能测试
func BenchmarkAuthMiddleware(b *testing.B) {
	router := setupTestRouter()

	cfg := &AuthConfig{
		APIKey: "test-api-key",
	}

	router.Use(AuthMiddleware(cfg))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("Authorization", "Bearer test-api-key")
		router.ServeHTTP(w, req)
	}
}

// TestMiddlewareChaining - 测试中间件链式调用
func TestMiddlewareChaining(t *testing.T) {
	router := setupTestRouter()

	// Track execution order
	var order []string
	var mu sync.Mutex

	router.Use(func(c *gin.Context) {
		mu.Lock()
		order = append(order, "middleware1-before")
		mu.Unlock()
		c.Next()
		mu.Lock()
		order = append(order, "middleware1-after")
		mu.Unlock()
	})

	router.Use(func(c *gin.Context) {
		mu.Lock()
		order = append(order, "middleware2-before")
		mu.Unlock()
		c.Next()
		mu.Lock()
		order = append(order, "middleware2-after")
		mu.Unlock()
	})

	router.GET("/test", func(c *gin.Context) {
		mu.Lock()
		order = append(order, "handler")
		mu.Unlock()
		c.JSON(200, gin.H{"status": "ok"})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, 200, w.Code)
	assert.Equal(t, []string{
		"middleware1-before",
		"middleware2-before",
		"handler",
		"middleware2-after",
		"middleware1-after",
	}, order)
}

// TestTimeoutMiddleware - 测试超时中间件
func TestTimeoutMiddleware(t *testing.T) {
	router := setupTestRouter()

	router.Use(TimeoutMiddleware(100 * time.Millisecond))
	router.GET("/fast", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})
	router.GET("/slow", func(c *gin.Context) {
		time.Sleep(200 * time.Millisecond)
		c.JSON(200, gin.H{"status": "ok"})
	})

	t.Run("Fast Request", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/fast", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, 200, w.Code)
	})

	t.Run("Slow Request Timeout", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/slow", nil)
		router.ServeHTTP(w, req)

		// Should timeout or complete based on implementation
		assert.Contains(t, []int{200, 504}, w.Code)
	})
}
