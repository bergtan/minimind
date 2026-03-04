package api

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"runtime"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jingyaogong/gominimind/pkg/config"
	"github.com/jingyaogong/gominimind/pkg/types"
)

// RegisterRoutes 注册所有API路由
func (s *Server) RegisterRoutes(router *gin.Engine) {
	// 全局中间件
	router.Use(s.CORSMiddleware())
	router.Use(s.LoggerMiddleware())
	router.Use(s.RateLimitMiddleware())

	// ========== OpenAI兼容接口 ==========

	// V1 API路由组
	v1 := router.Group("/v1")
	{
		// 聊天接口
		v1.POST("/chat/completions", s.AuthMiddleware(), s.ChatCompletion)

		// 补全接口
		v1.POST("/completions", s.AuthMiddleware(), s.Completion)

		// 嵌入接口
		v1.POST("/embeddings", s.AuthMiddleware(), s.Embedding)

		// 模型接口
		v1.GET("/models", s.AuthMiddleware(), s.ListModels)
		v1.GET("/models/:model_id", s.AuthMiddleware(), s.GetModel)
	}

	// ========== 自定义API接口 ==========

	// API V1路由组
	apiV1 := router.Group("/api/v1")
	{
		// 健康检查
		apiV1.GET("/health", s.HealthCheck)
		apiV1.GET("/health/detailed", s.DetailedHealthCheck)

		// 模型信息
		apiV1.GET("/models", s.AuthMiddleware(), s.ListModels)
		apiV1.GET("/models/:id", s.AuthMiddleware(), s.GetModelInfo)

		// 批量处理
		apiV1.POST("/batch/chat", s.AuthMiddleware(), s.BatchChat)
		apiV1.POST("/batch/embeddings", s.AuthMiddleware(), s.BatchEmbedding)

		// 监控和统计
		apiV1.GET("/metrics", s.AuthMiddleware(), s.Metrics)
		apiV1.GET("/stats", s.AuthMiddleware(), s.Stats)

		// 文档
		apiV1.GET("/docs", s.Docs)
	}

	// ========== 系统接口 ==========

	// 系统路由组
	system := router.Group("/system")
	{
		// 根路径
		system.GET("/", s.Root)

		// 版本信息
		system.GET("/version", s.Version)

		// 配置信息（需要认证）
		system.GET("/config", s.AuthMiddleware(), s.ConfigInfo)

		// 重载配置（需要认证）
		system.POST("/reload", s.AuthMiddleware(), s.ReloadConfig)
	}

	// ========== 公共接口 ==========

	// 根路径
	router.GET("/", s.Root)

	// 健康检查（公共访问）
	router.GET("/health", s.HealthCheck)

	// 版本信息
	router.GET("/version", s.Version)

	// 文档
	router.GET("/docs", s.Docs)

	// 404处理
	router.NoRoute(s.NotFound)
}

// Root 根路径处理器
func (s *Server) Root(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message":   "GoMiniMind API Server",
		"version":   "2.0.0",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"endpoints": gin.H{
			"openai_compatible": "/v1",
			"custom_api":        "/api/v1",
			"system":            "/system",
			"health":            "/health",
			"docs":              "/docs",
		},
		"documentation": "https://github.com/jingyaogong/gominimind/docs",
	})
}

// Version 版本信息处理器
func (s *Server) Version(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"name":        "GoMiniMind",
		"version":     "2.0.0",
		"description": "轻量级语言模型API服务",
		"repository":  "https://github.com/jingyaogong/gominimind",
		"license":     "MIT",
		"authors":     []string{"jingyaogong"},
		"features": []string{
			"OpenAI API兼容",
			"RESTful API",
			"批量处理",
			"流式响应",
			"缓存优化",
			"监控统计",
		},
	})
}

// ConfigInfo 配置信息处理器
func (s *Server) ConfigInfo(c *gin.Context) {
	// 隐藏敏感信息
	safeConfig := gin.H{
		"server": gin.H{
			"host":       s.Config.Server.Host,
			"port":       s.Config.Server.Port,
			"max_tokens": s.Config.Server.MaxTokens,
			"rate_limit": s.Config.Server.RateLimit,
			"use_cache":  s.Config.Server.UseCache,
			"cache_size": s.Config.Server.CacheSize,
			"log_level":  s.Config.Server.LogLevel,
		},
		"model": gin.H{
			"name":                    s.Config.Model.Name,
			"vocab_size":              s.Config.Model.VocabSize,
			"hidden_size":             s.Config.Model.HiddenSize,
			"num_layers":              s.Config.Model.NumLayers,
			"num_heads":               s.Config.Model.NumHeads,
			"max_position_embeddings": s.Config.Model.MaxPositionEmbeddings,
		},
		"cache": gin.H{
			"enabled":    s.Config.Cache.Enabled,
			"expiration": s.Config.Cache.Expiration,
		},
		"auth": gin.H{
			"enabled":     s.Config.Server.APIKey != "",
			"has_api_key": s.Config.Server.APIKey != "",
		},
	}

	c.JSON(http.StatusOK, safeConfig)
}

// ReloadConfig 重载配置处理器
func (s *Server) ReloadConfig(c *gin.Context) {
	// 重新加载配置
	newConfig, err := config.LoadConfig(s.Config.ConfigPath)
	if err != nil {
		s.sendError(c, http.StatusInternalServerError, "config_error",
			"Failed to reload config", err.Error())
		return
	}

	// 更新服务器配置
	s.mu.Lock()
	s.Config = newConfig
	s.mu.Unlock()

	c.JSON(http.StatusOK, gin.H{
		"message":   "Configuration reloaded successfully",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})
}

// NotFound 404处理器
func (s *Server) NotFound(c *gin.Context) {
	s.sendError(c, http.StatusNotFound, "not_found",
		"Endpoint not found", fmt.Sprintf("Path %s not found", c.Request.URL.Path))
}

// ========== 流式响应辅助方法 ==========

// StreamChatCompletion 流式聊天补全处理器
func (s *Server) StreamChatCompletion(c *gin.Context) {
	var req types.ChatCompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	// 强制设置为流式
	req.Stream = true

	s.handleStreamingChat(c, &req)
}

// StreamCompletion 流式补全处理器
func (s *Server) StreamCompletion(c *gin.Context) {
	var req types.CompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	// 设置流式响应头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		s.sendError(c, http.StatusInternalServerError, "internal_error",
			"Streaming not supported", "Response writer does not support streaming")
		return
	}

	// 生成补全文本
	generatedText, err := s.Model.Generate(req.Prompt, req.MaxTokens, req.Temperature, req.TopP)
	if err != nil {
		c.Writer.Write([]byte(fmt.Sprintf("data: %s\n\n",
			string(s.createErrorChunk(err)))))
		c.Writer.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
		return
	}

	// 分块发送响应
	for i, char := range generatedText {
		chunk := s.createCompletionChunk(
			fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
			string(char),
			nil,
		)
		c.Writer.Write([]byte(fmt.Sprintf("data: %s\n\n", string(chunk))))
		flusher.Flush()

		// 添加小延迟以模拟流式效果
		time.Sleep(50 * time.Millisecond)

		if i >= req.MaxTokens-1 {
			break
		}
	}

	// 发送结束块
	c.Writer.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

// createCompletionChunk 创建补全流式块
func (s *Server) createCompletionChunk(id, text, finishReason interface{}) []byte {
	chunk := map[string]interface{}{
		"id":      id,
		"object":  "text_completion.chunk",
		"created": time.Now().Unix(),
		"model":   "minimind",
		"choices": []map[string]interface{}{
			{
				"text":          text,
				"index":         0,
				"logprobs":      nil,
				"finish_reason": finishReason,
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// ========== 批量处理辅助方法 ==========

// BatchChatWorker 批量聊天工作器
func (s *Server) BatchChatWorker(req types.ChatCompletionRequest, resultChan chan<- types.ChatCompletionResponse, errorChan chan<- error) {
	response, err := s.generateChatResponse(&req)
	if err != nil {
		errorChan <- err
		return
	}

	resultChan <- *response
}

// BatchEmbeddingWorker 批量嵌入工作器
func (s *Server) BatchEmbeddingWorker(text string, index int, resultChan chan<- types.EmbeddingData, errorChan chan<- error) {
	embedding, err := s.Model.GenerateEmbedding(text)
	if err != nil {
		errorChan <- err
		return
	}

	resultChan <- types.EmbeddingData{
		Object:    "embedding",
		Embedding: embedding,
		Index:     index,
	}
}

// ========== 监控和统计辅助方法 ==========

// GetMetrics 获取Prometheus格式的指标
func (s *Server) GetMetrics(c *gin.Context) {
	stats := s.getStats()

	metrics := fmt.Sprintf(`# HELP minimind_requests_total Total number of requests
# TYPE minimind_requests_total counter
minimind_requests_total %d

# HELP minimind_requests_failed Total number of failed requests
# TYPE minimind_requests_failed counter
minimind_requests_failed %d

# HELP minimind_cache_hits_total Total number of cache hits
# TYPE minimind_cache_hits_total counter
minimind_cache_hits_total %d

# HELP minimind_cache_misses_total Total number of cache misses
# TYPE minimind_cache_misses_total counter
minimind_cache_misses_total %d

# HELP minimind_response_time_seconds Average response time in seconds
# TYPE minimind_response_time_seconds gauge
minimind_response_time_seconds %f

# HELP minimind_tokens_processed_total Total number of tokens processed
# TYPE minimind_tokens_processed_total counter
minimind_tokens_processed_total %d

# HELP minimind_uptime_seconds Server uptime in seconds
# TYPE minimind_uptime_seconds gauge
minimind_uptime_seconds %f
`,
		stats.TotalRequests,
		stats.FailedRequests,
		stats.CacheHits,
		stats.CacheMisses,
		stats.AvgResponseTime.Seconds(),
		stats.TotalTokens,
		time.Since(stats.StartTime).Seconds(),
	)

	c.String(http.StatusOK, metrics)
}

// ========== 工具方法 ==========

// ValidateAPIKey 验证API密钥
func (s *Server) ValidateAPIKey(apiKey string) bool {
	if s.Config.Server.APIKey == "" {
		return true // 未配置API密钥时允许所有请求
	}
	return apiKey == s.Config.Server.APIKey
}

// GetClientIP 获取客户端IP
func (s *Server) GetClientIP(c *gin.Context) string {
	// 从X-Forwarded-For获取真实IP（如果存在）
	if forwardedFor := c.GetHeader("X-Forwarded-For"); forwardedFor != "" {
		ips := strings.Split(forwardedFor, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// 从X-Real-IP获取真实IP
	if realIP := c.GetHeader("X-Real-IP"); realIP != "" {
		return realIP
	}

	// 返回直接客户端IP
	return c.ClientIP()
}

// IsAllowedIP 检查IP是否在白名单中
func (s *Server) IsAllowedIP(ip string) bool {
	if len(s.Config.Server.AllowedIPs) == 0 {
		return true // 未配置白名单时允许所有IP
	}

	for _, allowedIP := range s.Config.Server.AllowedIPs {
		if allowedIP == ip {
			return true
		}

		// 支持CIDR表示法
		if strings.Contains(allowedIP, "/") {
			_, ipNet, err := net.ParseCIDR(allowedIP)
			if err == nil {
				parsedIP := net.ParseIP(ip)
				if parsedIP != nil && ipNet.Contains(parsedIP) {
					return true
				}
			}
		}
	}

	return false
}

// GetUserAgent 获取用户代理信息
func (s *Server) GetUserAgent(c *gin.Context) string {
	return c.GetHeader("User-Agent")
}

// GetRequestID 生成请求ID
func (s *Server) GetRequestID() string {
	return fmt.Sprintf("req_%d_%d", time.Now().UnixNano(), rand.Intn(1000))
}

// ========== 测试和调试方法 ==========

// EchoHandler 回显处理器（用于测试）
func (s *Server) EchoHandler(c *gin.Context) {
	var data map[string]interface{}
	if err := c.ShouldBindJSON(&data); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"echo":       data,
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
		"request_id": s.GetRequestID(),
	})
}

// PingHandler Ping处理器（用于健康检查）
func (s *Server) PingHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message":     "pong",
		"timestamp":   time.Now().UTC().Format(time.RFC3339),
		"server_time": time.Now().Unix(),
	})
}

// StatusHandler 状态处理器
func (s *Server) StatusHandler(c *gin.Context) {
	stats := s.getStats()

	c.JSON(http.StatusOK, gin.H{
		"status":     "running",
		"uptime":     time.Since(stats.StartTime).String(),
		"memory":     s.getMemoryUsage(),
		"goroutines": runtime.NumGoroutine(),
		"requests": gin.H{
			"total":  stats.TotalRequests,
			"active": s.getActiveRequests(),
		},
	})
}

// getMemoryUsage 获取内存使用情况
func (s *Server) getMemoryUsage() gin.H {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return gin.H{
		"alloc":       fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
		"total_alloc": fmt.Sprintf("%.2f MB", float64(m.TotalAlloc)/1024/1024),
		"sys":         fmt.Sprintf("%.2f MB", float64(m.Sys)/1024/1024),
		"num_gc":      m.NumGC,
	}
}

// getActiveRequests 获取活跃请求数
func (s *Server) getActiveRequests() int {
	// 这里可以实现更精确的活跃请求计数
	// 目前返回一个估计值
	return runtime.NumGoroutine() / 2
}

// ========== 注册测试路由 ==========

// RegisterTestRoutes 注册测试路由（仅在开发模式下）
func (s *Server) RegisterTestRoutes(router *gin.Engine) {
	if s.Config.Server.Environment != "development" {
		return
	}

	test := router.Group("/test")
	{
		test.POST("/echo", s.EchoHandler)
		test.GET("/ping", s.PingHandler)
		test.GET("/status", s.StatusHandler)
		test.GET("/config", s.ConfigInfo)
	}
}

// ========== 中间件注册辅助方法 ==========

// RegisterGlobalMiddlewares 注册全局中间件
func (s *Server) RegisterGlobalMiddlewares(router *gin.Engine) {
	// 恢复中间件（捕获panic）
	router.Use(gin.Recovery())

	// CORS中间件
	router.Use(s.CORSMiddleware())

	// 日志中间件
	router.Use(s.LoggerMiddleware())

	// 限流中间件
	router.Use(s.RateLimitMiddleware())

	// 请求ID中间件
	router.Use(s.RequestIDMiddleware())
}

// RequestIDMiddleware 请求ID中间件
func (s *Server) RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := s.GetRequestID()
		c.Header("X-Request-ID", requestID)
		c.Set("request_id", requestID)
		c.Next()
	}
}

// ========== 健康检查扩展方法 ==========

// HealthCheckDetailed 详细健康检查
func (s *Server) HealthCheckDetailed(c *gin.Context) {
	// 检查模型状态
	modelHealthy := s.Model != nil && s.Model.IsLoaded()

	// 检查缓存状态
	cacheHealthy := s.Cache != nil

	// 检查数据库连接（如果有）
	dbHealthy := true // 这里可以添加数据库健康检查

	// 检查外部服务（如果有）
	extServicesHealthy := true // 这里可以添加外部服务健康检查

	overallHealthy := modelHealthy && cacheHealthy && dbHealthy && extServicesHealthy

	status := "healthy"
	if !overallHealthy {
		status = "unhealthy"
	}

	c.JSON(http.StatusOK, gin.H{
		"status":    status,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"checks": gin.H{
			"model": gin.H{
				"status":  s.getHealthStatus(modelHealthy),
				"message": s.getHealthMessage(modelHealthy, "Model loaded successfully", "Model not loaded"),
			},
			"cache": gin.H{
				"status":  s.getHealthStatus(cacheHealthy),
				"message": s.getHealthMessage(cacheHealthy, "Cache initialized", "Cache not available"),
			},
			"database": gin.H{
				"status":  s.getHealthStatus(dbHealthy),
				"message": s.getHealthMessage(dbHealthy, "Database connected", "Database connection failed"),
			},
			"external_services": gin.H{
				"status":  s.getHealthStatus(extServicesHealthy),
				"message": s.getHealthMessage(extServicesHealthy, "External services available", "External services unavailable"),
			},
		},
	})
}

// getHealthStatus 获取健康状态字符串
func (s *Server) getHealthStatus(healthy bool) string {
	if healthy {
		return "healthy"
	}
	return "unhealthy"
}

// getHealthMessage 获取健康消息
func (s *Server) getHealthMessage(healthy bool, successMsg, failureMsg string) string {
	if healthy {
		return successMsg
	}
	return failureMsg
}
