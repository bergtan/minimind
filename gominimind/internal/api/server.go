package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"gominimind/internal/cache"
	"gominimind/pkg/config"
	"gominimind/pkg/model"
	"gominimind/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
	"golang.org/x/time/rate"
)

// Server API服务器结构体
type Server struct {
	Config      *config.Config
	Model       *model.MiniMindModel
	Cache       cache.Cache
	Logger      *logrus.Logger
	RateLimiter *sync.Map // IP地址 -> 限流器
	ServerStats *ServerStats
	mu          sync.RWMutex
}

// ServerStats 服务器统计信息
type ServerStats struct {
	StartTime       time.Time
	TotalRequests   int64
	FailedRequests  int64
	TotalTokens     int64
	CacheHits       int64
	CacheMisses     int64
	AvgResponseTime time.Duration
	mu              sync.RWMutex
}

// NewServer 创建新的API服务器
func NewServer(cfg *config.Config, model *model.MiniMindModel) (*Server, error) {
	logger := logrus.New()

	// 初始化缓存
	var cacheImpl cache.Cache
	if cfg.Server.UseCache {
		cacheImpl = cache.NewMemoryCache(int64(cfg.Server.CacheSize), time.Duration(cfg.Cache.Expiration)*time.Second, logger)
	} else {
		cacheImpl = cache.NewMemoryCache(0, time.Minute, logger)
	}

	// 初始化统计信息
	stats := &ServerStats{
		StartTime: time.Now(),
	}

	server := &Server{
		Config:      cfg,
		Model:       model,
		Cache:       cacheImpl,
		Logger:      logger,
		RateLimiter: &sync.Map{},
		ServerStats: stats,
	}

	return server, nil
}

// SetupRouter 创建并配置路由器
func (s *Server) SetupRouter() *gin.Engine {
	router := gin.New()
	router.Use(gin.Recovery())

	// 健康检查（不需要认证）
	router.GET("/health", s.HealthCheck)
	router.GET("/health/detailed", s.DetailedHealthCheck)

	// OpenAI兼容API
	v1 := router.Group("/v1")
	{
		v1.Use(s.AuthMiddleware())
		v1.Use(s.RateLimitMiddleware())
		v1.POST("/chat/completions", s.ChatCompletion)
		v1.POST("/completions", s.Completion)
		v1.POST("/embeddings", s.Embedding)
		v1.GET("/models", s.ListModels)
		v1.GET("/models/:model", s.GetModel)
	}

	// 自定义API
	api := router.Group("/api/v1")
	{
		api.Use(s.AuthMiddleware())
		api.GET("/models", s.ListModels)
		api.GET("/models/:model", s.GetModel)
		api.GET("/model/info", s.GetModelInfo)
		api.POST("/batch/chat", s.BatchChat)
		api.POST("/batch/embedding", s.BatchEmbedding)
	}

	// 监控与文档
	router.GET("/metrics", s.Metrics)
	router.GET("/stats", s.Stats)
	router.GET("/docs", s.Docs)

	// 训练接口
	router.POST("/api/train/start", s.HandleTrainStart)
	router.POST("/api/train/stop", s.HandleTrainStop)
	router.GET("/api/train/status", s.HandleTrainStatus)
	router.GET("/api/train/logs", s.HandleTrainLogs)

	return router
}

// ========== OpenAI兼容接口 ==========

// ChatCompletion 聊天补全接口
func (s *Server) ChatCompletion(c *gin.Context) {
	var req types.ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	// 验证请求参数
	if err := s.validateChatRequest(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid request parameters", err.Error())
		return
	}

	// 检查缓存
	cacheKey := s.generateCacheKey("chat", req)
	if s.Config.Server.UseCache {
		if cached, err := s.Cache.Get(c.Request.Context(), cacheKey); err == nil && cached != nil {
			s.ServerStats.mu.Lock()
			s.ServerStats.CacheHits++
			s.ServerStats.mu.Unlock()

			c.JSON(http.StatusOK, cached)
			return
		}
	}

	startTime := time.Now()

	// 处理流式请求
	if req.Stream {
		s.handleStreamingChat(c, &req)
		return
	}

	// 生成响应
	response, err := s.generateChatResponse(&req)
	if err != nil {
		s.sendError(c, http.StatusInternalServerError, "internal_error", "Failed to generate response", err.Error())
		return
	}

	// 缓存结果
	if s.Config.Server.UseCache {
		respBytes, _ := json.Marshal(response)
		s.Cache.Set(c.Request.Context(), cacheKey, respBytes, time.Hour)
		s.ServerStats.mu.Lock()
		s.ServerStats.CacheMisses++
		s.ServerStats.mu.Unlock()
	}

	// 更新统计信息
	s.updateStats(time.Since(startTime), len(response.Choices[0].Message.Content))

	c.JSON(http.StatusOK, response)
}

// Completion 文本补全接口
func (s *Server) Completion(c *gin.Context) {
	var req types.CompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	// 验证请求参数
	if req.Prompt == "" {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Missing required parameter", "prompt is required")
		return
	}

	// 生成响应
	response, err := s.generateCompletionResponse(&req)
	if err != nil {
		s.sendError(c, http.StatusInternalServerError, "internal_error", "Failed to generate completion", err.Error())
		return
	}

	c.JSON(http.StatusOK, response)
}

// Embedding 嵌入接口
func (s *Server) Embedding(c *gin.Context) {
	var req types.EmbeddingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	// 验证请求参数
	if len(req.Input) == 0 {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Missing required parameter", "input is required")
		return
	}

	// 生成嵌入向量（取第一个输入）
	embeddingsF32, err := s.Model.GenerateEmbedding(req.Input[0])
	if err != nil {
		s.sendError(c, http.StatusInternalServerError, "internal_error", "Failed to generate embedding", err.Error())
		return
	}

	// 转换为float64
	embeddingsF64 := make([]float64, len(embeddingsF32))
	for i, v := range embeddingsF32 {
		embeddingsF64[i] = float64(v)
	}

	promptTokens, _ := s.Model.Tokenize(req.Input[0])

	response := types.EmbeddingResponse{
		Object: "list",
		Data: []types.EmbeddingData{
			{
				Object:    "embedding",
				Embedding: embeddingsF64,
				Index:     0,
			},
		},
		Model: req.Model,
		Usage: types.Usage{
			PromptTokens:     len(promptTokens),
			CompletionTokens: 0,
			TotalTokens:      len(promptTokens),
		},
	}

	c.JSON(http.StatusOK, response)
}

// ListModels 获取模型列表
func (s *Server) ListModels(c *gin.Context) {
	models := []types.ModelInfo{
		{
			ID:         "minimind",
			Object:     "model",
			Created:    time.Now().Unix(),
			OwnedBy:    "gominimind",
			Permission: []interface{}{},
			Root:       "minimind",
			Parent:     nil,
		},
	}

	response := types.ModelsListResponse{
		Object: "list",
		Data:   models,
	}

	c.JSON(http.StatusOK, response)
}

// GetModel 获取模型详情
func (s *Server) GetModel(c *gin.Context) {
	modelID := c.Param("model_id")

	if modelID != "minimind" {
		s.sendError(c, http.StatusNotFound, "model_not_found", "Model not found", fmt.Sprintf("Model %s not found", modelID))
		return
	}

	modelInfo := types.ModelInfo{
		ID:         "minimind",
		Object:     "model",
		Created:    time.Now().Unix(),
		OwnedBy:    "gominimind",
		Permission: []interface{}{},
		Root:       "minimind",
		Parent:     nil,
	}

	c.JSON(http.StatusOK, modelInfo)
}

// ========== 自定义API接口 ==========

// HealthCheck 健康检查
func (s *Server) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":       "healthy",
		"timestamp":    time.Now().UTC().Format(time.RFC3339),
		"version":      "2.0.0",
		"model_loaded": s.Model != nil,
	})
}

// DetailedHealthCheck 详细健康检查
func (s *Server) DetailedHealthCheck(c *gin.Context) {
	stats := s.getStats()

	c.JSON(http.StatusOK, gin.H{
		"status":            "healthy",
		"timestamp":         time.Now().UTC().Format(time.RFC3339),
		"version":           "2.0.0",
		"model_loaded":      s.Model != nil,
		"uptime":            time.Since(stats.StartTime).String(),
		"total_requests":    stats.TotalRequests,
		"failed_requests":   stats.FailedRequests,
		"cache_hits":        stats.CacheHits,
		"cache_misses":      stats.CacheMisses,
		"avg_response_time": stats.AvgResponseTime.String(),
	})
}

// GetModelInfo 获取模型详细信息
func (s *Server) GetModelInfo(c *gin.Context) {
	modelID := c.Param("id")

	if modelID != "minimind" {
		s.sendError(c, http.StatusNotFound, "model_not_found", "Model not found", fmt.Sprintf("Model %s not found", modelID))
		return
	}

	modelInfo := gin.H{
		"id":                 "minimind",
		"name":               "MiniMind",
		"description":        "轻量级语言模型",
		"version":            "2.0",
		"context_length":     s.Config.Model.MaxPositionEmbeddings,
		"parameters":         "25.8M",
		"supported_features": []string{"chat", "completion", "embedding"},
		"config":             s.Config.Model,
	}

	c.JSON(http.StatusOK, modelInfo)
}

// BatchChat 批量聊天
func (s *Server) BatchChat(c *gin.Context) {
	var req types.BatchChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	if len(req.Requests) == 0 {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Missing requests", "requests array is required")
		return
	}

	if req.Parallel < 1 {
		req.Parallel = 1
	}
	if req.Parallel > 10 {
		req.Parallel = 10
	}

	// 并行处理请求
	results := make([]types.ChatCompletionResponse, len(req.Requests))
	errors := make([]error, len(req.Requests))

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, req.Parallel)

	for i, chatReq := range req.Requests {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(index int, request types.ChatCompletionRequest) {
			defer wg.Done()
			defer func() { <-semaphore }()

			response, err := s.generateChatResponse(&request)
			if err != nil {
				errors[index] = err
			} else {
				results[index] = *response
			}
		}(i, chatReq)
	}

	wg.Wait()

	// 检查是否有错误
	var hasError bool
	for _, err := range errors {
		if err != nil {
			hasError = true
			break
		}
	}

	if hasError {
		c.JSON(http.StatusMultiStatus, gin.H{
			"results": results,
			"errors":  errors,
		})
	} else {
		c.JSON(http.StatusOK, gin.H{
			"results": results,
		})
	}
}

// BatchEmbedding 批量嵌入
func (s *Server) BatchEmbedding(c *gin.Context) {
	var req types.BatchEmbeddingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON format", err.Error())
		return
	}

	if len(req.Texts) == 0 {
		s.sendError(c, http.StatusBadRequest, "invalid_request_error", "Missing texts", "texts array is required")
		return
	}

	embeddings := make([][]float32, len(req.Texts))
	for i, text := range req.Texts {
		embedding, err := s.Model.GenerateEmbedding(text)
		if err != nil {
			s.sendError(c, http.StatusInternalServerError, "internal_error", "Failed to generate embedding", err.Error())
			return
		}
		embeddings[i] = embedding
	}

	c.JSON(http.StatusOK, gin.H{
		"embeddings": embeddings,
		"count":      len(embeddings),
	})
}

// Metrics 监控指标
func (s *Server) Metrics(c *gin.Context) {
	stats := s.getStats()

	c.JSON(http.StatusOK, gin.H{
		"metrics": gin.H{
			"requests_total":         stats.TotalRequests,
			"requests_failed":        stats.FailedRequests,
			"cache_hits":             stats.CacheHits,
			"cache_misses":           stats.CacheMisses,
			"cache_hit_rate":         s.calculateHitRate(stats),
			"avg_response_time_ms":   stats.AvgResponseTime.Milliseconds(),
			"uptime_seconds":         time.Since(stats.StartTime).Seconds(),
			"total_tokens_processed": stats.TotalTokens,
		},
	})
}

// GetStats 获取统计信息
func (s *Server) Stats(c *gin.Context) {
	stats := s.getStats()

	c.JSON(http.StatusOK, gin.H{
		"server": gin.H{
			"start_time": stats.StartTime.Format(time.RFC3339),
			"uptime":     time.Since(stats.StartTime).String(),
			"version":    "2.0.0",
		},
		"requests": gin.H{
			"total":        stats.TotalRequests,
			"failed":       stats.FailedRequests,
			"success":      stats.TotalRequests - stats.FailedRequests,
			"success_rate": s.calculateSuccessRate(stats),
		},
		"cache": gin.H{
			"hits":     stats.CacheHits,
			"misses":   stats.CacheMisses,
			"hit_rate": s.calculateHitRate(stats),
		},
		"performance": gin.H{
			"avg_response_time": stats.AvgResponseTime.String(),
			"tokens_per_second": s.calculateTokensPerSecond(stats),
		},
	})
}

// Docs Web聊天界面
func (s *Server) Docs(c *gin.Context) {
	htmlPath := findIndexHTML()
	if htmlPath != "" {
		c.File(htmlPath)
		return
	}
	// 找不到index.html时降级为简单提示页
	c.Header("Content-Type", "text/html; charset=utf-8")
	c.String(http.StatusOK, `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>GoMiniMind</title></head><body>
<h2>GoMiniMind API 服务运行中</h2>
<p>Web界面文件未找到，请启动 <code>web_demo</code> 服务（端口7860）访问完整界面。</p>
<p>API端点：<a href="/v1/chat/completions">/v1/chat/completions</a></p>
</body></html>`)
}

// findIndexHTML 查找index.html文件路径
func findIndexHTML() string {
	candidates := []string{}

	// 1. 相对于可执行文件
	if execPath, err := os.Executable(); err == nil {
		candidates = append(candidates,
			filepath.Join(filepath.Dir(execPath), "templates", "index.html"),
			filepath.Join(filepath.Dir(execPath), "..", "web_demo", "templates", "index.html"),
		)
	}

	// 2. 相对于当前工作目录
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates,
			filepath.Join(cwd, "cmd", "web_demo", "templates", "index.html"),
			filepath.Join(cwd, "templates", "index.html"),
		)
	}

	// 3. 相对于源文件位置
	if _, filename, _, ok := runtime.Caller(0); ok {
		// internal/api/server.go -> 项目根目录
		root := filepath.Join(filepath.Dir(filename), "..", "..")
		candidates = append(candidates,
			filepath.Join(root, "cmd", "web_demo", "templates", "index.html"),
		)
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// ========== 中间件 ==========

// LoggerMiddleware 日志中间件
func (s *Server) LoggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery

		c.Next()

		end := time.Now()
		latency := end.Sub(start)
		statusCode := c.Writer.Status()
		clientIP := c.ClientIP()
		method := c.Request.Method
		userAgent := c.Request.UserAgent()

		s.Logger.Infof("%s %s %d %v | %s | %s | %s",
			method, path, statusCode, latency, clientIP, query, userAgent)
	}
}

// CORSMiddleware CORS中间件
func (s *Server) CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

// RateLimitMiddleware 限流中间件
func (s *Server) RateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 训练日志SSE与训练状态查询可能是长连接/高频请求，避免被全局限流误伤
		path := c.Request.URL.Path
		if path == "/api/train/logs" || path == "/api/train/status" {
			c.Next()
			return
		}

		clientIP := c.ClientIP()

		// 获取或创建限流器
		limiter, _ := s.RateLimiter.LoadOrStore(clientIP, rate.NewLimiter(
			rate.Limit(s.Config.Server.RateLimit/60), // 转换为每秒
			s.Config.Server.RateLimit/60,
		))

		if !limiter.(*rate.Limiter).Allow() {
			s.sendError(c, http.StatusTooManyRequests, "rate_limit_exceeded",
				"Rate limit exceeded", "Too many requests")
			c.Abort()
			return
		}

		c.Next()
	}
}

// AuthMiddleware 认证中间件
func (s *Server) AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 如果配置了API密钥，则进行认证
		if s.Config.Server.APIKey != "" {
			authHeader := c.GetHeader("Authorization")
			if authHeader == "" {
				s.sendError(c, http.StatusUnauthorized, "authentication_error",
					"Missing API key", "Authorization header is required")
				c.Abort()
				return
			}

			if !strings.HasPrefix(authHeader, "Bearer ") {
				s.sendError(c, http.StatusUnauthorized, "authentication_error",
					"Invalid API key format", "Bearer token required")
				c.Abort()
				return
			}

			token := strings.TrimPrefix(authHeader, "Bearer ")
			if token != s.Config.Server.APIKey {
				s.sendError(c, http.StatusUnauthorized, "authentication_error",
					"Invalid API key", "API key is invalid")
				c.Abort()
				return
			}
		}

		c.Next()
	}
}

// ========== 辅助方法 ==========

// sendError 发送错误响应
func (s *Server) sendError(c *gin.Context, statusCode int, errorType, message, details string) {
	s.ServerStats.mu.Lock()
	s.ServerStats.FailedRequests++
	s.ServerStats.mu.Unlock()

	errorResponse := types.ErrorResponse{
		Error: types.APIError{
			Code:    statusCode,
			Message: message,
			Type:    errorType,
			Param:   "",
			Details: details,
		},
	}

	c.JSON(statusCode, errorResponse)
}

// validateChatRequest 验证聊天请求
func (s *Server) validateChatRequest(req *types.ChatRequest) error {
	if len(req.Messages) == 0 {
		return fmt.Errorf("messages array cannot be empty")
	}

	if req.MaxTokens < 1 || req.MaxTokens > s.Config.Server.MaxTokens {
		return fmt.Errorf("max_tokens must be between 1 and %d", s.Config.Server.MaxTokens)
	}

	if req.Temperature < 0 || req.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}

	if req.TopP < 0 || req.TopP > 1 {
		return fmt.Errorf("top_p must be between 0 and 1")
	}

	return nil
}

// generateCacheKey 生成缓存键
func (s *Server) generateCacheKey(prefix string, req interface{}) string {
	data, _ := json.Marshal(req)
	return fmt.Sprintf("%s:%s", prefix, string(data))
}

// updateStats 更新统计信息
func (s *Server) updateStats(responseTime time.Duration, tokens int) {
	s.ServerStats.mu.Lock()
	defer s.ServerStats.mu.Unlock()

	s.ServerStats.TotalRequests++
	s.ServerStats.TotalTokens += int64(tokens)

	// 更新平均响应时间
	if s.ServerStats.TotalRequests == 1 {
		s.ServerStats.AvgResponseTime = responseTime
	} else {
		s.ServerStats.AvgResponseTime = time.Duration(
			(int64(s.ServerStats.AvgResponseTime)*s.ServerStats.TotalRequests + int64(responseTime)) / (s.ServerStats.TotalRequests + 1),
		)
	}
}

// getStats 获取统计信息副本
func (s *Server) getStats() *ServerStats {
	return s.ServerStats
}

// calculateHitRate 计算缓存命中率
func (s *Server) calculateHitRate(stats *ServerStats) float64 {
	total := stats.CacheHits + stats.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(stats.CacheHits) / float64(total)
}

// calculateSuccessRate 计算请求成功率
func (s *Server) calculateSuccessRate(stats *ServerStats) float64 {
	if stats.TotalRequests == 0 {
		return 0
	}
	return float64(stats.TotalRequests-stats.FailedRequests) / float64(stats.TotalRequests)
}

// calculateTokensPerSecond 计算每秒处理token数
func (s *Server) calculateTokensPerSecond(stats *ServerStats) float64 {
	uptime := time.Since(stats.StartTime).Seconds()
	if uptime == 0 {
		return 0
	}
	return float64(stats.TotalTokens) / uptime
}

// generateChatResponse 生成聊天响应
func (s *Server) generateChatResponse(req *types.ChatRequest) (*types.ChatResponse, error) {
	// 构建提示文本
	prompt := s.buildPrompt(req.Messages)

	// 生成文本
	generatedText, err := s.Model.Generate(prompt, req.MaxTokens, req.Temperature, req.TopP)
	if err != nil {
		return nil, err
	}

	// 计算token使用量
	promptTokens, _ := s.Model.Tokenize(prompt)
	completionTokens, _ := s.Model.Tokenize(generatedText)

	response := &types.ChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []types.Choice{
			{
				Index: 0,
				Message: types.Message{
					Role:    "assistant",
					Content: generatedText,
				},
				FinishReason: "stop",
			},
		},
		Usage: types.Usage{
			PromptTokens:     len(promptTokens),
			CompletionTokens: len(completionTokens),
			TotalTokens:      len(promptTokens) + len(completionTokens),
		},
	}

	return response, nil
}

// generateCompletionResponse 生成补全响应
func (s *Server) generateCompletionResponse(req *types.CompletionRequest) (*types.CompletionResponse, error) {
	generatedText, err := s.Model.Generate(req.Prompt, req.MaxTokens, req.Temperature, req.TopP)
	if err != nil {
		return nil, err
	}

	promptTokens, _ := s.Model.Tokenize(req.Prompt)
	completionTokens, _ := s.Model.Tokenize(generatedText)

	response := &types.CompletionResponse{
		ID:      fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []types.CompletionChoice{
			{
				Text:         generatedText,
				Index:        0,
				LogProbs:     nil,
				FinishReason: "length",
			},
		},
		Usage: types.Usage{
			PromptTokens:     len(promptTokens),
			CompletionTokens: len(completionTokens),
			TotalTokens:      len(promptTokens) + len(completionTokens),
		},
	}

	return response, nil
}

// buildPrompt 构建提示文本
func (s *Server) buildPrompt(messages []types.Message) string {
	var prompt strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			prompt.WriteString("<|system|>\n")
			prompt.WriteString(msg.Content)
			prompt.WriteString("<|end|>\n")
		case "user":
			prompt.WriteString("<|user|>\n")
			prompt.WriteString(msg.Content)
			prompt.WriteString("<|end|>\n")
		case "assistant":
			prompt.WriteString("<|assistant|>\n")
			prompt.WriteString(msg.Content)
			prompt.WriteString("<|end|>\n")
		}
	}

	prompt.WriteString("<|assistant|>\n")
	return prompt.String()
}

// handleStreamingChat 处理流式聊天请求
func (s *Server) handleStreamingChat(c *gin.Context, req *types.ChatRequest) {
	// 设置流式响应头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	// 构建提示
	prompt := s.buildPrompt(req.Messages)

	// 流式生成文本
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		s.sendError(c, http.StatusInternalServerError, "internal_error",
			"Streaming not supported", "Response writer does not support streaming")
		return
	}

	chunkID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	// 发送初始块
	c.Writer.Write([]byte(fmt.Sprintf("data: %s\n\n",
		string(s.createStreamChunk(chunkID, "", "assistant", nil)))))
	flusher.Flush()

	// 模拟流式生成（实际实现需要模型支持）
	generatedText, err := s.Model.Generate(prompt, req.MaxTokens, req.Temperature, req.TopP)
	if err != nil {
		c.Writer.Write([]byte(fmt.Sprintf("data: %s\n\n",
			string(s.createErrorChunk(err)))))
		c.Writer.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
		return
	}

	// 分块发送响应
	for i, char := range generatedText {
		chunk := s.createStreamChunk(chunkID, string(char), "", nil)
		c.Writer.Write([]byte(fmt.Sprintf("data: %s\n\n", string(chunk))))
		flusher.Flush()

		// 添加小延迟以模拟流式效果
		time.Sleep(50 * time.Millisecond)

		if i >= req.MaxTokens-1 {
			break
		}
	}

	// 发送结束块
	c.Writer.Write([]byte(fmt.Sprintf("data: %s\n\n",
		string(s.createStreamChunk(chunkID, "", "", "stop")))))
	c.Writer.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

// createStreamChunk 创建流式响应块
func (s *Server) createStreamChunk(id, content, role, finishReason interface{}) []byte {
	chunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   "minimind",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{
					"role":    role,
					"content": content,
				},
				"finish_reason": finishReason,
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// createErrorChunk 创建错误块
func (s *Server) createErrorChunk(err error) []byte {
	errorChunk := map[string]interface{}{
		"error": map[string]interface{}{
			"message": err.Error(),
			"type":    "internal_error",
		},
	}

	data, _ := json.Marshal(errorChunk)
	return data
}
