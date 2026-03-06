package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
	"golang.org/x/time/rate"
)

// ========== 认证中间件 ==========

// AuthConfig 认证配置
type AuthConfig struct {
	APIKey         string   `json:"api_key"`
	APIKeys        []string `json:"api_keys"`
	RequireAuth    bool     `json:"require_auth"`
	AllowPublic    bool     `json:"allow_public"`
	IPWhitelist    []string `json:"ip_whitelist"`
	AllowedOrigins []string `json:"allowed_origins"`
	AllowedMethods []string `json:"allowed_methods"`
	AllowedHeaders []string `json:"allowed_headers"`
}

// AuthMiddleware 认证中间件
func AuthMiddleware(config *AuthConfig, logger *logrus.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 检查IP白名单
		if len(config.IPWhitelist) > 0 {
			clientIP := c.ClientIP()
			allowed := false

			for _, ip := range config.IPWhitelist {
				if ip == clientIP || ip == "*" {
					allowed = true
					break
				}
			}

			if !allowed {
				logger.Warnf("IP address not allowed: %s", clientIP)
				c.JSON(http.StatusForbidden, gin.H{
					"error": "IP address not allowed",
					"code":  403,
				})
				c.Abort()
				return
			}
		}

		// 检查认证要求
		if !config.RequireAuth {
			c.Next()
			return
		}

		// 获取API密钥
		authHeader := c.GetHeader("Authorization")
		apiKey := ""

		if authHeader != "" {
			parts := strings.Split(authHeader, " ")
			if len(parts) == 2 && parts[0] == "Bearer" {
				apiKey = parts[1]
			}
		}

		// 检查API密钥
		if apiKey == "" {
			// 尝试从查询参数获取
			apiKey = c.Query("api_key")
		}

		// 验证API密钥
		valid := false

		if config.APIKey != "" && apiKey == config.APIKey {
			valid = true
		} else if len(config.APIKeys) > 0 {
			for _, key := range config.APIKeys {
				if apiKey == key {
					valid = true
					break
				}
			}
		} else if config.AllowPublic && apiKey == "" {
			// 允许公开访问
			valid = true
		}

		if !valid {
			logger.Warnf("Invalid API key: %s", apiKey)
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "Invalid API key",
				"code":  401,
			})
			c.Abort()
			return
		}

		// 设置用户上下文
		c.Set("api_key", apiKey)
		c.Set("client_ip", c.ClientIP())

		logger.Debugf("Authenticated request from %s", c.ClientIP())
		c.Next()
	}
}

// ========== 限流中间件 ==========

// RateLimitConfig 限流配置
type RateLimitConfig struct {
	Enabled     bool          `json:"enabled"`
	Requests    int           `json:"requests"`
	Window      time.Duration `json:"window"`
	Burst       int           `json:"burst"`
	ByIP        bool          `json:"by_ip"`
	ByAPIKey    bool          `json:"by_api_key"`
	ExemptPaths []string      `json:"exempt_paths"`
}

// RateLimiter 限流器
type RateLimiter struct {
	limiters map[string]*rate.Limiter
	mutex    sync.RWMutex
	config   *RateLimitConfig
	logger   *logrus.Logger
}

// NewRateLimiter 创建限流器
func NewRateLimiter(config *RateLimitConfig, logger *logrus.Logger) *RateLimiter {
	if config == nil {
		config = &RateLimitConfig{
			Enabled:  true,
			Requests: 100,
			Window:   time.Minute,
			Burst:    50,
			ByIP:     true,
		}
	}

	return &RateLimiter{
		limiters: make(map[string]*rate.Limiter),
		config:   config,
		logger:   logger,
	}
}

// GetLimiter 获取限流器
func (rl *RateLimiter) GetLimiter(key string) *rate.Limiter {
	rl.mutex.RLock()
	limiter, exists := rl.limiters[key]
	rl.mutex.RUnlock()

	if exists {
		return limiter
	}

	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	// 双重检查
	limiter, exists = rl.limiters[key]
	if exists {
		return limiter
	}

	// 创建新的限流器
	limiter = rate.NewLimiter(rate.Every(rl.config.Window/time.Duration(rl.config.Requests)), rl.config.Burst)
	rl.limiters[key] = limiter

	return limiter
}

// Cleanup 清理过期限流器
func (rl *RateLimiter) Cleanup() {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	// 简单实现：定期清理所有限流器
	rl.limiters = make(map[string]*rate.Limiter)
	rl.logger.Info("Rate limiter cache cleaned up")
}

// RateLimitMiddleware 限流中间件
func RateLimitMiddleware(limiter *RateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !limiter.config.Enabled {
			c.Next()
			return
		}

		// 检查豁免路径
		for _, path := range limiter.config.ExemptPaths {
			if c.Request.URL.Path == path {
				c.Next()
				return
			}
		}

		// 确定限流键
		key := ""

		if limiter.config.ByAPIKey {
			if apiKey, exists := c.Get("api_key"); exists {
				key = fmt.Sprintf("api_key:%s", apiKey)
			}
		}

		if key == "" && limiter.config.ByIP {
			key = fmt.Sprintf("ip:%s", c.ClientIP())
		}

		if key == "" {
			key = "global"
		}

		// 获取限流器
		limiterInstance := limiter.GetLimiter(key)

		// 检查限流
		if !limiterInstance.Allow() {
			limiter.logger.Warnf("Rate limit exceeded for key: %s", key)

			c.Header("X-RateLimit-Limit", strconv.Itoa(limiter.config.Requests))
			c.Header("X-RateLimit-Remaining", "0")
			c.Header("X-RateLimit-Reset", fmt.Sprintf("%d", time.Now().Add(limiter.config.Window).Unix()))

			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "Rate limit exceeded",
				"code":        429,
				"message":     "Too many requests, please try again later",
				"retry_after": limiter.config.Window.Seconds(),
			})
			c.Abort()
			return
		}

		// 设置响应头
		c.Header("X-RateLimit-Limit", strconv.Itoa(limiter.config.Requests))
		c.Header("X-RateLimit-Remaining", strconv.Itoa(limiterInstance.Burst()))
		c.Header("X-RateLimit-Reset", fmt.Sprintf("%d", time.Now().Add(limiter.config.Window).Unix()))

		c.Next()
	}
}

// ========== CORS中间件 ==========

// CORSMiddlewareFromAuth 基于AuthConfig的CORS中间件
func CORSMiddlewareFromAuth(config *AuthConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 设置允许的源
		if len(config.AllowedOrigins) > 0 {
			origin := c.Request.Header.Get("Origin")
			allowed := false

			for _, allowedOrigin := range config.AllowedOrigins {
				if allowedOrigin == "*" || allowedOrigin == origin {
					c.Header("Access-Control-Allow-Origin", allowedOrigin)
					allowed = true
					break
				}
			}

			if !allowed && origin != "" {
				c.Header("Access-Control-Allow-Origin", "")
			}
		} else {
			c.Header("Access-Control-Allow-Origin", "*")
		}

		// 设置允许的方法
		if len(config.AllowedMethods) > 0 {
			c.Header("Access-Control-Allow-Methods", strings.Join(config.AllowedMethods, ", "))
		} else {
			c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		}

		// 设置允许的头部
		if len(config.AllowedHeaders) > 0 {
			c.Header("Access-Control-Allow-Headers", strings.Join(config.AllowedHeaders, ", "))
		} else {
			c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		}

		// 设置凭据
		c.Header("Access-Control-Allow-Credentials", "true")

		// 处理OPTIONS请求
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

// ========== 日志中间件 ==========

// LoggingConfig 日志配置
type LoggingConfig struct {
	Enabled     bool     `json:"enabled"`
	Level       string   `json:"level"`
	Format      string   `json:"format"`
	IncludeBody bool     `json:"include_body"`
	MaxBodySize int      `json:"max_body_size"`
	SkipPaths   []string `json:"skip_paths"`
}

// LoggingMiddleware 日志中间件
func LoggingMiddleware(config *LoggingConfig, logger *logrus.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !config.Enabled {
			c.Next()
			return
		}

		// 检查跳过路径
		for _, path := range config.SkipPaths {
			if c.Request.URL.Path == path {
				c.Next()
				return
			}
		}

		startTime := time.Now()

		// 记录请求信息
		clientIP := c.ClientIP()
		method := c.Request.Method
		path := c.Request.URL.Path
		userAgent := c.Request.UserAgent()

		// 记录请求体（如果启用）
		var requestBody string
		if config.IncludeBody && c.Request.Body != nil {
			bodyBytes, err := io.ReadAll(c.Request.Body)
			if err == nil {
				// 限制请求体大小
				if len(bodyBytes) > config.MaxBodySize {
					requestBody = string(bodyBytes[:config.MaxBodySize]) + "..."
				} else {
					requestBody = string(bodyBytes)
				}

				// 恢复请求体
				c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			}
		}

		// 处理响应
		c.Next()

		// 计算处理时间
		latency := time.Since(startTime)

		// 获取响应信息
		statusCode := c.Writer.Status()
		responseSize := c.Writer.Size()

		// 创建日志条目
		fields := logrus.Fields{
			"client_ip":     clientIP,
			"method":        method,
			"path":          path,
			"status":        statusCode,
			"latency":       latency.String(),
			"user_agent":    userAgent,
			"response_size": responseSize,
		}

		// 添加请求体（如果启用）
		if requestBody != "" {
			fields["request_body"] = requestBody
		}

		// 添加API密钥（如果存在）
		if apiKey, exists := c.Get("api_key"); exists {
			fields["api_key"] = apiKey
		}

		// 根据状态码选择日志级别
		entry := logger.WithFields(fields)

		if statusCode >= 500 {
			entry.Error("Server error")
		} else if statusCode >= 400 {
			entry.Warn("Client error")
		} else {
			entry.Info("Request processed")
		}
	}
}

// ========== 恢复中间件 ==========

// RecoveryMiddleware 恢复中间件
func RecoveryMiddleware(logger *logrus.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				// 记录panic信息
				logger.Errorf("Panic recovered: %v", err)

				// 返回错误响应
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":   "Internal server error",
					"code":    500,
					"message": "An unexpected error occurred",
				})

				c.Abort()
			}
		}()

		c.Next()
	}
}

// ========== 缓存中间件 ==========

// CacheConfig 缓存配置
type CacheConfig struct {
	Enabled     bool          `json:"enabled"`
	Duration    time.Duration `json:"duration"`
	Store       string        `json:"store"`
	SkipPaths   []string      `json:"skip_paths"`
	OnlyMethods []string      `json:"only_methods"`
}

// CacheMiddleware 缓存中间件
func CacheMiddleware(config *CacheConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !config.Enabled {
			c.Next()
			return
		}

		// 检查跳过路径
		for _, path := range config.SkipPaths {
			if c.Request.URL.Path == path {
				c.Next()
				return
			}
		}

		// 检查方法限制
		if len(config.OnlyMethods) > 0 {
			allowed := false
			for _, method := range config.OnlyMethods {
				if c.Request.Method == method {
					allowed = true
					break
				}
			}
			if !allowed {
				c.Next()
				return
			}
		}

		// 设置缓存头
		if config.Duration > 0 {
			c.Header("Cache-Control", fmt.Sprintf("public, max-age=%d", int(config.Duration.Seconds())))
			c.Header("Expires", time.Now().Add(config.Duration).Format(time.RFC1123))
		}

		c.Next()
	}
}

// ========== 压缩中间件 ==========

// CompressionMiddleware 压缩中间件
func CompressionMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 检查客户端是否支持压缩
		acceptEncoding := c.Request.Header.Get("Accept-Encoding")

		if strings.Contains(acceptEncoding, "gzip") {
			c.Header("Content-Encoding", "gzip")
		} else if strings.Contains(acceptEncoding, "deflate") {
			c.Header("Content-Encoding", "deflate")
		}

		c.Next()
	}
}

// ========== 请求ID中间件 ==========

// RequestIDMiddleware 请求ID中间件
func RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 优先使用已有的请求ID
		requestID := c.GetHeader("X-Request-ID")
		if requestID == "" {
			// 生成新请求ID
			requestID = generateRequestID()
		}

		// 设置请求ID到上下文和响应头
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)

		c.Next()
	}
}

// generateRequestID 生成请求ID
func generateRequestID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// ========== 指标中间件 ==========

// MetricsConfig 指标配置
type MetricsConfig struct {
	Enabled    bool     `json:"enabled"`
	Path       string   `json:"path"`
	Collectors []string `json:"collectors"`
}

// MetricsMiddleware 指标中间件
func MetricsMiddleware(config *MetricsConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !config.Enabled {
			c.Next()
			return
		}

		startTime := time.Now()

		c.Next()

		// 记录指标
		latency := time.Since(startTime).Seconds()
		statusCode := c.Writer.Status()

		// 这里可以集成Prometheus或其他指标系统
		// 简化实现：记录到日志
		c.Set("metrics_latency", latency)
		c.Set("metrics_status", statusCode)
	}
}

// ========== 中间件管理器 ==========

// MiddlewareManager 中间件管理器
type MiddlewareManager struct {
	AuthConfig      *AuthConfig
	RateLimitConfig *RateLimitConfig
	LoggingConfig   *LoggingConfig
	CacheConfig     *CacheConfig
	MetricsConfig   *MetricsConfig
	Logger          *logrus.Logger
	RateLimiter     *RateLimiter
}

// NewMiddlewareManager 创建中间件管理器
func NewMiddlewareManager(logger *logrus.Logger) *MiddlewareManager {
	authConfig := &AuthConfig{
		RequireAuth:    true,
		AllowPublic:    false,
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"Content-Type", "Authorization"},
	}

	rateLimitConfig := &RateLimitConfig{
		Enabled:  true,
		Requests: 100,
		Window:   time.Minute,
		Burst:    50,
		ByIP:     true,
	}

	loggingConfig := &LoggingConfig{
		Enabled:     true,
		Level:       "info",
		Format:      "json",
		IncludeBody: false,
		MaxBodySize: 1024,
		SkipPaths:   []string{"/health", "/metrics"},
	}

	cacheConfig := &CacheConfig{
		Enabled:     false,
		Duration:    5 * time.Minute,
		SkipPaths:   []string{"/api/v1/chat/completions"},
		OnlyMethods: []string{"GET"},
	}

	metricsConfig := &MetricsConfig{
		Enabled:    true,
		Path:       "/metrics",
		Collectors: []string{"requests", "latency", "errors"},
	}

	return &MiddlewareManager{
		AuthConfig:      authConfig,
		RateLimitConfig: rateLimitConfig,
		LoggingConfig:   loggingConfig,
		CacheConfig:     cacheConfig,
		MetricsConfig:   metricsConfig,
		Logger:          logger,
		RateLimiter:     NewRateLimiter(rateLimitConfig, logger),
	}
}

// SetupMiddlewares 设置所有中间件
func (mm *MiddlewareManager) SetupMiddlewares(router *gin.Engine) {
	// 恢复中间件
	router.Use(RecoveryMiddleware(mm.Logger))

	// 请求ID中间件
	router.Use(RequestIDMiddleware())

	// 压缩中间件
	router.Use(CompressionMiddleware())

	// CORS中间件
	router.Use(CORSMiddlewareFromAuth(mm.AuthConfig))

	// 认证中间件
	router.Use(AuthMiddleware(mm.AuthConfig, mm.Logger))

	// 限流中间件
	router.Use(RateLimitMiddleware(mm.RateLimiter))

	// 缓存中间件
	router.Use(CacheMiddleware(mm.CacheConfig))

	// 日志中间件
	router.Use(LoggingMiddleware(mm.LoggingConfig, mm.Logger))

	// 指标中间件
	router.Use(MetricsMiddleware(mm.MetricsConfig))
}

// UpdateConfig 更新配置
func (mm *MiddlewareManager) UpdateConfig(config map[string]interface{}) error {
	configJSON, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	var newConfig struct {
		Auth      *AuthConfig      `json:"auth"`
		RateLimit *RateLimitConfig `json:"rate_limit"`
		Logging   *LoggingConfig   `json:"logging"`
		Cache     *CacheConfig     `json:"cache"`
		Metrics   *MetricsConfig   `json:"metrics"`
	}

	if err := json.Unmarshal(configJSON, &newConfig); err != nil {
		return fmt.Errorf("failed to unmarshal config: %w", err)
	}

	if newConfig.Auth != nil {
		mm.AuthConfig = newConfig.Auth
	}

	if newConfig.RateLimit != nil {
		mm.RateLimitConfig = newConfig.RateLimit
		mm.RateLimiter = NewRateLimiter(newConfig.RateLimit, mm.Logger)
	}

	if newConfig.Logging != nil {
		mm.LoggingConfig = newConfig.Logging
	}

	if newConfig.Cache != nil {
		mm.CacheConfig = newConfig.Cache
	}

	if newConfig.Metrics != nil {
		mm.MetricsConfig = newConfig.Metrics
	}

	mm.Logger.Info("Middleware configuration updated")

	return nil
}

// GetStats 获取统计信息
func (mm *MiddlewareManager) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"auth_enabled":        mm.AuthConfig.RequireAuth,
		"rate_limit_enabled":  mm.RateLimitConfig.Enabled,
		"logging_enabled":     mm.LoggingConfig.Enabled,
		"cache_enabled":       mm.CacheConfig.Enabled,
		"metrics_enabled":     mm.MetricsConfig.Enabled,
		"rate_limiters_count": len(mm.RateLimiter.limiters),
	}
}

// Cleanup 清理资源
func (mm *MiddlewareManager) Cleanup() {
	if mm.RateLimiter != nil {
		mm.RateLimiter.Cleanup()
	}

	mm.Logger.Info("Middleware manager cleaned up")
}

// ========== CORS 中间件 ==========

// CORSConfig CORS配置
type CORSConfig struct {
	AllowOrigins     []string `json:"allow_origins"`
	AllowMethods     []string `json:"allow_methods"`
	AllowHeaders     []string `json:"allow_headers"`
	ExposeHeaders    []string `json:"expose_headers"`
	AllowCredentials bool     `json:"allow_credentials"`
	MaxAge           int      `json:"max_age"`
}

// CORSMiddleware 创建CORS中间件
func CORSMiddleware(config *CORSConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.GetHeader("Origin")

		// 检查是否允许的来源
		allowed := false
		for _, allowedOrigin := range config.AllowOrigins {
			if allowedOrigin == "*" || allowedOrigin == origin {
				allowed = true
				break
			}
		}

		if allowed {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		if config.AllowCredentials {
			c.Header("Access-Control-Allow-Credentials", "true")
		}

		if len(config.AllowMethods) > 0 {
			c.Header("Access-Control-Allow-Methods", strings.Join(config.AllowMethods, ", "))
		}

		if len(config.AllowHeaders) > 0 {
			c.Header("Access-Control-Allow-Headers", strings.Join(config.AllowHeaders, ", "))
		}

		if len(config.ExposeHeaders) > 0 {
			c.Header("Access-Control-Expose-Headers", strings.Join(config.ExposeHeaders, ", "))
		}

		if config.MaxAge > 0 {
			c.Header("Access-Control-Max-Age", strconv.Itoa(config.MaxAge))
		}

		// 处理预检请求
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

// ========== 错误处理中间件 ==========

// ErrorResponse 错误响应结构
type ErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// MiddlewareError 中间件错误
type MiddlewareError struct {
	StatusCode int
	Code       string
	Message    string
}

func (e *MiddlewareError) Error() string {
	return e.Message
}

// InternalServerError 创建内部服务器错误
func InternalServerError(message string) *MiddlewareError {
	return &MiddlewareError{
		StatusCode: http.StatusInternalServerError,
		Code:       "internal_error",
		Message:    message,
	}
}

// ErrorHandlingMiddleware 错误处理中间件
func ErrorHandlingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				c.JSON(http.StatusInternalServerError, ErrorResponse{
					Code:    "internal_error",
					Message: fmt.Sprintf("panic recovered: %v", err),
				})
				c.Abort()
			}
		}()

		c.Next()

		// 处理gin错误
		if len(c.Errors) > 0 {
			lastErr := c.Errors.Last()
			if mErr, ok := lastErr.Err.(*MiddlewareError); ok {
				c.JSON(mErr.StatusCode, ErrorResponse{
					Code:    mErr.Code,
					Message: mErr.Message,
				})
			} else {
				c.JSON(http.StatusInternalServerError, ErrorResponse{
					Code:    "internal_error",
					Message: lastErr.Error(),
				})
			}
		}
	}
}

// ========== IP白名单中间件 ==========

// IPWhitelistConfig IP白名单配置
type IPWhitelistConfig struct {
	Enabled    bool     `json:"enabled"`
	AllowedIPs []string `json:"allowed_ips"`
}

// IPWhitelistMiddleware IP白名单中间件
func IPWhitelistMiddleware(config *IPWhitelistConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !config.Enabled {
			c.Next()
			return
		}

		clientIP := c.ClientIP()
		allowed := false

		for _, allowedIP := range config.AllowedIPs {
			if allowedIP == clientIP || allowedIP == "*" {
				allowed = true
				break
			}
			// 简单CIDR匹配
			if strings.Contains(allowedIP, "/") {
				// 提取CIDR前缀
				parts := strings.Split(allowedIP, "/")
				if len(parts) == 2 {
					prefix := parts[0]
					prefixParts := strings.Split(prefix, ".")
					clientParts := strings.Split(clientIP, ".")
					if len(prefixParts) > 0 && len(clientParts) > 0 {
						if prefixParts[0] == clientParts[0] {
							allowed = true
							break
						}
					}
				}
			}
		}

		if !allowed {
			c.JSON(http.StatusForbidden, gin.H{
				"error": "IP address not allowed",
				"code":  403,
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// ========== 超时中间件 ==========

// TimeoutMiddleware 超时中间件
func TimeoutMiddleware(timeout time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 设置超时时间到上下文
		c.Set("timeout", timeout)

		done := make(chan struct{}, 1)

		go func() {
			c.Next()
			done <- struct{}{}
		}()

		select {
		case <-done:
			// 正常完成
		case <-time.After(timeout):
			c.JSON(http.StatusGatewayTimeout, gin.H{
				"error":   "Request timeout",
				"code":    504,
				"timeout": timeout.String(),
			})
			c.Abort()
		}
	}
}

// ========== 日志辅助函数 ==========

// CreateStructuredLogger 创建结构化日志器
func CreateStructuredLogger(level string) *logrus.Logger {
	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{})

	switch strings.ToLower(level) {
	case "debug":
		logger.SetLevel(logrus.DebugLevel)
	case "info":
		logger.SetLevel(logrus.InfoLevel)
	case "warn", "warning":
		logger.SetLevel(logrus.WarnLevel)
	case "error":
		logger.SetLevel(logrus.ErrorLevel)
	default:
		logger.SetLevel(logrus.InfoLevel)
	}

	return logger
}

// GinLoggerMiddleware 基于logrus的Gin日志中间件
func GinLoggerMiddleware(logger *logrus.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		startTime := time.Now()
		path := c.Request.URL.Path

		c.Next()

		latency := time.Since(startTime)
		statusCode := c.Writer.Status()

		entry := logger.WithFields(logrus.Fields{
			"status":    statusCode,
			"latency":   latency.String(),
			"method":    c.Request.Method,
			"path":      path,
			"client_ip": c.ClientIP(),
		})

		if requestID := c.GetHeader("X-Request-ID"); requestID != "" {
			entry = entry.WithField("request_id", requestID)
		}

		if statusCode >= 500 {
			entry.Error("Server error")
		} else if statusCode >= 400 {
			entry.Warn("Client error")
		} else {
			entry.Info("Request processed")
		}
	}
}
