package monitoring

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
)

// ========== 监控指标定义 ==========

// Metrics 监控指标结构体
type Metrics struct {
	// HTTP请求指标
	RequestsTotal    *prometheus.CounterVec
	RequestDuration  *prometheus.HistogramVec
	RequestSize      *prometheus.HistogramVec
	ResponseSize     *prometheus.HistogramVec
	
	// 模型推理指标
	InferenceTotal   *prometheus.CounterVec
	InferenceDuration *prometheus.HistogramVec
	TokensGenerated  *prometheus.CounterVec
	TokensProcessed  *prometheus.CounterVec
	
	// 缓存指标
	CacheHits        *prometheus.CounterVec
	CacheMisses      *prometheus.CounterVec
	CacheSize        *prometheus.GaugeVec
	
	// 系统指标
	MemoryUsage      *prometheus.GaugeVec
	CPUUsage         *prometheus.GaugeVec
	Goroutines       prometheus.Gauge
	
	// 错误指标
	ErrorsTotal      *prometheus.CounterVec
	
	// 业务指标
	ActiveConnections prometheus.Gauge
	QueueLength      prometheus.Gauge
	
	// 注册表
	registry         *prometheus.Registry
}

// NewMetrics 创建监控指标
func NewMetrics() *Metrics {
	registry := prometheus.NewRegistry()
	
	metrics := &Metrics{
		registry: registry,
		
		// HTTP请求指标
		RequestsTotal: promauto.With(registry).NewCounterVec(
			prometheus.CounterOpts{
				Name: "minimind_http_requests_total",
				Help: "Total number of HTTP requests",
			},
			[]string{"method", "path", "status"},
		),
		
		RequestDuration: promauto.With(registry).NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "minimind_http_request_duration_seconds",
				Help:    "HTTP request duration in seconds",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 10),
			},
			[]string{"method", "path"},
		),
		
		RequestSize: promauto.With(registry).NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "minimind_http_request_size_bytes",
				Help:    "HTTP request size in bytes",
				Buckets: prometheus.ExponentialBuckets(100, 10, 6),
			},
			[]string{"method", "path"},
		),
		
		ResponseSize: promauto.With(registry).NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "minimind_http_response_size_bytes",
				Help:    "HTTP response size in bytes",
				Buckets: prometheus.ExponentialBuckets(100, 10, 6),
			},
			[]string{"method", "path"},
		),
		
		// 模型推理指标
		InferenceTotal: promauto.With(registry).NewCounterVec(
			prometheus.CounterOpts{
				Name: "minimind_inference_total",
				Help: "Total number of model inferences",
			},
			[]string{"model", "type"},
		),
		
		InferenceDuration: promauto.With(registry).NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "minimind_inference_duration_seconds",
				Help:    "Model inference duration in seconds",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 10),
			},
			[]string{"model", "type"},
		),
		
		TokensGenerated: promauto.With(registry).NewCounterVec(
			prometheus.CounterOpts{
				Name: "minimind_tokens_generated_total",
				Help: "Total number of tokens generated",
			},
			[]string{"model"},
		),
		
		TokensProcessed: promauto.With(registry).NewCounterVec(
			prometheus.CounterOpts{
				Name: "minimind_tokens_processed_total",
				Help: "Total number of tokens processed",
			},
			[]string{"model"},
		),
		
		// 缓存指标
		CacheHits: promauto.With(registry).NewCounterVec(
			prometheus.CounterOpts{
				Name: "minimind_cache_hits_total",
				Help: "Total number of cache hits",
			},
			[]string{"cache"},
		),
		
		CacheMisses: promauto.With(registry).NewCounterVec(
			prometheus.CounterOpts{
				Name: "minimind_cache_misses_total",
				Help: "Total number of cache misses",
			},
			[]string{"cache"},
		),
		
		CacheSize: promauto.With(registry).NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "minimind_cache_size_bytes",
				Help: "Cache size in bytes",
			},
			[]string{"cache"},
		),
		
		// 系统指标
		MemoryUsage: promauto.With(registry).NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "minimind_memory_usage_bytes",
				Help: "Memory usage in bytes",
			},
			[]string{"type"},
		),
		
		CPUUsage: promauto.With(registry).NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "minimind_cpu_usage_percent",
				Help: "CPU usage percentage",
			},
			[]string{"type"},
		),
		
		Goroutines: promauto.With(registry).NewGauge(
			prometheus.GaugeOpts{
				Name: "minimind_goroutines_total",
				Help: "Number of goroutines",
			},
		),
		
		// 错误指标
		ErrorsTotal: promauto.With(registry).NewCounterVec(
			prometheus.CounterOpts{
				Name: "minimind_errors_total",
				Help: "Total number of errors",
			},
			[]string{"type", "source"},
		),
		
		// 业务指标
		ActiveConnections: promauto.With(registry).NewGauge(
			prometheus.GaugeOpts{
				Name: "minimind_active_connections",
				Help: "Number of active connections",
			},
		),
		
		QueueLength: promauto.With(registry).NewGauge(
			prometheus.GaugeOpts{
				Name: "minimind_queue_length",
				Help: "Length of processing queue",
			},
		),
	}
	
	return metrics
}

// ========== 监控管理器 ==========

// MonitorManager 监控管理器
type MonitorManager struct {
	metrics     *Metrics
	logger      *logrus.Logger
	server      *http.Server
	isRunning   bool
	mutex       sync.RWMutex
	collectors  []func()
}

// NewMonitorManager 创建监控管理器
func NewMonitorManager(logger *logrus.Logger) *MonitorManager {
	return &MonitorManager{
		metrics:    NewMetrics(),
		logger:     logger,
		collectors: make([]func(), 0),
	}
}

// StartMetricsServer 启动指标服务器
func (mm *MonitorManager) StartMetricsServer(addr string) error {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	if mm.isRunning {
		return fmt.Errorf("metrics server is already running")
	}
	
	// 创建HTTP路由
	router := gin.New()
	router.Use(gin.Recovery())
	
	// 健康检查端点
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"timestamp": time.Now().Format(time.RFC3339),
		})
	})
	
	// 指标端点
	router.GET("/metrics", gin.WrapH(promhttp.HandlerFor(
		mm.metrics.registry,
		promhttp.HandlerOpts{
			EnableOpenMetrics: true,
		},
	)))
	
	// 启动服务器
	mm.server = &http.Server{
		Addr:    addr,
		Handler: router,
	}
	
	go func() {
		mm.logger.Infof("Starting metrics server on %s", addr)
		
		if err := mm.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			mm.logger.Errorf("Failed to start metrics server: %v", err)
		}
	}()
	
	mm.isRunning = true
	
	// 启动指标收集器
	mm.startCollectors()
	
	return nil
}

// StopMetricsServer 停止指标服务器
func (mm *MonitorManager) StopMetricsServer() error {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()
	
	if !mm.isRunning {
		return fmt.Errorf("metrics server is not running")
	}
	
	if mm.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		if err := mm.server.Shutdown(ctx); err != nil {
			return fmt.Errorf("failed to shutdown metrics server: %w", err)
		}
	}
	
	mm.isRunning = false
	mm.logger.Info("Metrics server stopped")
	
	return nil
}

// startCollectors 启动指标收集器
func (mm *MonitorManager) startCollectors() {
	// 系统指标收集器
	go mm.collectSystemMetrics()
	
	// 自定义收集器
	for _, collector := range mm.collectors {
		go collector()
	}
}

// collectSystemMetrics 收集系统指标
func (mm *MonitorManager) collectSystemMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		// 更新goroutine数量
		mm.metrics.Goroutines.Set(float64(getGoroutineCount()))
		
		// 更新内存使用情况
		if memStats, err := getMemoryStats(); err == nil {
			mm.metrics.MemoryUsage.WithLabelValues("alloc").Set(float64(memStats.Alloc))
			mm.metrics.MemoryUsage.WithLabelValues("sys").Set(float64(memStats.Sys))
			mm.metrics.MemoryUsage.WithLabelValues("heap").Set(float64(memStats.HeapAlloc))
		}
		
		// 更新CPU使用情况
		if cpuStats, err := getCPUStats(); err == nil {
			mm.metrics.CPUUsage.WithLabelValues("user").Set(cpuStats.User)
			mm.metrics.CPUUsage.WithLabelValues("system").Set(cpuStats.System)
		}
	}
}

// AddCollector 添加自定义收集器
func (mm *MonitorManager) AddCollector(collector func()) {
	mm.collectors = append(mm.collectors, collector)
}

// GetMetrics 获取指标实例
func (mm *MonitorManager) GetMetrics() *Metrics {
	return mm.metrics
}

// ========== HTTP监控中间件 ==========

// HTTPMetricsMiddleware HTTP监控中间件
func HTTPMetricsMiddleware(metrics *Metrics) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 记录开始时间
		startTime := time.Now()
		
		// 记录请求大小
		if c.Request.ContentLength > 0 {
			metrics.RequestSize.WithLabelValues(
				c.Request.Method,
				c.FullPath(),
			).Observe(float64(c.Request.ContentLength))
		}
		
		// 处理请求
		c.Next()
		
		// 记录响应时间
		duration := time.Since(startTime).Seconds()
		
		// 记录指标
		status := strconv.Itoa(c.Writer.Status())
		
		metrics.RequestsTotal.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
			status,
		).Inc()
		
		metrics.RequestDuration.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
		).Observe(duration)
		
		// 记录响应大小
		if c.Writer.Size() > 0 {
			metrics.ResponseSize.WithLabelValues(
				c.Request.Method,
				c.FullPath(),
			).Observe(float64(c.Writer.Size()))
		}
	}
}

// ========== 模型推理监控 ==========

// InferenceMetrics 模型推理指标
type InferenceMetrics struct {
	metrics *Metrics
	model   string
}

// NewInferenceMetrics 创建模型推理指标
func NewInferenceMetrics(metrics *Metrics, model string) *InferenceMetrics {
	return &InferenceMetrics{
		metrics: metrics,
		model:   model,
	}
}

// RecordInference 记录推理指标
func (im *InferenceMetrics) RecordInference(inferenceType string, duration time.Duration, tokensGenerated, tokensProcessed int) {
	im.metrics.InferenceTotal.WithLabelValues(im.model, inferenceType).Inc()
	im.metrics.InferenceDuration.WithLabelValues(im.model, inferenceType).Observe(duration.Seconds())
	
	if tokensGenerated > 0 {
		im.metrics.TokensGenerated.WithLabelValues(im.model).Add(float64(tokensGenerated))
	}
	
	if tokensProcessed > 0 {
		im.metrics.TokensProcessed.WithLabelValues(im.model).Add(float64(tokensProcessed))
	}
}

// ========== 缓存监控 ==========

// CacheMetrics 缓存指标
type CacheMetrics struct {
	metrics *Metrics
	cache   string
}

// NewCacheMetrics 创建缓存指标
func NewCacheMetrics(metrics *Metrics, cache string) *CacheMetrics {
	return &CacheMetrics{
		metrics: metrics,
		cache:   cache,
	}
}

// RecordHit 记录缓存命中
func (cm *CacheMetrics) RecordHit() {
	cm.metrics.CacheHits.WithLabelValues(cm.cache).Inc()
}

// RecordMiss 记录缓存未命中
func (cm *CacheMetrics) RecordMiss() {
	cm.metrics.CacheMisses.WithLabelValues(cm.cache).Inc()
}

// UpdateSize 更新缓存大小
func (cm *CacheMetrics) UpdateSize(size int64) {
	cm.metrics.CacheSize.WithLabelValues(cm.cache).Set(float64(size))
}

// ========== 错误监控 ==========

// ErrorMetrics 错误指标
type ErrorMetrics struct {
	metrics *Metrics
}

// NewErrorMetrics 创建错误指标
func NewErrorMetrics(metrics *Metrics) *ErrorMetrics {
	return &ErrorMetrics{
		metrics: metrics,
	}
}

// RecordError 记录错误
func (em *ErrorMetrics) RecordError(errorType, source string) {
	em.metrics.ErrorsTotal.WithLabelValues(errorType, source).Inc()
}

// ========== 业务监控 ==========

// BusinessMetrics 业务指标
type BusinessMetrics struct {
	metrics *Metrics
}

// NewBusinessMetrics 创建业务指标
func NewBusinessMetrics(metrics *Metrics) *BusinessMetrics {
	return &BusinessMetrics{
		metrics: metrics,
	}
}

// UpdateActiveConnections 更新活跃连接数
func (bm *BusinessMetrics) UpdateActiveConnections(count int) {
	bm.metrics.ActiveConnections.Set(float64(count))
}

// UpdateQueueLength 更新队列长度
func (bm *BusinessMetrics) UpdateQueueLength(length int) {
	bm.metrics.QueueLength.Set(float64(length))
}

// ========== 日志记录器 ==========

// StructuredLogger 结构化日志记录器
type StructuredLogger struct {
	*logrus.Logger
	metrics *Metrics
}

// NewStructuredLogger 创建结构化日志记录器
func NewStructuredLogger(metrics *Metrics) *StructuredLogger {
	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: time.RFC3339Nano,
	})
	
	return &StructuredLogger{
		Logger:  logger,
		metrics: metrics,
	}
}

// LogHTTPRequest 记录HTTP请求日志
func (sl *StructuredLogger) LogHTTPRequest(c *gin.Context, latency time.Duration, status int) {
	entry := sl.WithFields(logrus.Fields{
		"method":     c.Request.Method,
		"path":       c.Request.URL.Path,
		"status":     status,
		"latency":    latency,
		"client_ip":  c.ClientIP(),
		"user_agent": c.Request.UserAgent(),
	})
	
	if status >= 400 {
		entry.Error("HTTP request error")
		sl.metrics.ErrorsTotal.WithLabelValues("http", c.Request.URL.Path).Inc()
	} else {
		entry.Info("HTTP request processed")
	}
}

// LogInference 记录推理日志
func (sl *StructuredLogger) LogInference(model, inferenceType string, duration time.Duration, tokensGenerated, tokensProcessed int) {
	sl.WithFields(logrus.Fields{
		"model":           model,
		"type":            inferenceType,
		"duration":        duration,
		"tokens_generated": tokensGenerated,
		"tokens_processed": tokensProcessed,
	}).Info("Model inference completed")
}

// LogError 记录错误日志
func (sl *StructuredLogger) LogError(errorType, source string, err error, fields map[string]interface{}) {
	logFields := logrus.Fields{
		"error_type": errorType,
		"source":     source,
		"error":      err.Error(),
	}
	
	for key, value := range fields {
		logFields[key] = value
	}
	
	sl.WithFields(logFields).Error("Error occurred")
	sl.metrics.ErrorsTotal.WithLabelValues(errorType, source).Inc()
}

// ========== 监控配置 ==========

// MonitoringConfig 监控配置
type MonitoringConfig struct {
	Enabled        bool   `json:"enabled"`
	MetricsEnabled bool   `json:"metrics_enabled"`
	LoggingEnabled bool   `json:"logging_enabled"`
	
	MetricsConfig struct {
		Port        int    `json:"port"`
		Path        string `json:"path"`
		CollectInterval time.Duration `json:"collect_interval"`
	} `json:"metrics_config"`
	
	LoggingConfig struct {
		Level       string `json:"level"`
		Format      string `json:"format"`
		Output      string `json:"output"`
	} `json:"logging_config"`
}

// ========== 工具函数 ==========

// getGoroutineCount 获取goroutine数量
func getGoroutineCount() int {
	return runtime.NumGoroutine()
}

// getMemoryStats 获取内存统计信息
func getMemoryStats() (*runtime.MemStats, error) {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	return &memStats, nil
}

// getCPUStats 获取CPU统计信息
type CPUStats struct {
	User   float64
	System float64
}

func getCPUStats() (*CPUStats, error) {
	// 这里需要实现具体的CPU统计逻辑
	// 可以使用gopsutil包或类似库
	return &CPUStats{
		User:   0.0,
		System: 0.0,
	}, nil
}

// ========== 监控报告 ==========

// MonitoringReport 监控报告
type MonitoringReport struct {
	Timestamp     time.Time `json:"timestamp"`
	Uptime        string    `json:"uptime"`
	SystemMetrics struct {
		MemoryUsage    map[string]uint64 `json:"memory_usage"`
		CPUUsage       map[string]float64 `json:"cpu_usage"`
		Goroutines     int               `json:"goroutines"`
	} `json:"system_metrics"`
	
	HTTPMetrics struct {
		RequestsTotal  int64   `json:"requests_total"`
		RequestRate    float64 `json:"request_rate"`
		AvgDuration    float64 `json:"avg_duration"`
	} `json:"http_metrics"`
	
	ModelMetrics struct {
		InferencesTotal int64   `json:"inferences_total"`
		AvgInferenceTime float64 `json:"avg_inference_time"`
		TokensGenerated int64   `json:"tokens_generated"`
	} `json:"model_metrics"`
	
	ErrorMetrics struct {
		ErrorsTotal    int64   `json:"errors_total"`
		ErrorRate      float64 `json:"error_rate"`
	} `json:"error_metrics"`
}

// GenerateReport 生成监控报告
func (mm *MonitorManager) GenerateReport() (*MonitoringReport, error) {
	report := &MonitoringReport{
		Timestamp: time.Now(),
	}
	
	// 获取系统指标
	if memStats, err := getMemoryStats(); err == nil {
		report.SystemMetrics.MemoryUsage = map[string]uint64{
			"alloc":     memStats.Alloc,
			"sys":       memStats.Sys,
			"heap_alloc": memStats.HeapAlloc,
		}
	}
	
	report.SystemMetrics.Goroutines = getGoroutineCount()
	
	// 这里可以添加更多指标收集逻辑
	
	return report, nil
}

// ========== 监控告警 ==========

// AlertConfig 告警配置
type AlertConfig struct {
	Enabled     bool    `json:"enabled"`
	Thresholds  map[string]float64 `json:"thresholds"`
	Notifiers   []string `json:"notifiers"`
}

// AlertManager 告警管理器
type AlertManager struct {
	config    *AlertConfig
	metrics   *Metrics
	logger    *logrus.Logger
	alerts    map[string]bool
	mutex     sync.RWMutex
}

// NewAlertManager 创建告警管理器
func NewAlertManager(config *AlertConfig, metrics *Metrics, logger *logrus.Logger) *AlertManager {
	return &AlertManager{
		config:  config,
		metrics: metrics,
		logger:  logger,
		alerts:  make(map[string]bool),
	}
}

// CheckAlerts 检查告警条件
func (am *AlertManager) CheckAlerts() {
	if !am.config.Enabled {
		return
	}
	
	// 这里实现具体的告警检查逻辑
	// 例如检查错误率、响应时间等
}

// SendAlert 发送告警
func (am *AlertManager) SendAlert(alertType string, message string, value float64) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	
	// 防止重复告警
	if am.alerts[alertType] {
		return
	}
	
	am.logger.WithFields(logrus.Fields{
		"alert_type": alertType,
		"message":    message,
		"value":      value,
	}).Warn("Alert triggered")
	
	am.alerts[alertType] = true
	
	// 这里可以添加邮件、Slack等通知方式
}

// ClearAlert 清除告警
func (am *AlertManager) ClearAlert(alertType string) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	
	delete(am.alerts, alertType)
	
	am.logger.WithFields(logrus.Fields{
		"alert_type": alertType,
	}).Info("Alert cleared")
}

// ========== 导入runtime包 ==========

import "runtime"