package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"gominimind/internal/api"
	"gominimind/pkg/config"
	"gominimind/pkg/model"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	// 导入runtime包
	"runtime"
)

var (
	configPath = flag.String("config", "", "配置文件路径")
	modelPath  = flag.String("model_path", "./MiniMind2", "模型权重路径")
	host       = flag.String("host", "0.0.0.0", "服务监听地址")
	port       = flag.Int("port", 8000, "服务监听端口")
	apiKey     = flag.String("api_key", "", "API密钥")
	maxTokens  = flag.Int("max_tokens", 2048, "最大生成token数")
	temperature = flag.Float64("temperature", 0.7, "采样温度")
	topP       = flag.Float64("top_p", 0.9, "核采样参数")
	workers    = flag.Int("workers", 4, "工作进程数")
	logLevel   = flag.String("log_level", "info", "日志级别")
	useCache   = flag.Bool("use_cache", true, "是否使用缓存")
	cacheSize  = flag.Int("cache_size", 1000, "缓存大小")
	rateLimit  = flag.Int("rate_limit", 100, "请求频率限制")
	sslEnabled = flag.Bool("ssl_enabled", false, "是否启用SSL")
	sslCert    = flag.String("ssl_cert", "", "SSL证书路径")
	sslKey     = flag.String("ssl_key", "", "SSL私钥路径")
	version    = flag.Bool("version", false, "显示版本信息")
)

const (
	AppName    = "GoMiniMind"
	AppVersion = "2.0.0"
	BuildTime  = "2024-01-01"
)

func main() {
	flag.Parse()

	if *version {
		printVersion()
		return
	}

	// 初始化配置
	cfg, err := initConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// 初始化日志
	log := logrus.New()
	log.SetLevel(logrus.InfoLevel)
	if cfg.Logging.Level == "debug" {
		log.SetLevel(logrus.DebugLevel)
	}

	// 显示启动信息
	printStartupInfo(cfg, log)

	// 初始化模型
	log.Info("Loading model...")
	model, err := model.LoadModel(cfg.Model.Path, cfg.Model.ToTypesConfig())
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	log.Info("Model loaded successfully")

	// 初始化API服务器
	log.Info("Initializing API server...")
	server, err := api.NewServer(cfg, model)
	if err != nil {
		log.Fatalf("Failed to create API server: %v", err)
	}

	// 设置路由
	router := setupRouter(server)

	// 启动HTTP服务器
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:    addr,
		Handler: router,
	}

	// 优雅关闭处理
	go func() {
		log.Infof("Server starting on %s", addr)
		
		if cfg.Server.SSLEnabled {
			if err := srv.ListenAndServeTLS(cfg.Server.SSLCertFile, cfg.Server.SSLKeyFile); err != nil && err != http.ErrServerClosed {
				log.Fatalf("Failed to start HTTPS server: %v", err)
			}
		} else {
			if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				log.Fatalf("Failed to start HTTP server: %v", err)
			}
		}
	}()

	// 等待中断信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("Shutting down server...")

	// 优雅关闭
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Info("Server exited")
}

func initConfig() (*config.Config, error) {
	var cfg *config.Config
	var err error

	if *configPath != "" {
		cfg, err = config.LoadConfig(*configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load config from %s: %v", *configPath, err)
		}
	} else {
		cfg = config.DefaultConfig()
	}

	// 合并命令行参数
	mergeFlags(cfg)

	// 合并环境变量
	config.MergeEnvVars(cfg)

	// 验证配置
	if err := config.ValidateConfig(cfg); err != nil {
		return nil, fmt.Errorf("config validation failed: %v", err)
	}

	return cfg, nil
}

func mergeFlags(cfg *config.Config) {
	if *modelPath != "./MiniMind2" {
		cfg.Model.Path = *modelPath
	}
	if *host != "0.0.0.0" {
		cfg.Server.Host = *host
	}
	if *port != 8000 {
		cfg.Server.Port = *port
	}
	if *apiKey != "" {
		cfg.Server.APIKey = *apiKey
	}
	if *maxTokens != 2048 {
		cfg.Server.MaxTokens = *maxTokens
	}
	if *temperature != 0.7 {
		cfg.Server.Temperature = *temperature
	}
	if *topP != 0.9 {
		cfg.Server.TopP = *topP
	}
	if *workers != 4 {
		cfg.Server.Workers = *workers
	}
	if *logLevel != "info" {
		cfg.Server.LogLevel = *logLevel
	}
	if *useCache != true {
		cfg.Server.UseCache = *useCache
	}
	if *cacheSize != 1000 {
		cfg.Server.CacheSize = *cacheSize
	}
	if *rateLimit != 100 {
		cfg.Server.RateLimit = *rateLimit
	}
	if *sslEnabled != false {
		cfg.Server.SSLEnabled = *sslEnabled
	}
	if *sslCert != "" {
		cfg.Server.SSLCertFile = *sslCert
	}
	if *sslKey != "" {
		cfg.Server.SSLKeyFile = *sslKey
	}
}

func setupRouter(server *api.Server) *gin.Engine {
	// 设置Gin模式
	if server.Config.Server.LogLevel == "debug" {
		gin.SetMode(gin.DebugMode)
	} else {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// 全局中间件
	router.Use(gin.Recovery())
	router.Use(server.LoggerMiddleware())
	router.Use(server.CORSMiddleware())
	router.Use(server.RateLimitMiddleware())
	router.Use(server.AuthMiddleware())

	// OpenAI兼容接口
	openai := router.Group("/v1")
	{
		// 聊天接口
		openai.POST("/chat/completions", server.ChatCompletion)
		
		// 补全接口
		openai.POST("/completions", server.Completion)
		
		// 嵌入接口
		openai.POST("/embeddings", server.Embedding)
		
		// 模型列表
		openai.GET("/models", server.ListModels)
		openai.GET("/models/:model_id", server.GetModel)
	}

	// 自定义API接口
	custom := router.Group("/api/v1")
	{
		// 健康检查
		custom.GET("/health", server.HealthCheck)
		custom.GET("/health/detailed", server.DetailedHealthCheck)
		
		// 模型管理
		custom.GET("/models", server.ListModels)
		custom.GET("/models/:id", server.GetModelInfo)
		
		// 批量处理
		custom.POST("/batch/chat", server.BatchChat)
		custom.POST("/batch/embeddings", server.BatchEmbedding)
		
		// 监控指标
		custom.GET("/metrics", server.Metrics)
		custom.GET("/stats", server.Stats)
	}

	// 根路径重定向到文档
	router.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/docs")
	})

	// API文档
	router.GET("/docs", server.Docs)

	return router
}

func printVersion() {
	fmt.Printf("%s v%s\n", AppName, AppVersion)
	fmt.Printf("Build time: %s\n", BuildTime)
	fmt.Printf("Go version: %s\n", runtime.Version())
	fmt.Printf("Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
}

func printStartupInfo(cfg *config.Config, log *logrus.Logger) {
	log.Infof("=== %s v%s ===", AppName, AppVersion)
	log.Infof("Server: %s:%d", cfg.Server.Host, cfg.Server.Port)
	log.Infof("Model: %s", cfg.Model.Path)
	log.Infof("Workers: %d", cfg.Server.Workers)
	log.Infof("Max tokens: %d", cfg.Server.MaxTokens)
	log.Infof("Temperature: %.2f", cfg.Server.Temperature)
	log.Infof("Top P: %.2f", cfg.Server.TopP)
	log.Infof("Cache: %v (size: %d)", cfg.Server.UseCache, cfg.Server.CacheSize)
	log.Infof("Rate limit: %d/min", cfg.Server.RateLimit)
	log.Infof("SSL: %v", cfg.Server.SSLEnabled)
	
	if cfg.Server.APIKey != "" {
		log.Info("API key authentication enabled")
	} else {
		log.Warn("API key authentication disabled - anyone can access the API")
	}
	
	log.Info("=== Startup completed ===")
}

