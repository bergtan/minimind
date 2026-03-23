package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
)

var (
	host = flag.String("host", "0.0.0.0", "服务监听地址")
	port = flag.Int("port", 7860, "服务监听端口（与Streamlit默认端口一致）")
)

const (
	AppName    = "GoMiniMind Web Demo"
	AppVersion = "1.0.0"
)

// ========== 训练任务管理 ==========

// TrainTask 训练任务
type TrainTask struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`   // pretrain / sft / dpo
	Status    string    `json:"status"` // running / stopped / finished / error
	StartTime time.Time `json:"start_time"`
	Logs      []string  `json:"logs"`
	mu        sync.Mutex
	cancel    context.CancelFunc
}

var (
	trainTaskMu sync.Mutex
	trainTask   *TrainTask // 同时只允许一个训练任务
)

// TrainRequest 启动训练请求
type TrainRequest struct {
	Type              string  `json:"type"`               // pretrain / sft / dpo
	SaveDir           string  `json:"save_dir"`           // 保存目录
	SaveWeight        string  `json:"save_weight"`        // 权重前缀
	Epochs            int     `json:"epochs"`             // 训练轮数
	BatchSize         int     `json:"batch_size"`         // batch size
	LearningRate      float64 `json:"learning_rate"`      // 学习率
	DataPath          string  `json:"data_path"`          // 数据路径
	FromWeight        string  `json:"from_weight"`        // 基础权重
	HiddenSize        int     `json:"hidden_size"`        // 隐藏层维度
	NumHiddenLayers   int     `json:"num_hidden_layers"`  // 层数
	MaxSeqLen         int     `json:"max_seq_len"`        // 最大序列长度
	AccumulationSteps int     `json:"accumulation_steps"` // 梯度累积
	GradClip          float64 `json:"grad_clip"`          // 梯度裁剪
	WarmupSteps       int     `json:"warmup_steps"`       // warmup步数
	DPOBeta           float64 `json:"dpo_beta"`           // DPO beta参数
	FromResume        int     `json:"from_resume"`        // 是否续训
}

// ChatStreamRequest 前端发来的聊天流式请求
type ChatStreamRequest struct {
	Messages    []ChatMessage `json:"messages"`
	APIURL      string        `json:"api_url"`
	Model       string        `json:"model"`
	APIKey      string        `json:"api_key"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

// ChatMessage 聊天消息
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIChatRequest 发送给OpenAI兼容API的请求
type OpenAIChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

func main() {
	flag.Parse()

	printBanner()

	// 设置Gin模式
	gin.SetMode(gin.ReleaseMode)

	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(gin.Logger())

	// CORS中间件
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	// 获取模板目录路径
	templateDir := getTemplateDir()
	indexHTMLPath := filepath.Join(templateDir, "index.html")

	// ========== 路由注册 ==========

	// 首页 - 直接返回HTML文件内容（不经过Go模板引擎，避免emoji被转义）
	router.GET("/", func(c *gin.Context) {
		c.File(indexHTMLPath)
	})

	// 聊天流式接口 - 代理转发到OpenAI兼容API
	router.POST("/api/chat/stream", handleChatStream)

	// 训练相关接口
	router.POST("/api/train/start", handleTrainStart)
	router.POST("/api/train/stop", handleTrainStop)
	router.GET("/api/train/status", handleTrainStatus)
	router.GET("/api/train/logs", handleTrainLogs)

	// 健康检查
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"service":   AppName,
			"version":   AppVersion,
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		})
	})

	// 启动HTTP服务
	addr := fmt.Sprintf("%s:%d", *host, *port)
	srv := &http.Server{
		Addr:    addr,
		Handler: router,
	}

	go func() {
		log.Printf("🚀 %s v%s 启动成功", AppName, AppVersion)
		log.Printf("📡 Web界面地址: http://127.0.0.1:%d", *port)
		log.Printf("💡 请确保MiniMind API服务已启动（默认 http://127.0.0.1:8000/v1）")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("服务启动失败: %v", err)
		}
	}()

	// 优雅关闭
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("正在关闭服务...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 停止正在运行的训练任务
	trainTaskMu.Lock()
	if trainTask != nil && trainTask.cancel != nil {
		trainTask.cancel()
	}
	trainTaskMu.Unlock()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("服务关闭异常: %v", err)
	}
	log.Println("服务已安全退出")
}

// ========== 训练接口处理 ==========

// handleTrainStart 启动训练任务
func handleTrainStart(c *gin.Context) {
	var req TrainRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误: " + err.Error()})
		return
	}

	trainTaskMu.Lock()
	defer trainTaskMu.Unlock()

	// 检查是否已有任务在运行
	if trainTask != nil && trainTask.Status == "running" {
		c.JSON(http.StatusConflict, gin.H{"error": "已有训练任务正在运行，请先停止"})
		return
	}

	// 构建命令行参数
	args, err := buildTrainArgs(req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 创建任务
	ctx, cancel := context.WithCancel(context.Background())
	task := &TrainTask{
		ID:        fmt.Sprintf("train-%d", time.Now().UnixNano()),
		Type:      req.Type,
		Status:    "running",
		StartTime: time.Now(),
		Logs:      []string{},
		cancel:    cancel,
	}
	trainTask = task

	// 异步执行训练
	go runTrainTask(ctx, task, args)

	c.JSON(http.StatusOK, gin.H{
		"id":     task.ID,
		"type":   task.Type,
		"status": task.Status,
	})
}

// handleTrainStop 停止训练任务
func handleTrainStop(c *gin.Context) {
	trainTaskMu.Lock()
	defer trainTaskMu.Unlock()

	if trainTask == nil || trainTask.Status != "running" {
		c.JSON(http.StatusOK, gin.H{"message": "没有正在运行的训练任务"})
		return
	}

	trainTask.cancel()
	trainTask.mu.Lock()
	trainTask.Status = "stopped"
	trainTask.Logs = append(trainTask.Logs, "[系统] 训练已手动停止")
	trainTask.mu.Unlock()

	c.JSON(http.StatusOK, gin.H{"message": "训练任务已停止"})
}

// handleTrainStatus 查询训练状态
func handleTrainStatus(c *gin.Context) {
	trainTaskMu.Lock()
	defer trainTaskMu.Unlock()

	if trainTask == nil {
		c.JSON(http.StatusOK, gin.H{"status": "idle"})
		return
	}

	trainTask.mu.Lock()
	defer trainTask.mu.Unlock()

	c.JSON(http.StatusOK, gin.H{
		"id":         trainTask.ID,
		"type":       trainTask.Type,
		"status":     trainTask.Status,
		"start_time": trainTask.StartTime.Format(time.RFC3339),
		"log_count":  len(trainTask.Logs),
	})
}

// handleTrainLogs 获取训练日志（SSE流式推送）
func handleTrainLogs(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "不支持流式响应"})
		return
	}

	// 记录已发送的日志行数
	sent := 0
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	clientGone := c.Request.Context().Done()

	for {
		select {
		case <-clientGone:
			return
		case <-ticker.C:
			trainTaskMu.Lock()
			task := trainTask
			trainTaskMu.Unlock()

			if task == nil {
				data, _ := json.Marshal(map[string]interface{}{"status": "idle", "logs": []string{}})
				fmt.Fprintf(c.Writer, "data: %s\n\n", data)
				flusher.Flush()
				continue
			}

			task.mu.Lock()
			logs := task.Logs
			status := task.Status
			task.mu.Unlock()

			// 发送新增日志
			newLogs := []string{}
			if sent < len(logs) {
				newLogs = logs[sent:]
				sent = len(logs)
			}

			data, _ := json.Marshal(map[string]interface{}{
				"status":   status,
				"new_logs": newLogs,
				"type":     task.Type,
			})
			fmt.Fprintf(c.Writer, "data: %s\n\n", data)
			flusher.Flush()

			// 训练结束后再发一次最终状态然后退出
			if status != "running" {
				time.Sleep(1 * time.Second)
				return
			}
		}
	}
}

// buildTrainArgs 根据请求构建训练命令参数
func buildTrainArgs(req TrainRequest) ([]string, error) {
	var cmdPath string
	switch req.Type {
	case "pretrain":
		cmdPath = "cmd/train_pretrain/main.go"
	case "sft":
		cmdPath = "cmd/train_sft/main.go"
	case "dpo":
		cmdPath = "cmd/train_dpo/main.go"
	default:
		return nil, fmt.Errorf("未知训练类型: %s", req.Type)
	}

	args := []string{"run", cmdPath}

	if req.SaveDir != "" {
		args = append(args, "-save_dir", req.SaveDir)
	}
	if req.SaveWeight != "" {
		args = append(args, "-save_weight", req.SaveWeight)
	}
	if req.Epochs > 0 {
		args = append(args, "-epochs", fmt.Sprintf("%d", req.Epochs))
	}
	if req.BatchSize > 0 {
		args = append(args, "-batch_size", fmt.Sprintf("%d", req.BatchSize))
	}
	if req.LearningRate > 0 {
		args = append(args, "-learning_rate", fmt.Sprintf("%g", req.LearningRate))
	}
	if req.DataPath != "" {
		args = append(args, "-data_path", req.DataPath)
	}
	if req.FromWeight != "" {
		args = append(args, "-from_weight", req.FromWeight)
	}
	if req.HiddenSize > 0 {
		args = append(args, "-hidden_size", fmt.Sprintf("%d", req.HiddenSize))
	}
	if req.NumHiddenLayers > 0 {
		args = append(args, "-num_hidden_layers", fmt.Sprintf("%d", req.NumHiddenLayers))
	}
	if req.MaxSeqLen > 0 {
		args = append(args, "-max_seq_len", fmt.Sprintf("%d", req.MaxSeqLen))
	}
	if req.AccumulationSteps > 0 {
		args = append(args, "-accumulation_steps", fmt.Sprintf("%d", req.AccumulationSteps))
	}
	if req.GradClip > 0 {
		args = append(args, "-grad_clip", fmt.Sprintf("%g", req.GradClip))
	}
	if req.WarmupSteps > 0 {
		args = append(args, "-warmup_steps", fmt.Sprintf("%d", req.WarmupSteps))
	}
	if req.FromResume > 0 {
		args = append(args, "-from_resume", fmt.Sprintf("%d", req.FromResume))
	}
	if req.Type == "dpo" && req.DPOBeta > 0 {
		args = append(args, "-beta", fmt.Sprintf("%g", req.DPOBeta))
	}

	return args, nil
}

// runTrainTask 在后台执行训练命令
func runTrainTask(ctx context.Context, task *TrainTask, args []string) {
	addLog := func(line string) {
		task.mu.Lock()
		task.Logs = append(task.Logs, line)
		task.mu.Unlock()
	}

	addLog(fmt.Sprintf("[系统] 启动训练: go %s", strings.Join(args, " ")))

	cmd := exec.CommandContext(ctx, "go", args...)

	// 获取工作目录（项目根目录）
	workDir, err := getProjectRoot()
	if err != nil {
		addLog(fmt.Sprintf("[错误] 无法确定项目根目录: %v", err))
	} else {
		cmd.Dir = workDir
	}

	// 合并stdout和stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		addLog(fmt.Sprintf("[错误] 无法获取输出管道: %v", err))
		task.mu.Lock()
		task.Status = "error"
		task.mu.Unlock()
		return
	}
	cmd.Stderr = cmd.Stdout

	if err := cmd.Start(); err != nil {
		addLog(fmt.Sprintf("[错误] 启动训练进程失败: %v", err))
		task.mu.Lock()
		task.Status = "error"
		task.mu.Unlock()
		return
	}

	addLog(fmt.Sprintf("[系统] 训练进程已启动 (PID: %d)", cmd.Process.Pid))

	// 逐行读取输出
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		addLog(line)
	}
	if scanErr := scanner.Err(); scanErr != nil {
		addLog(fmt.Sprintf("[错误] 读取训练输出失败: %v", scanErr))
	}

	// 等待进程结束
	err = cmd.Wait()
	finalStatus := "finished"
	finalLog := "[系统] 训练完成！"
	if ctx.Err() != nil {
		finalStatus = "stopped"
		finalLog = "[系统] 训练已停止"
	} else if err != nil {
		finalStatus = "error"
		finalLog = fmt.Sprintf("[错误] 训练异常退出: %v", err)
	}

	task.mu.Lock()
	task.Status = finalStatus
	task.Logs = append(task.Logs, finalLog)
	task.mu.Unlock()
}

// getProjectRoot 获取项目根目录
func getProjectRoot() (string, error) {
	// 尝试从可执行文件位置推断
	execPath, err := os.Executable()
	if err == nil {
		// 向上查找go.mod
		dir := filepath.Dir(execPath)
		for i := 0; i < 5; i++ {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				return dir, nil
			}
			dir = filepath.Dir(dir)
		}
	}

	// 从当前工作目录查找
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for i := 0; i < 5; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return os.Getwd()
}

// ========== 聊天接口处理 ==========

// handleChatStream 处理聊天流式请求，代理转发到用户指定的OpenAI兼容API
func handleChatStream(c *gin.Context) {
	var req ChatStreamRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误: " + err.Error()})
		return
	}

	// 验证参数
	if req.APIURL == "" {
		req.APIURL = "http://127.0.0.1:8000/v1"
	}
	if req.Model == "" {
		req.Model = "minimind"
	}
	if req.Temperature <= 0 {
		req.Temperature = 0.85
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 8192
	}

	// 构建OpenAI API请求
	apiURL := strings.TrimSuffix(req.APIURL, "/") + "/chat/completions"
	openAIReq := OpenAIChatRequest{
		Model:       req.Model,
		Messages:    req.Messages,
		Stream:      true,
		Temperature: req.Temperature,
	}

	reqBody, err := json.Marshal(openAIReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "请求序列化失败"})
		return
	}

	// 创建HTTP请求
	httpReq, err := http.NewRequestWithContext(c.Request.Context(), "POST", apiURL, strings.NewReader(string(reqBody)))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建请求失败: " + err.Error()})
		return
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if req.APIKey != "" && req.APIKey != "none" {
		httpReq.Header.Set("Authorization", "Bearer "+req.APIKey)
	}

	// 发送请求
	client := &http.Client{
		Timeout: 5 * time.Minute, // 流式请求需要较长超时
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		// 设置SSE头后返回错误
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")

		errorData, _ := json.Marshal(map[string]string{
			"error": fmt.Sprintf("无法连接到API服务 (%s): %v", apiURL, err),
		})
		c.Writer.Write([]byte("data: " + string(errorData) + "\n\n"))
		c.Writer.Write([]byte("data: [DONE]\n\n"))
		c.Writer.(http.Flusher).Flush()
		return
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")

		errorData, _ := json.Marshal(map[string]string{
			"error": fmt.Sprintf("API返回错误 (HTTP %d): %s", resp.StatusCode, string(body)),
		})
		c.Writer.Write([]byte("data: " + string(errorData) + "\n\n"))
		c.Writer.Write([]byte("data: [DONE]\n\n"))
		c.Writer.(http.Flusher).Flush()
		return
	}

	// 设置SSE流式响应头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("X-Accel-Buffering", "no") // 禁用Nginx缓冲

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "不支持流式响应"})
		return
	}

	// 逐行读取并转发SSE流
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			c.Writer.Write(buf[:n])
			flusher.Flush()
		}
		if err != nil {
			if err != io.EOF {
				log.Printf("读取API响应流出错: %v", err)
			}
			break
		}
	}
}

// getTemplateDir 获取模板目录的绝对路径
func getTemplateDir() string {
	// 首先尝试相对于可执行文件的位置
	execPath, err := os.Executable()
	if err == nil {
		dir := filepath.Join(filepath.Dir(execPath), "templates")
		if _, err := os.Stat(filepath.Join(dir, "index.html")); err == nil {
			return dir
		}
	}

	// 然后尝试相对于当前工作目录
	if dir, err := os.Getwd(); err == nil {
		templateDir := filepath.Join(dir, "templates")
		if _, err := os.Stat(filepath.Join(templateDir, "index.html")); err == nil {
			return templateDir
		}

		// 尝试 cmd/web_demo/templates
		templateDir = filepath.Join(dir, "cmd", "web_demo", "templates")
		if _, err := os.Stat(filepath.Join(templateDir, "index.html")); err == nil {
			return templateDir
		}
	}

	// 最后尝试相对于源文件的位置
	_, filename, _, ok := runtime.Caller(0)
	if ok {
		dir := filepath.Join(filepath.Dir(filename), "templates")
		if _, err := os.Stat(filepath.Join(dir, "index.html")); err == nil {
			return dir
		}
	}

	// 默认路径
	return "templates"
}

func printBanner() {
	banner := `
   ____       __  __ _       _ __  __ _           _ 
  / ___| ___ |  \/  (_)_ __ (_)  \/  (_)_ __   __| |
 | |  _ / _ \| |\/| | | '_ \| | |\/| | | '_ \ / _' |
 | |_| | (_) | |  | | | | | | | |  | | | | | | (_| |
  \____|\___/|_|  |_|_|_| |_|_|_|  |_|_|_| |_|\__,_|
                                          Web Demo v%s
`
	fmt.Printf(banner, AppVersion)
	fmt.Println()
}
