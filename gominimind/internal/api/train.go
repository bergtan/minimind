package api

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
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

// TrainRequest 启动训练请求
type TrainRequest struct {
	Type              string  `json:"type"`
	SaveDir           string  `json:"save_dir"`
	SaveWeight        string  `json:"save_weight"`
	Epochs            int     `json:"epochs"`
	BatchSize         int     `json:"batch_size"`
	LearningRate      float64 `json:"learning_rate"`
	DataPath          string  `json:"data_path"`
	FromWeight        string  `json:"from_weight"`
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	MaxSeqLen         int     `json:"max_seq_len"`
	AccumulationSteps int     `json:"accumulation_steps"`
	GradClip          float64 `json:"grad_clip"`
	WarmupSteps       int     `json:"warmup_steps"`
	DPOBeta           float64 `json:"dpo_beta"`
	FromResume        int     `json:"from_resume"`
}

var (
	trainTaskMu sync.Mutex
	trainTask   *TrainTask // 同时只允许一个训练任务
)

// HandleTrainStart 启动训练任务
func (s *Server) HandleTrainStart(c *gin.Context) {
	var req TrainRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误: " + err.Error()})
		return
	}

	trainTaskMu.Lock()
	defer trainTaskMu.Unlock()

	if trainTask != nil && trainTask.Status == "running" {
		c.JSON(http.StatusConflict, gin.H{"error": "已有训练任务正在运行，请先停止"})
		return
	}

	args, err := buildTrainArgs(req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

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

	go runTrainTask(ctx, task, args)

	c.JSON(http.StatusOK, gin.H{
		"id":     task.ID,
		"type":   task.Type,
		"status": task.Status,
	})
}

// HandleTrainStop 停止训练任务
func (s *Server) HandleTrainStop(c *gin.Context) {
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

// HandleTrainStatus 查询训练状态
func (s *Server) HandleTrainStatus(c *gin.Context) {
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

// HandleTrainLogs 获取训练日志（SSE流式推送）
func (s *Server) HandleTrainLogs(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	c.Header("Access-Control-Allow-Origin", "*")

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "不支持流式响应"})
		return
	}

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
				data, _ := json.Marshal(map[string]interface{}{"status": "idle", "new_logs": []string{}})
				fmt.Fprintf(c.Writer, "data: %s\n\n", data)
				flusher.Flush()
				continue
			}

			task.mu.Lock()
			logs := task.Logs
			status := task.Status
			task.mu.Unlock()

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

	// 使用 os/exec 包
	cmd := exec.CommandContext(ctx, "go", args...)

	workDir, err := getTrainProjectRoot()
	if err != nil {
		addLog(fmt.Sprintf("[错误] 无法确定项目根目录: %v", err))
	} else {
		cmd.Dir = workDir
	}

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

	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		addLog(scanner.Text())
	}
	if scanErr := scanner.Err(); scanErr != nil {
		addLog(fmt.Sprintf("[错误] 读取训练输出失败: %v", scanErr))
	}

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

// getTrainProjectRoot 获取项目根目录
func getTrainProjectRoot() (string, error) {
	if execPath, err := os.Executable(); err == nil {
		dir := filepath.Dir(execPath)
		for i := 0; i < 5; i++ {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				return dir, nil
			}
			dir = filepath.Dir(dir)
		}
	}

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
