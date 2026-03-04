package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// ========== 客户端配置 ==========

// ClientConfig 客户端配置
type ClientConfig struct {
	BaseURL    string        `json:"base_url"`
	APIKey     string        `json:"api_key"`
	Timeout    time.Duration `json:"timeout"`
	MaxRetries int           `json:"max_retries"`
	RetryDelay time.Duration `json:"retry_delay"`
	UserAgent  string        `json:"user_agent"`
	
	// HTTP客户端配置
	HTTPClient *http.Client `json:"-"`
	
	// 日志配置
	Logger     *logrus.Logger `json:"-"`
	LogLevel   string         `json:"log_level"`
}

// DefaultConfig 默认配置
func DefaultConfig() *ClientConfig {
	return &ClientConfig{
		BaseURL:    "http://localhost:8000",
		APIKey:     "",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
		RetryDelay: 1 * time.Second,
		UserAgent:  "minimind-go-client/1.0.0",
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		LogLevel: "info",
	}
}

// ========== 基础客户端 ==========

// MiniMindClient MiniMind客户端
type MiniMindClient struct {
	config     *ClientConfig
	httpClient *http.Client
	logger     *logrus.Logger
	
	// 服务客户端
	Chat       *ChatClient
	Completions *CompletionsClient
	Embeddings *EmbeddingsClient
	Models     *ModelsClient
	Files      *FilesClient
}

// NewMiniMindClient 创建新的MiniMind客户端
func NewMiniMindClient(config *ClientConfig) *MiniMindClient {
	if config == nil {
		config = DefaultConfig()
	}
	
	// 初始化HTTP客户端
	if config.HTTPClient == nil {
		config.HTTPClient = &http.Client{
			Timeout: config.Timeout,
		}
	}
	
	// 初始化日志记录器
	logger := config.Logger
	if logger == nil {
		logger = logrus.New()
		level, err := logrus.ParseLevel(config.LogLevel)
		if err != nil {
			level = logrus.InfoLevel
		}
		logger.SetLevel(level)
	}
	
	client := &MiniMindClient{
		config:     config,
		httpClient: config.HTTPClient,
		logger:     logger,
	}
	
	// 初始化服务客户端
	client.Chat = &ChatClient{client: client}
	client.Completions = &CompletionsClient{client: client}
	client.Embeddings = &EmbeddingsClient{client: client}
	client.Models = &ModelsClient{client: client}
	client.Files = &FilesClient{client: client}
	
	return client
}

// ========== 通用请求处理 ==========

// RequestOptions 请求选项
type RequestOptions struct {
	Headers map[string]string
	Query   map[string]string
	Timeout time.Duration
	Retries int
}

// doRequest 执行HTTP请求
func (c *MiniMindClient) doRequest(ctx context.Context, method, path string, body interface{}, options *RequestOptions) (*http.Response, error) {
	var retries int
	if options != nil && options.Retries > 0 {
		retries = options.Retries
	} else {
		retries = c.config.MaxRetries
	}
	
	var lastErr error
	
	for i := 0; i <= retries; i++ {
		resp, err := c.doSingleRequest(ctx, method, path, body, options)
		if err == nil {
			return resp, nil
		}
		
		lastErr = err
		
		// 检查是否应该重试
		if !c.shouldRetry(err, resp, i) {
			break
		}
		
		// 等待重试延迟
		if i < retries {
			delay := c.config.RetryDelay * time.Duration(i+1)
			c.logger.Debugf("Request failed, retrying in %v: %v", delay, err)
			time.Sleep(delay)
		}
	}
	
	return nil, lastErr
}

// doSingleRequest 执行单次请求
func (c *MiniMindClient) doSingleRequest(ctx context.Context, method, path string, body interface{}, options *RequestOptions) (*http.Response, error) {
	// 构建URL
	u, err := url.Parse(c.config.BaseURL + path)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}
	
	// 添加查询参数
	if options != nil && options.Query != nil {
		query := u.Query()
		for key, value := range options.Query {
			query.Set(key, value)
		}
		u.RawQuery = query.Encode()
	}
	
	// 构建请求体
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonData)
	}
	
	// 创建请求
	req, err := http.NewRequestWithContext(ctx, method, u.String(), reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	// 设置请求头
	req.Header.Set("User-Agent", c.config.UserAgent)
	req.Header.Set("Content-Type", "application/json")
	
	if c.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}
	
	// 添加自定义请求头
	if options != nil && options.Headers != nil {
		for key, value := range options.Headers {
			req.Header.Set(key, value)
		}
	}
	
	// 执行请求
	c.logger.Debugf("Making %s request to %s", method, u.String())
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	
	return resp, nil
}

// shouldRetry 判断是否应该重试
func (c *MiniMindClient) shouldRetry(err error, resp *http.Response, attempt int) bool {
	if err != nil {
		return true
	}
	
	if resp == nil {
		return false
	}
	
	// 5xx状态码可以重试
	if resp.StatusCode >= 500 && resp.StatusCode < 600 {
		return true
	}
	
	// 429状态码（限流）可以重试
	if resp.StatusCode == 429 {
		return true
	}
	
	return false
}

// parseResponse 解析响应
func (c *MiniMindClient) parseResponse(resp *http.Response, result interface{}) error {
	defer resp.Body.Close()
	
	// 检查状态码
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return c.handleErrorResponse(resp)
	}
	
	// 解析响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}
	
	if result != nil {
		if err := json.Unmarshal(body, result); err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}
	}
	
	return nil
}

// handleErrorResponse 处理错误响应
func (c *MiniMindClient) handleErrorResponse(resp *http.Response) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read error response: %w", err)
	}
	
	var errorResp ErrorResponse
	if err := json.Unmarshal(body, &errorResp); err == nil && errorResp.Error != nil {
		return &APIError{
			StatusCode: resp.StatusCode,
			Error:      errorResp.Error,
		}
	}
	
	return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
}

// ========== 错误处理 ==========

// ErrorResponse 错误响应
type ErrorResponse struct {
	Error *APIErrorDetail `json:"error"`
}

// APIErrorDetail API错误详情
type APIErrorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
}

// APIError API错误
type APIError struct {
	StatusCode int
	Error      *APIErrorDetail
}

func (e *APIError) Error() string {
	if e.Error != nil {
		return fmt.Sprintf("API error %d: %s (%s)", e.StatusCode, e.Error.Message, e.Error.Code)
	}
	return fmt.Sprintf("API error %d", e.StatusCode)
}

// ========== 聊天客户端 ==========

// ChatClient 聊天客户端
type ChatClient struct {
	client *MiniMindClient
}

// ChatCompletionRequest 聊天补全请求
type ChatCompletionRequest struct {
	Model            string                  `json:"model"`
	Messages         []ChatMessage           `json:"messages"`
	MaxTokens        *int                    `json:"max_tokens,omitempty"`
	Temperature      *float64                `json:"temperature,omitempty"`
	TopP             *float64                `json:"top_p,omitempty"`
	Stream           bool                    `json:"stream,omitempty"`
	Stop             []string                `json:"stop,omitempty"`
	PresencePenalty  *float64                `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64                `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]float64      `json:"logit_bias,omitempty"`
	User             string                  `json:"user,omitempty"`
}

// ChatMessage 聊天消息
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// ChatCompletionResponse 聊天补全响应
type ChatCompletionResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage               `json:"usage"`
}

// ChatCompletionChoice 聊天补全选择
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatCompletionChunk 聊天补全流式响应块
type ChatCompletionChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []ChatCompletionChunkChoice `json:"choices"`
}

// ChatCompletionChunkChoice 聊天补全流式响应选择
type ChatCompletionChunkChoice struct {
	Index        int         `json:"index"`
	Delta        ChatMessage `json:"delta"`
	FinishReason string      `json:"finish_reason"`
}

// CreateCompletion 创建聊天补全
func (cc *ChatClient) CreateCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	var resp ChatCompletionResponse
	
	httpResp, err := cc.client.doRequest(ctx, "POST", "/v1/chat/completions", req, nil)
	if err != nil {
		return nil, err
	}
	
	if err := cc.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// CreateCompletionStream 创建流式聊天补全
func (cc *ChatClient) CreateCompletionStream(ctx context.Context, req ChatCompletionRequest) (*StreamReader, error) {
	req.Stream = true
	
	httpResp, err := cc.client.doRequest(ctx, "POST", "/v1/chat/completions", req, nil)
	if err != nil {
		return nil, err
	}
	
	return NewStreamReader(httpResp.Body), nil
}

// ========== 补全客户端 ==========

// CompletionsClient 补全客户端
type CompletionsClient struct {
	client *MiniMindClient
}

// CompletionRequest 补全请求
type CompletionRequest struct {
	Model            string            `json:"model"`
	Prompt           string            `json:"prompt"`
	MaxTokens        *int              `json:"max_tokens,omitempty"`
	Temperature      *float64          `json:"temperature,omitempty"`
	TopP             *float64          `json:"top_p,omitempty"`
	Stream           bool              `json:"stream,omitempty"`
	Stop             []string          `json:"stop,omitempty"`
	PresencePenalty  *float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64          `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]float64 `json:"logit_bias,omitempty"`
	User             string            `json:"user,omitempty"`
}

// CompletionResponse 补全响应
type CompletionResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage             `json:"usage"`
}

// CompletionChoice 补全选择
type CompletionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	Logprobs     interface{} `json:"logprobs"`
	FinishReason string `json:"finish_reason"`
}

// CreateCompletion 创建补全
func (cc *CompletionsClient) CreateCompletion(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	var resp CompletionResponse
	
	httpResp, err := cc.client.doRequest(ctx, "POST", "/v1/completions", req, nil)
	if err != nil {
		return nil, err
	}
	
	if err := cc.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// CreateCompletionStream 创建流式补全
func (cc *CompletionsClient) CreateCompletionStream(ctx context.Context, req CompletionRequest) (*StreamReader, error) {
	req.Stream = true
	
	httpResp, err := cc.client.doRequest(ctx, "POST", "/v1/completions", req, nil)
	if err != nil {
		return nil, err
	}
	
	return NewStreamReader(httpResp.Body), nil
}

// ========== 嵌入客户端 ==========

// EmbeddingsClient 嵌入客户端
type EmbeddingsClient struct {
	client *MiniMindClient
}

// EmbeddingRequest 嵌入请求
type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
	User  string   `json:"user,omitempty"`
}

// EmbeddingResponse 嵌入响应
type EmbeddingResponse struct {
	Object string        `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string        `json:"model"`
	Usage  Usage         `json:"usage"`
}

// EmbeddingData 嵌入数据
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// CreateEmbedding 创建嵌入
func (ec *EmbeddingsClient) CreateEmbedding(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	var resp EmbeddingResponse
	
	httpResp, err := ec.client.doRequest(ctx, "POST", "/v1/embeddings", req, nil)
	if err != nil {
		return nil, err
	}
	
	if err := ec.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// ========== 模型客户端 ==========

// ModelsClient 模型客户端
type ModelsClient struct {
	client *MiniMindClient
}

// ModelInfo 模型信息
type ModelInfo struct {
	ID               string   `json:"id"`
	Object           string   `json:"object"`
	Created          int64    `json:"created"`
	OwnedBy          string   `json:"owned_by"`
	Permissions      []string `json:"permissions"`
	Root             string   `json:"root"`
	Parent           string   `json:"parent"`
}

// ModelsListResponse 模型列表响应
type ModelsListResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ListModels 列出模型
func (mc *ModelsClient) ListModels(ctx context.Context) (*ModelsListResponse, error) {
	var resp ModelsListResponse
	
	httpResp, err := mc.client.doRequest(ctx, "GET", "/v1/models", nil, nil)
	if err != nil {
		return nil, err
	}
	
	if err := mc.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// RetrieveModel 获取模型详情
func (mc *ModelsClient) RetrieveModel(ctx context.Context, modelID string) (*ModelInfo, error) {
	var resp ModelInfo
	
	httpResp, err := mc.client.doRequest(ctx, "GET", "/v1/models/"+modelID, nil, nil)
	if err != nil {
		return nil, err
	}
	
	if err := mc.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// ========== 文件客户端 ==========

// FilesClient 文件客户端
type FilesClient struct {
	client *MiniMindClient
}

// FileInfo 文件信息
type FileInfo struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int64  `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

// FilesListResponse 文件列表响应
type FilesListResponse struct {
	Object string     `json:"object"`
	Data   []FileInfo `json:"data"`
}

// ListFiles 列出文件
func (fc *FilesClient) ListFiles(ctx context.Context) (*FilesListResponse, error) {
	var resp FilesListResponse
	
	httpResp, err := fc.client.doRequest(ctx, "GET", "/v1/files", nil, nil)
	if err != nil {
		return nil, err
	}
	
	if err := fc.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// ========== 流式读取器 ==========

// StreamReader 流式读取器
type StreamReader struct {
	reader   io.ReadCloser
	decoder  *json.Decoder
	scanner  *bufio.Scanner
}

// NewStreamReader 创建流式读取器
func NewStreamReader(body io.ReadCloser) *StreamReader {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024), 1024*1024) // 1MB缓冲区
	
	return &StreamReader{
		reader:  body,
		scanner: scanner,
	}
}

// ReadChunk 读取流式块
func (sr *StreamReader) ReadChunk() (*ChatCompletionChunk, error) {
	if !sr.scanner.Scan() {
		if err := sr.scanner.Err(); err != nil {
			return nil, err
		}
		return nil, io.EOF
	}
	
	line := sr.scanner.Text()
	
	// 跳过空行和[DONE]标记
	if line == "" || line == "data: [DONE]" {
		return sr.ReadChunk()
	}
	
	// 解析数据行
	if strings.HasPrefix(line, "data: ") {
		line = strings.TrimPrefix(line, "data: ")
		
		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			return nil, fmt.Errorf("failed to unmarshal chunk: %w", err)
		}
		
		return &chunk, nil
	}
	
	return sr.ReadChunk()
}

// Close 关闭读取器
func (sr *StreamReader) Close() error {
	return sr.reader.Close()
}

// ========== 使用统计 ==========

// Usage 使用统计
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ========== 批量处理 ==========

// BatchRequest 批量请求
type BatchRequest struct {
	Requests []BatchRequestItem `json:"requests"`
	Parallel int               `json:"parallel,omitempty"`
}

// BatchRequestItem 批量请求项
type BatchRequestItem struct {
	Messages  []ChatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens,omitempty"`
}

// BatchResponse 批量响应
type BatchResponse struct {
	Responses []ChatCompletionResponse `json:"responses"`
}

// BatchChat 批量聊天
func (cc *ChatClient) BatchChat(ctx context.Context, req BatchRequest) (*BatchResponse, error) {
	var resp BatchResponse
	
	httpResp, err := cc.client.doRequest(ctx, "POST", "/api/v1/batch/chat", req, nil)
	if err != nil {
		return nil, err
	}
	
	if err := cc.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// BatchEmbeddingRequest 批量嵌入请求
type BatchEmbeddingRequest struct {
	Texts     []string `json:"texts"`
	BatchSize int      `json:"batch_size,omitempty"`
}

// BatchEmbeddingResponse 批量嵌入响应
type BatchEmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// BatchEmbedding 批量嵌入
func (ec *EmbeddingsClient) BatchEmbedding(ctx context.Context, req BatchEmbeddingRequest) (*BatchEmbeddingResponse, error) {
	var resp BatchEmbeddingResponse
	
	httpResp, err := ec.client.doRequest(ctx, "POST", "/api/v1/batch/embeddings", req, nil)
	if err != nil {
		return nil, err
	}
	
	if err := ec.client.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// ========== 健康检查 ==========

// HealthResponse 健康检查响应
type HealthResponse struct {
	Status      string `json:"status"`
	Timestamp   string `json:"timestamp"`
	Version     string `json:"version"`
	ModelLoaded bool   `json:"model_loaded"`
	GPUAvailable bool  `json:"gpu_available"`
	GPUMemoryUsage string `json:"gpu_memory_usage"`
}

// Health 健康检查
func (c *MiniMindClient) Health(ctx context.Context) (*HealthResponse, error) {
	var resp HealthResponse
	
	httpResp, err := c.doRequest(ctx, "GET", "/health", nil, nil)
	if err != nil {
		return nil, err
	}
	
	if err := c.parseResponse(httpResp, &resp); err != nil {
		return nil, err
	}
	
	return &resp, nil
}

// ========== 导入bufio包 ==========

import "bufio"