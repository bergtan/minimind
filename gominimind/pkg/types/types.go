package types

import (
	"time"
)

// MiniMindConfig 模型配置结构
type MiniMindConfig struct {
	VocabSize             int     `json:"vocab_size"`
	HiddenSize            int     `json:"hidden_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	UseFlashAttention     bool    `json:"use_flash_attention"`
	RopeTheta             float64 `json:"rope_theta"`
	HiddenAct             string  `json:"hidden_act"`
	IntermediateSize      int     `json:"intermediate_size"`
	LayerNormEps          float64 `json:"layer_norm_eps"`
	InitializerRange      float64 `json:"initializer_range"`
}

// ChatRequest 聊天请求结构
type ChatRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	MaxTokens        int       `json:"max_tokens,omitempty"`
	Temperature      float64   `json:"temperature,omitempty"`
	TopP             float64   `json:"top_p,omitempty"`
	Stream           bool      `json:"stream,omitempty"`
	Stop             []string  `json:"stop,omitempty"`
	PresencePenalty  float64   `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitempty"`
	LogProbs         bool      `json:"log_probs,omitempty"`
	TopLogProbs      int       `json:"top_log_probs,omitempty"`
}

// Message 消息结构
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatResponse 聊天响应结构
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice 选择结构
type Choice struct {
	Index        int       `json:"index"`
	Message      Message   `json:"message"`
	FinishReason string    `json:"finish_reason"`
	LogProbs     *LogProbs `json:"log_probs,omitempty"`
}

// Usage token使用统计
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// LogProbs 对数概率
type LogProbs struct {
	Tokens        []string             `json:"tokens"`
	TokenLogProbs []float64            `json:"token_logprobs"`
	TopLogProbs   []map[string]float64 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

// CompletionRequest 补全请求
type CompletionRequest struct {
	Model            string   `json:"model"`
	Prompt           string   `json:"prompt"`
	MaxTokens        int      `json:"max_tokens,omitempty"`
	Temperature      float64  `json:"temperature,omitempty"`
	TopP             float64  `json:"top_p,omitempty"`
	Stream           bool     `json:"stream,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	PresencePenalty  float64  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitempty"`
	LogProbs         bool     `json:"log_probs,omitempty"`
	Echo             bool     `json:"echo,omitempty"`
}

// CompletionResponse 补全响应
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`
}

// CompletionChoice 补全选择
type CompletionChoice struct {
	Text         string    `json:"text"`
	Index        int       `json:"index"`
	LogProbs     *LogProbs `json:"log_probs,omitempty"`
	FinishReason string    `json:"finish_reason"`
}

// EmbeddingRequest 嵌入请求
type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// EmbeddingResponse 嵌入响应
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

// EmbeddingData 嵌入数据
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// ModelInfo 模型信息
type ModelInfo struct {
	ID                string   `json:"id"`
	Name              string   `json:"name"`
	Description       string   `json:"description"`
	Version           string   `json:"version"`
	ContextLength     int      `json:"context_length"`
	Parameters        string   `json:"parameters"`
	SupportedFeatures []string `json:"supported_features"`
}

// HealthResponse 健康检查响应
type HealthResponse struct {
	Status         string    `json:"status"`
	Timestamp      time.Time `json:"timestamp"`
	Version        string    `json:"version"`
	ModelLoaded    bool      `json:"model_loaded"`
	GPUAvailable   bool      `json:"gpu_available"`
	GPUMemoryUsage string    `json:"gpu_memory_usage"`
	MemoryUsage    string    `json:"memory_usage"`
	Uptime         string    `json:"uptime"`
}

// ErrorResponse 错误响应
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail 错误详情
type ErrorDetail struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Details string `json:"details,omitempty"`
}

// TrainingConfig 训练配置
type TrainingConfig struct {
	Epochs                    int     `json:"epochs"`
	BatchSize                 int     `json:"batch_size"`
	LearningRate              float64 `json:"learning_rate"`
	GradientAccumulationSteps int     `json:"gradient_accumulation_steps"`
	MaxSeqLen                 int     `json:"max_seq_len"`
	UseAmp                    bool    `json:"use_amp"`
	AmpDtype                  string  `json:"amp_dtype"`
	GradientCheckpointing     bool    `json:"gradient_checkpointing"`
	SaveSteps                 int     `json:"save_steps"`
	EvalSteps                 int     `json:"eval_steps"`
	LogSteps                  int     `json:"log_steps"`
}

// DatasetConfig 数据集配置
type DatasetConfig struct {
	DataPath       string `json:"data_path"`
	MaxLength      int    `json:"max_length"`
	Shuffle        bool   `json:"shuffle"`
	NumWorkers     int    `json:"num_workers"`
	PrefetchFactor int    `json:"prefetch_factor"`
	PinMemory      bool   `json:"pin_memory"`
}

// ServerConfig 服务器配置
type ServerConfig struct {
	Host        string   `json:"host"`
	Port        int      `json:"port"`
	APIKey      string   `json:"api_key"`
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature"`
	TopP        float64  `json:"top_p"`
	Workers     int      `json:"workers"`
	LogLevel    string   `json:"log_level"`
	UseCache    bool     `json:"use_cache"`
	CacheSize   int      `json:"cache_size"`
	RateLimit   int      `json:"rate_limit"`
	AllowedIPs  []string `json:"allowed_ips"`
}

// StreamChunk 流式响应块
type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// StreamChoice 流式选择
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta 增量数据
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// BatchRequest 批量请求
type BatchRequest struct {
	Requests []ChatRequest `json:"requests"`
	Parallel int           `json:"parallel"`
}

// BatchResponse 批量响应
type BatchResponse struct {
	Responses []ChatResponse `json:"responses"`
	TotalTime string         `json:"total_time"`
}

// TokenizerConfig 分词器配置
type TokenizerConfig struct {
	VocabSize     int               `json:"vocab_size"`
	SpecialTokens map[string]string `json:"special_tokens"`
	ModelPath     string            `json:"model_path"`
}

// KV缓存结构
type KVCache struct {
	Keys   [][]float32 `json:"keys"`
	Values [][]float32 `json:"values"`
	Length int         `json:"length"`
}

// 训练状态
type TrainingStatus struct {
	Epoch        int     `json:"epoch"`
	Step         int     `json:"step"`
	Loss         float64 `json:"loss"`
	LearningRate float64 `json:"learning_rate"`
	Progress     float64 `json:"progress"`
	ETA          string  `json:"eta"`
}

// 模型性能指标
type PerformanceMetrics struct {
	Throughput     float64 `json:"throughput"` // tokens/秒
	Latency        float64 `json:"latency"`    // 毫秒
	MemoryUsage    string  `json:"memory_usage"`
	GPUUtilization float64 `json:"gpu_utilization"`
}

// ========== 模型相关类型定义 ==========

// ModelConfig 模型配置结构
type ModelConfig struct {
	Name                  string  `json:"name"`
	ModelType             string  `json:"model_type"`
	VocabSize             int     `json:"vocab_size"`
	HiddenSize            int     `json:"hidden_size"`
	NumLayers             int     `json:"num_layers"`
	NumHeads              int     `json:"num_heads"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	IntermediateSize      int     `json:"intermediate_size"`
	HiddenAct             string  `json:"hidden_act"`
	InitializerRange      float64 `json:"initializer_range"`
	LayerNormEps          float64 `json:"layer_norm_eps"`
	UseCache              bool    `json:"use_cache"`
	TorchDtype            string  `json:"torch_dtype"`
	RopeTheta             float64 `json:"rope_theta"`
	AttentionBias         bool    `json:"attention_bias"`
	AttentionDropout      float64 `json:"attention_dropout"`
	HiddenDropout         float64 `json:"hidden_dropout"`
	ModelPath             string  `json:"model_path"`
	CacheSize             int     `json:"cache_size"`
	Device                string  `json:"device"`
	MemoryLimit           int     `json:"memory_limit"`
}

// InferenceStats 推理统计信息
type InferenceStats struct {
	TotalRequests    int64   `json:"total_requests"`
	SuccessfulRequests int64   `json:"successful_requests"`
	FailedRequests   int64   `json:"failed_requests"`
	AvgResponseTime  float64 `json:"avg_response_time"`
	MaxResponseTime  float64 `json:"max_response_time"`
	MinResponseTime  float64 `json:"min_response_time"`
	TokensGenerated  int64   `json:"tokens_generated"`
	TokensProcessed  int64   `json:"tokens_processed"`
}

// MemoryUsage 内存使用信息
type MemoryUsage struct {
	TotalMemory     int64   `json:"total_memory"`
	UsedMemory      int64   `json:"used_memory"`
	FreeMemory      int64   `json:"free_memory"`
	MemoryUsageRate float64 `json:"memory_usage_rate"`
	PeakMemoryUsage int64   `json:"peak_memory_usage"`
}

// CacheStats 缓存统计信息
type CacheStats struct {
	CacheHits       int64   `json:"cache_hits"`
	CacheMisses     int64   `json:"cache_misses"`
	CacheSize       int     `json:"cache_size"`
	CacheUsage     int     `json:"cache_usage"`
	HitRate        float64 `json:"hit_rate"`
	Evictions       int64   `json:"evictions"`
}

// ModelStatus 模型状态
type ModelStatus struct {
	Status        string `json:"status"`
	Message       string `json:"message"`
	LastUsed      string `json:"last_used"`
	LoadTime      string `json:"load_time"`
	IsDefault     bool   `json:"is_default"`
}

// ModelHealth 模型健康状态
type ModelHealth struct {
	Status    string            `json:"status"`
	Message   string            `json:"message"`
	Timestamp string            `json:"timestamp"`
	Checks    map[string]string `json:"checks"`
}

// ManagerHealth 管理器健康状态
type ManagerHealth struct {
	Status    string            `json:"status"`
	Message   string            `json:"message"`
	Timestamp string            `json:"timestamp"`
	Models    map[string]string `json:"models"`
}

// ManagerStats 管理器统计信息
type ManagerStats struct {
	TotalModels     int     `json:"total_models"`
	LoadedModels    int     `json:"loaded_models"`
	MemoryUsage     float64 `json:"memory_usage_mb"`
	TotalRequests   int64   `json:"total_requests"`
	FailedRequests  int64   `json:"failed_requests"`
	AvgResponseTime float64 `json:"avg_response_time_ms"`
}

// OptimizationOptions 优化选项
type OptimizationOptions struct {
	OptimizationType string `json:"optimization_type"`
	TargetPlatform   string `json:"target_platform"`
	Quantization     string `json:"quantization"`
	Compression      string `json:"compression"`
}

// AsyncResult 异步结果结构
type AsyncResult struct {
	Text      string `json:"text"`
	Error     error  `json:"error"`
	Done      bool   `json:"done"`
	RequestID string `json:"request_id"`
}
