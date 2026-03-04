package model

import (
	"errors"
	"strings"

	"gominimind/pkg/types"
)

// Model 模型接口定义
// 定义了语言模型的核心功能，包括文本生成、嵌入生成、模型管理等
type Model interface {
	// ========== 基础模型方法 ==========

	// Load 加载模型
	// path: 模型权重文件路径
	// config: 模型配置
	// 返回: 错误信息
	Load(path string, config *types.ModelConfig) error

	// Unload 卸载模型
	// 释放模型占用的资源
	Unload() error

	// IsLoaded 检查模型是否已加载
	// 返回: 模型加载状态
	IsLoaded() bool

	// GetConfig 获取模型配置
	// 返回: 模型配置信息
	GetConfig() *types.ModelConfig

	// ========== 文本生成方法 ==========

	// Generate 生成文本
	// prompt: 输入提示文本
	// maxTokens: 最大生成token数
	// temperature: 采样温度
	// topP: 核采样参数
	// 返回: 生成的文本和错误信息
	Generate(prompt string, maxTokens int, temperature, topP float64) (string, error)

	// GenerateWithContext 带上下文的文本生成
	// messages: 对话消息列表
	// maxTokens: 最大生成token数
	// temperature: 采样温度
	// topP: 核采样参数
	// 返回: 生成的文本和错误信息
	GenerateWithContext(messages []types.Message, maxTokens int, temperature, topP float64) (string, error)

	// GenerateStream 流式文本生成
	// prompt: 输入提示文本
	// maxTokens: 最大生成token数
	// temperature: 采样温度
	// topP: 核采样参数
	// callback: 流式回调函数
	// 返回: 错误信息
	GenerateStream(prompt string, maxTokens int, temperature, topP float64, callback func(string) error) error

	// ========== 嵌入生成方法 ==========

	// GenerateEmbedding 生成文本嵌入
	// text: 输入文本
	// 返回: 嵌入向量和错误信息
	GenerateEmbedding(text string) ([]float32, error)

	// GenerateEmbeddings 批量生成文本嵌入
	// texts: 输入文本列表
	// 返回: 嵌入向量列表和错误信息
	GenerateEmbeddings(texts []string) ([][]float32, error)

	// GetEmbeddingDimension 获取嵌入维度
	// 返回: 嵌入向量维度
	GetEmbeddingDimension() int

	// ========== 模型信息方法 ==========

	// GetModelInfo 获取模型信息
	// 返回: 模型详细信息
	GetModelInfo() *types.ModelInfo

	// GetParameters 获取模型参数数量
	// 返回: 参数数量（单位：百万）
	GetParameters() float64

	// GetContextLength 获取上下文长度
	// 返回: 最大上下文长度
	GetContextLength() int

	// GetVocabSize 获取词汇表大小
	// 返回: 词汇表大小
	GetVocabSize() int

	// ========== 性能监控方法 ==========

	// GetInferenceStats 获取推理统计信息
	// 返回: 推理统计信息
	GetInferenceStats() *types.InferenceStats

	// ResetStats 重置统计信息
	ResetStats()

	// GetMemoryUsage 获取内存使用情况
	// 返回: 内存使用信息
	GetMemoryUsage() *types.MemoryUsage

	// ========== 模型管理方法 ==========

	// Save 保存模型
	// path: 保存路径
	// 返回: 错误信息
	Save(path string) error

	// Export 导出模型
	// format: 导出格式（onnx, gguf, etc.）
	// path: 导出路径
	// 返回: 错误信息
	Export(format string, path string) error

	// Quantize 量化模型
	// method: 量化方法（int8, int4, etc.）
	// 返回: 错误信息
	Quantize(method string) error

	// ========== 上下文管理方法 ==========

	// CreateContext 创建推理上下文
	// 返回: 上下文ID和错误信息
	CreateContext() (string, error)

	// SetContext 设置当前上下文
	// contextID: 上下文ID
	// 返回: 错误信息
	SetContext(contextID string) error

	// DeleteContext 删除上下文
	// contextID: 上下文ID
	// 返回: 错误信息
	DeleteContext(contextID string) error

	// GetContexts 获取所有上下文
	// 返回: 上下文ID列表
	GetContexts() []string

	// ========== 缓存管理方法 ==========

	// EnableCache 启用KV缓存
	// 返回: 错误信息
	EnableCache() error

	// DisableCache 禁用KV缓存
	// 返回: 错误信息
	DisableCache() error

	// ClearCache 清空缓存
	// 返回: 错误信息
	ClearCache() error

	// GetCacheStats 获取缓存统计信息
	// 返回: 缓存统计信息
	GetCacheStats() *types.CacheStats

	// ========== 工具方法 ==========

	// Tokenize 分词
	// text: 输入文本
	// 返回: token列表和错误信息
	Tokenize(text string) ([]int, error)

	// Detokenize 反分词
	// tokens: token列表
	// 返回: 文本和错误信息
	Detokenize(tokens []int) (string, error)

	// GetTokenCount 获取token数量
	// text: 输入文本
	// 返回: token数量
	GetTokenCount(text string) (int, error)

	// ========== 异步方法 ==========

	// GenerateAsync 异步生成文本
	// prompt: 输入提示文本
	// maxTokens: 最大生成token数
	// temperature: 采样温度
	// topP: 核采样参数
	// 返回: 通道，用于接收结果和错误
	GenerateAsync(prompt string, maxTokens int, temperature, topP float64) (<-chan types.AsyncResult, error)

	// CancelGeneration 取消生成
	// requestID: 请求ID
	// 返回: 错误信息
	CancelGeneration(requestID string) error

	// ========== 批量处理方法 ==========

	// BatchGenerate 批量生成文本
	// prompts: 输入提示文本列表
	// maxTokens: 最大生成token数
	// temperature: 采样温度
	// topP: 核采样参数
	// 返回: 生成结果列表和错误信息
	BatchGenerate(prompts []string, maxTokens int, temperature, topP float64) ([]string, error)

	// BatchGenerateWithContext 批量带上下文生成
	// messageBatches: 消息批次列表
	// maxTokens: 最大生成token数
	// temperature: 采样温度
	// topP: 核采样参数
	// 返回: 生成结果列表和错误信息
	BatchGenerateWithContext(messageBatches [][]types.Message, maxTokens int, temperature, topP float64) ([]string, error)

	// ========== 模型状态方法 ==========

	// HealthCheck 健康检查
	// 返回: 健康状态和错误信息
	HealthCheck() (*types.ModelHealth, error)

	// GetStatus 获取模型状态
	// 返回: 模型状态信息
	GetStatus() *types.ModelStatus

	// SetStatus 设置模型状态
	// status: 模型状态
	// 返回: 错误信息
	SetStatus(status types.ModelStatus) error

	// ========== 配置管理方法 ==========

	// UpdateConfig 更新模型配置
	// config: 新的配置
	// 返回: 错误信息
	UpdateConfig(config *types.ModelConfig) error

	// GetDefaultConfig 获取默认配置
	// 返回: 默认配置
	GetDefaultConfig() *types.ModelConfig

	// ValidateConfig 验证配置
	// config: 配置信息
	// 返回: 验证错误列表
	ValidateConfig(config *types.ModelConfig) []error

	// ========== 资源管理方法 ==========

	// SetDevice 设置设备
	// device: 设备名称（cpu, cuda:0, etc.）
	// 返回: 错误信息
	SetDevice(device string) error

	// GetDevice 获取当前设备
	// 返回: 设备名称
	GetDevice() string

	// SetMemoryLimit 设置内存限制
	// limit: 内存限制（MB）
	// 返回: 错误信息
	SetMemoryLimit(limit int) error

	// GetMemoryLimit 获取内存限制
	// 返回: 内存限制（MB）
	GetMemoryLimit() int

	// ========== 扩展方法 ==========

	// RegisterExtension 注册扩展
	// name: 扩展名称
	// extension: 扩展接口
	// 返回: 错误信息
	RegisterExtension(name string, extension interface{}) error

	// GetExtension 获取扩展
	// name: 扩展名称
	// 返回: 扩展接口
	GetExtension(name string) interface{}

	// ListExtensions 列出所有扩展
	// 返回: 扩展名称列表
	ListExtensions() []string

	// ========== 事件回调方法 ==========

	// SetProgressCallback 设置进度回调
	// callback: 进度回调函数
	SetProgressCallback(callback func(progress float32, message string))

	// SetErrorCallback 设置错误回调
	// callback: 错误回调函数
	SetErrorCallback(callback func(error))

	// SetLogCallback 设置日志回调
	// callback: 日志回调函数
	SetLogCallback(callback func(level, message string))
}



// ModelManager 模型管理器接口
// 用于管理多个模型实例
type ModelManager interface {
	// LoadModel 加载模型
	// modelID: 模型ID
	// modelType: 模型类型
	// modelPath: 模型路径
	// config: 模型配置
	// 返回: 错误信息
	LoadModel(modelID, modelType, modelPath string, config *types.ModelConfig) error

	// UnloadModel 卸载模型
	// modelID: 模型ID
	// 返回: 错误信息
	UnloadModel(modelID string) error

	// GetModel 获取模型实例
	// modelID: 模型ID
	// 返回: 模型实例和错误信息
	GetModel(modelID string) (Model, error)

	// ListModels 列出所有已加载模型
	// 返回: 模型ID列表
	ListModels() []string

	// SetDefaultModel 设置默认模型
	// modelID: 模型ID
	// 返回: 错误信息
	SetDefaultModel(modelID string) error

	// GetDefaultModel 获取默认模型
	// 返回: 模型实例和错误信息
	GetDefaultModel() (Model, error)

	// ModelExists 检查模型是否存在
	// modelID: 模型ID
	// 返回: 是否存在
	ModelExists(modelID string) bool

	// GetModelStatus 获取模型状态
	// modelID: 模型ID
	// 返回: 模型状态和错误信息
	GetModelStatus(modelID string) (*types.ModelStatus, error)

	// GetModelInfo 获取模型信息
	// modelID: 模型ID
	// 返回: 模型信息和错误信息
	GetModelInfo(modelID string) (*types.ModelInfo, error)

	// UpdateModelConfig 更新模型配置
	// modelID: 模型ID
	// config: 新的配置
	// 返回: 错误信息
	UpdateModelConfig(modelID string, config *types.ModelConfig) error

	// HealthCheck 健康检查
	// 返回: 健康状态和错误信息
	HealthCheck() (*types.ManagerHealth, error)

	// GetStats 获取管理器统计信息
	// 返回: 统计信息
	GetStats() *types.ManagerStats

	// Cleanup 清理资源
	// 返回: 错误信息
	Cleanup() error
}

// Tokenizer 分词器接口
// 用于文本分词和反分词
type Tokenizer interface {
	// Tokenize 分词
	// text: 输入文本
	// 返回: token列表和错误信息
	Tokenize(text string) ([]int, error)

	// Detokenize 反分词
	// tokens: token列表
	// 返回: 文本和错误信息
	Detokenize(tokens []int) (string, error)

	// GetTokenCount 获取token数量
	// text: 输入文本
	// 返回: token数量
	GetTokenCount(text string) (int, error)

	// GetVocabSize 获取词汇表大小
	// 返回: 词汇表大小
	GetVocabSize() int

	// GetSpecialTokens 获取特殊token
	// 返回: 特殊token映射
	GetSpecialTokens() map[string]int

	// IsValidToken 检查token是否有效
	// token: token ID
	// 返回: 是否有效
	IsValidToken(token int) bool

	// Save 保存分词器
	// path: 保存路径
	// 返回: 错误信息
	Save(path string) error

	// Load 加载分词器
	// path: 加载路径
	// 返回: 错误信息
	Load(path string) error
}

// EmbeddingModel 嵌入模型接口
// 专门用于生成文本嵌入
type EmbeddingModel interface {
	// GenerateEmbedding 生成文本嵌入
	// text: 输入文本
	// 返回: 嵌入向量和错误信息
	GenerateEmbedding(text string) ([]float32, error)

	// GenerateEmbeddings 批量生成文本嵌入
	// texts: 输入文本列表
	// 返回: 嵌入向量列表和错误信息
	GenerateEmbeddings(texts []string) ([][]float32, error)

	// GetEmbeddingDimension 获取嵌入维度
	// 返回: 嵌入向量维度
	GetEmbeddingDimension() int

	// GetMaxSequenceLength 获取最大序列长度
	// 返回: 最大序列长度
	GetMaxSequenceLength() int

	// NormalizeEmbeddings 归一化嵌入向量
	// embeddings: 嵌入向量列表
	// 返回: 归一化后的嵌入向量
	NormalizeEmbeddings(embeddings [][]float32) ([][]float32, error)

	// Similarity 计算相似度
	// embedding1: 嵌入向量1
	// embedding2: 嵌入向量2
	// 返回: 相似度分数
	Similarity(embedding1, embedding2 []float32) (float32, error)

	// BatchSimilarity 批量计算相似度
	// embeddings1: 嵌入向量列表1
	// embeddings2: 嵌入向量列表2
	// 返回: 相似度分数列表
	BatchSimilarity(embeddings1, embeddings2 [][]float32) ([]float32, error)
}

// CacheManager 缓存管理器接口
// 用于管理模型推理缓存
type CacheManager interface {
	// Get 获取缓存值
	// key: 缓存键
	// 返回: 缓存值和是否存在
	Get(key string) (interface{}, bool)

	// Set 设置缓存值
	// key: 缓存键
	// value: 缓存值
	// ttl: 过期时间
	// 返回: 错误信息
	Set(key string, value interface{}, ttl int) error

	// Delete 删除缓存值
	// key: 缓存键
	// 返回: 错误信息
	Delete(key string) error

	// Clear 清空缓存
	// 返回: 错误信息
	Clear() error

	// Size 获取缓存大小
	// 返回: 缓存项数量
	Size() int

	// Stats 获取缓存统计信息
	// 返回: 缓存统计信息
	Stats() *types.CacheStats

	// HealthCheck 健康检查
	// 返回: 健康状态
	HealthCheck() bool
}

// ModelOptimizer 模型优化器接口
// 用于模型优化和量化
type ModelOptimizer interface {
	// Quantize 量化模型
	// model: 模型实例
	// method: 量化方法
	// 返回: 错误信息
	Quantize(model Model, method string) error

	// Optimize 优化模型
	// model: 模型实例
	// optimization: 优化选项
	// 返回: 错误信息
	Optimize(model Model, optimization types.OptimizationOptions) error

	// Compile 编译模型
	// model: 模型实例
	// target: 目标平台
	// 返回: 错误信息
	Compile(model Model, target string) error

	// GetSupportedOptimizations 获取支持的优化方法
	// 返回: 优化方法列表
	GetSupportedOptimizations() []string

	// GetSupportedTargets 获取支持的目标平台
	// 返回: 目标平台列表
	GetSupportedTargets() []string

	// ValidateOptimization 验证优化选项
	// optimization: 优化选项
	// 返回: 验证错误
	ValidateOptimization(optimization types.OptimizationOptions) error
}



// 常量定义
const (
	// 模型状态
	ModelStatusLoading  = "loading"
	ModelStatusLoaded   = "loaded"
	ModelStatusUnloaded = "unloaded"
	ModelStatusError    = "error"
	ModelStatusDisabled = "disabled"

	// 健康状态
	HealthStatusHealthy   = "healthy"
	HealthStatusUnhealthy = "unhealthy"
	HealthStatusDegraded  = "degraded"

	// 量化方法
	QuantizeMethodInt8 = "int8"
	QuantizeMethodInt4 = "int4"
	QuantizeMethodFP16 = "fp16"
	QuantizeMethodBF16 = "bf16"

	// 优化选项
	OptimizationSpeed   = "speed"
	OptimizationMemory  = "memory"
	OptimizationBalance = "balance"

	// 目标平台
	TargetCPU   = "cpu"
	TargetCUDA  = "cuda"
	TargetMetal = "metal"
	TargetWASM  = "wasm"
)

// 错误定义
var (
	// 模型错误
	ErrModelNotLoaded     = errors.New("model not loaded")
	ErrModelAlreadyLoaded = errors.New("model already loaded")
	ErrModelNotFound      = errors.New("model not found")
	ErrModelTypeInvalid   = errors.New("invalid model type")
	ErrModelConfigInvalid = errors.New("invalid model configuration")

	// 推理错误
	ErrInferenceFailed    = errors.New("inference failed")
	ErrTokenLimitExceeded = errors.New("token limit exceeded")
	ErrContextTooLong     = errors.New("context too long")
	ErrInvalidTemperature = errors.New("invalid temperature value")
	ErrInvalidTopP        = errors.New("invalid top_p value")

	// 缓存错误
	ErrCacheNotAvailable = errors.New("cache not available")
	ErrCacheKeyNotFound  = errors.New("cache key not found")
	ErrCacheFull         = errors.New("cache is full")

	// 资源错误
	ErrMemoryLimitExceeded = errors.New("memory limit exceeded")
	ErrDeviceNotAvailable  = errors.New("device not available")
	ErrOutOfMemory         = errors.New("out of memory")

	// 异步错误
	ErrRequestCancelled = errors.New("request cancelled")
	ErrRequestTimeout   = errors.New("request timeout")
	ErrRequestNotFound  = errors.New("request not found")
)

// Helper functions

// IsModelHealthy 检查模型是否健康
func IsModelHealthy(health *ModelHealth) bool {
	return health != nil && health.Status == HealthStatusHealthy
}

// IsManagerHealthy 检查管理器是否健康
func IsManagerHealthy(health *ManagerHealth) bool {
	return health != nil && health.Status == HealthStatusHealthy
}

// ValidateTemperature 验证温度值
func ValidateTemperature(temperature float64) error {
	if temperature < 0 || temperature > 2 {
		return ErrInvalidTemperature
	}
	return nil
}

// ValidateTopP 验证top_p值
func ValidateTopP(topP float64) error {
	if topP < 0 || topP > 1 {
		return ErrInvalidTopP
	}
	return nil
}

// ValidateMaxTokens 验证最大token数
func ValidateMaxTokens(maxTokens int) error {
	if maxTokens < 1 || maxTokens > 10000 {
		return ErrTokenLimitExceeded
	}
	return nil
}





// GetModelTypeFromPath 从路径获取模型类型
func GetModelTypeFromPath(path string) string {
	if strings.Contains(path, "minimind") {
		return "minimind"
	}
	if strings.Contains(path, "llama") {
		return "llama"
	}
	if strings.Contains(path, "bert") {
		return "bert"
	}
	return "unknown"
}
