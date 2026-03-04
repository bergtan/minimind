package model

import (
	"fmt"
	"sync"

	"github.com/jingyaogong/gominimind/pkg/types"
	"github.com/sirupsen/logrus"
)

// ModelFactory 模型工厂接口
type ModelFactory interface {
	// CreateModel 创建模型实例
	CreateModel(modelType string, config *types.ModelConfig) (Model, error)

	// ValidateModelType 验证模型类型是否支持
	ValidateModelType(modelType string) bool

	// GetSupportedTypes 获取支持的模型类型列表
	GetSupportedTypes() []string

	// GetDefaultConfig 获取默认模型配置
	GetDefaultConfig(modelType string) *types.ModelConfig

	// RegisterModelType 注册新的模型类型
	RegisterModelType(modelType string, creator ModelCreator) error

	// UnregisterModelType 注销模型类型
	UnregisterModelType(modelType string) error
}

// ModelCreator 模型创建器函数类型
type ModelCreator func(config *types.ModelConfig) (Model, error)

// ModelFactoryImpl 模型工厂实现
type ModelFactoryImpl struct {
	mu sync.RWMutex

	// 模型创建器注册表
	creators map[string]ModelCreator

	// 默认配置映射
	defaultConfigs map[string]*types.ModelConfig

	// 日志记录器
	logger *logrus.Logger
}

// NewModelFactory 创建新的模型工厂
func NewModelFactory() *ModelFactoryImpl {
	factory := &ModelFactoryImpl{
		creators:       make(map[string]ModelCreator),
		defaultConfigs: make(map[string]*types.ModelConfig),
		logger:         logrus.New(),
	}

	// 注册内置模型类型
	factory.registerBuiltinTypes()

	return factory
}

// CreateModel 创建模型实例
func (f *ModelFactoryImpl) CreateModel(modelType string, config *types.ModelConfig) (Model, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	// 检查模型类型是否支持
	creator, exists := f.creators[modelType]
	if !exists {
		return nil, fmt.Errorf("unsupported model type: %s", modelType)
	}

	// 如果配置为空，使用默认配置
	if config == nil {
		config = f.GetDefaultConfig(modelType)
	}

	// 验证配置
	if err := ValidateModelConfig(config); err != nil {
		return nil, fmt.Errorf("invalid model config: %w", err)
	}

	f.logger.Infof("Creating model of type %s", modelType)

	// 创建模型实例
	model, err := creator(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create model: %w", err)
	}

	f.logger.Infof("Model created successfully: %s", modelType)

	return model, nil
}

// ValidateModelType 验证模型类型是否支持
func (f *ModelFactoryImpl) ValidateModelType(modelType string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()

	_, exists := f.creators[modelType]
	return exists
}

// GetSupportedTypes 获取支持的模型类型列表
func (f *ModelFactoryImpl) GetSupportedTypes() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	types := make([]string, 0, len(f.creators))
	for modelType := range f.creators {
		types = append(types, modelType)
	}

	return types
}

// GetDefaultConfig 获取默认模型配置
func (f *ModelFactoryImpl) GetDefaultConfig(modelType string) *types.ModelConfig {
	f.mu.RLock()
	defer f.mu.RUnlock()

	config, exists := f.defaultConfigs[modelType]
	if !exists {
		// 返回通用默认配置
		return GetDefaultModelConfig()
	}

	return config
}

// RegisterModelType 注册新的模型类型
func (f *ModelFactoryImpl) RegisterModelType(modelType string, creator ModelCreator) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if modelType == "" {
		return fmt.Errorf("model type cannot be empty")
	}

	if creator == nil {
		return fmt.Errorf("model creator cannot be nil")
	}

	if _, exists := f.creators[modelType]; exists {
		return fmt.Errorf("model type already registered: %s", modelType)
	}

	f.creators[modelType] = creator

	// 设置默认配置
	f.defaultConfigs[modelType] = GetDefaultModelConfig()

	f.logger.Infof("Registered new model type: %s", modelType)

	return nil
}

// UnregisterModelType 注销模型类型
func (f *ModelFactoryImpl) UnregisterModelType(modelType string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if _, exists := f.creators[modelType]; !exists {
		return fmt.Errorf("model type not registered: %s", modelType)
	}

	delete(f.creators, modelType)
	delete(f.defaultConfigs, modelType)

	f.logger.Infof("Unregistered model type: %s", modelType)

	return nil
}

// registerBuiltinTypes 注册内置模型类型
func (f *ModelFactoryImpl) registerBuiltinTypes() {
	// 注册MiniMind模型类型
	f.RegisterModelType("minimind", func(config *types.ModelConfig) (Model, error) {
		return NewMiniMindModel(config)
	})

	// 注册Llama模型类型
	f.RegisterModelType("llama", func(config *types.ModelConfig) (Model, error) {
		return NewLlamaModel(config)
	})

	// 注册BERT模型类型
	f.RegisterModelType("bert", func(config *types.ModelConfig) (Model, error) {
		return NewBERTModel(config)
	})

	// 注册GPT模型类型
	f.RegisterModelType("gpt", func(config *types.ModelConfig) (Model, error) {
		return NewGPTModel(config)
	})

	// 设置特定模型的默认配置
	f.setModelSpecificDefaults()
}

// setModelSpecificDefaults 设置模型特定的默认配置
func (f *ModelFactoryImpl) setModelSpecificDefaults() {
	// MiniMind模型默认配置
	f.defaultConfigs["minimind"] = &types.ModelConfig{
		Name:                  "MiniMind",
		ModelType:             "minimind",
		VocabSize:             6400,
		HiddenSize:            512,
		NumLayers:             8,
		NumHeads:              8,
		MaxPositionEmbeddings: 32768,
		IntermediateSize:      2048,
		HiddenAct:             "swiglu",
		InitializerRange:      0.02,
		LayerNormEps:          1e-5,
		UseCache:              true,
		TorchDtype:            "float32",
		RopeTheta:             10000.0,
		AttentionBias:         false,
		AttentionDropout:      0.0,
		HiddenDropout:         0.0,
	}

	// Llama模型默认配置
	f.defaultConfigs["llama"] = &types.ModelConfig{
		Name:                  "Llama",
		ModelType:             "llama",
		VocabSize:             32000,
		HiddenSize:            4096,
		NumLayers:             32,
		NumHeads:              32,
		MaxPositionEmbeddings: 2048,
		IntermediateSize:      11008,
		HiddenAct:             "silu",
		InitializerRange:      0.02,
		LayerNormEps:          1e-5,
		UseCache:              true,
		TorchDtype:            "float32",
		RopeTheta:             10000.0,
		AttentionBias:         false,
		AttentionDropout:      0.1,
		HiddenDropout:         0.1,
	}

	// BERT模型默认配置
	f.defaultConfigs["bert"] = &types.ModelConfig{
		Name:                  "BERT",
		ModelType:             "bert",
		VocabSize:             30522,
		HiddenSize:            768,
		NumLayers:             12,
		NumHeads:              12,
		MaxPositionEmbeddings: 512,
		IntermediateSize:      3072,
		HiddenAct:             "gelu",
		InitializerRange:      0.02,
		LayerNormEps:          1e-12,
		UseCache:              false,
		TorchDtype:            "float32",
		AttentionBias:         true,
		AttentionDropout:      0.1,
		HiddenDropout:         0.1,
	}

	// GPT模型默认配置
	f.defaultConfigs["gpt"] = &types.ModelConfig{
		Name:                  "GPT",
		ModelType:             "gpt",
		VocabSize:             50257,
		HiddenSize:            768,
		NumLayers:             12,
		NumHeads:              12,
		MaxPositionEmbeddings: 1024,
		IntermediateSize:      3072,
		HiddenAct:             "gelu",
		InitializerRange:      0.02,
		LayerNormEps:          1e-5,
		UseCache:              true,
		TorchDtype:            "float32",
		AttentionBias:         true,
		AttentionDropout:      0.1,
		HiddenDropout:         0.1,
	}
}

// GetDefaultModelConfig 获取通用默认模型配置
func GetDefaultModelConfig() *types.ModelConfig {
	return &types.ModelConfig{
		Name:                  "DefaultModel",
		ModelType:             "transformer",
		VocabSize:             32000,
		HiddenSize:            768,
		NumLayers:             12,
		NumHeads:              12,
		MaxPositionEmbeddings: 2048,
		IntermediateSize:      3072,
		HiddenAct:             "gelu",
		InitializerRange:      0.02,
		LayerNormEps:          1e-5,
		UseCache:              true,
		TorchDtype:            "float32",
		AttentionBias:         false,
		AttentionDropout:      0.1,
		HiddenDropout:         0.1,
	}
}

// ========== 模型创建器函数 ==========

// NewMiniMindModel 创建MiniMind模型实例
func NewMiniMindModel(config *types.ModelConfig) (Model, error) {
	// 验证配置
	if err := ValidateMiniMindConfig(config); err != nil {
		return nil, err
	}

	model := &MiniMindModel{
		config: config,
		logger: logrus.New(),
		status: ModelStatusCreated,
	}

	// 初始化模型组件
	if err := model.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize MiniMind model: %w", err)
	}

	return model, nil
}

// NewLlamaModel 创建Llama模型实例
func NewLlamaModel(config *types.ModelConfig) (Model, error) {
	// 验证配置
	if err := ValidateLlamaConfig(config); err != nil {
		return nil, err
	}

	model := &LlamaModel{
		config: config,
		logger: logrus.New(),
		status: ModelStatusCreated,
	}

	// 初始化模型组件
	if err := model.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize Llama model: %w", err)
	}

	return model, nil
}

// NewBERTModel 创建BERT模型实例
func NewBERTModel(config *types.ModelConfig) (Model, error) {
	// 验证配置
	if err := ValidateBERTConfig(config); err != nil {
		return nil, err
	}

	model := &BERTModel{
		config: config,
		logger: logrus.New(),
		status: ModelStatusCreated,
	}

	// 初始化模型组件
	if err := model.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize BERT model: %w", err)
	}

	return model, nil
}

// NewGPTModel 创建GPT模型实例
func NewGPTModel(config *types.ModelConfig) (Model, error) {
	// 验证配置
	if err := ValidateGPTConfig(config); err != nil {
		return nil, err
	}

	model := &GPTModel{
		config: config,
		logger: logrus.New(),
		status: ModelStatusCreated,
	}

	// 初始化模型组件
	if err := model.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize GPT model: %w", err)
	}

	return model, nil
}

// ========== 配置验证函数 ==========

// ValidateMiniMindConfig 验证MiniMind模型配置
func ValidateMiniMindConfig(config *types.ModelConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	if config.VocabSize != 6400 {
		return fmt.Errorf("MiniMind vocab size must be 6400")
	}

	if config.HiddenSize != 512 {
		return fmt.Errorf("MiniMind hidden size must be 512")
	}

	if config.NumLayers != 8 {
		return fmt.Errorf("MiniMind number of layers must be 8")
	}

	if config.NumHeads != 8 {
		return fmt.Errorf("MiniMind number of heads must be 8")
	}

	if config.MaxPositionEmbeddings != 32768 {
		return fmt.Errorf("MiniMind max position embeddings must be 32768")
	}

	if config.HiddenAct != "swiglu" {
		return fmt.Errorf("MiniMind hidden activation must be swiglu")
	}

	return nil
}

// ValidateLlamaConfig 验证Llama模型配置
func ValidateLlamaConfig(config *types.ModelConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	if config.HiddenAct != "silu" {
		return fmt.Errorf("Llama hidden activation must be silu")
	}

	if config.RopeTheta == 0 {
		return fmt.Errorf("Llama rope theta must be set")
	}

	return nil
}

// ValidateBERTConfig 验证BERT模型配置
func ValidateBERTConfig(config *types.ModelConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	if config.LayerNormEps != 1e-12 {
		return fmt.Errorf("BERT layer norm epsilon must be 1e-12")
	}

	if !config.AttentionBias {
		return fmt.Errorf("BERT attention bias must be true")
	}

	return nil
}

// ValidateGPTConfig 验证GPT模型配置
func ValidateGPTConfig(config *types.ModelConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	if config.HiddenAct != "gelu" {
		return fmt.Errorf("GPT hidden activation must be gelu")
	}

	if !config.AttentionBias {
		return fmt.Errorf("GPT attention bias must be true")
	}

	return nil
}

// ========== 工厂工具函数 ==========

// CreateModelFromPath 从模型路径创建模型
func CreateModelFromPath(modelPath string, factory ModelFactory) (Model, error) {
	// 从路径推断模型类型
	modelType := InferModelTypeFromPath(modelPath)
	if modelType == "" {
		return nil, fmt.Errorf("cannot infer model type from path: %s", modelPath)
	}

	// 创建模型配置
	config := factory.GetDefaultConfig(modelType)
	config.ModelPath = modelPath

	// 创建模型实例
	return factory.CreateModel(modelType, config)
}

// InferModelTypeFromPath 从模型路径推断模型类型
func InferModelTypeFromPath(modelPath string) string {
	// 根据路径中的关键词推断模型类型
	if Contains(modelPath, "minimind") {
		return "minimind"
	} else if Contains(modelPath, "llama") {
		return "llama"
	} else if Contains(modelPath, "bert") {
		return "bert"
	} else if Contains(modelPath, "gpt") {
		return "gpt"
	}

	// 尝试从配置文件推断
	return ""
}

// Contains 检查字符串是否包含子字符串
func Contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		len(s) > len(substr) && (s[:len(substr)] == substr ||
			Contains(s[1:], substr)))
}

// CreateModelWithAutoConfig 使用自动配置创建模型
func CreateModelWithAutoConfig(modelType string, factory ModelFactory, autoConfig bool) (Model, error) {
	var config *types.ModelConfig

	if autoConfig {
		// 自动配置：从预训练模型加载配置
		config = LoadConfigFromPretrained(modelType)
	} else {
		config = factory.GetDefaultConfig(modelType)
	}

	return factory.CreateModel(modelType, config)
}

// LoadConfigFromPretrained 从预训练模型加载配置
func LoadConfigFromPretrained(modelType string) *types.ModelConfig {
	// 这里可以扩展为从HuggingFace Hub或其他模型仓库加载配置
	// 目前返回默认配置
	return GetDefaultModelConfig()
}

// ========== 工厂单例模式 ==========

var (
	factoryInstance *ModelFactoryImpl
	factoryOnce     sync.Once
)

// GetDefaultFactory 获取默认模型工厂（单例模式）
func GetDefaultFactory() *ModelFactoryImpl {
	factoryOnce.Do(func() {
		factoryInstance = NewModelFactory()
	})
	return factoryInstance
}

// ResetFactory 重置工厂（主要用于测试）
func ResetFactory() {
	factoryInstance = nil
	factoryOnce = sync.Once{}
}

// ========== 工厂扩展接口 ==========

// ModelFactoryExtension 模型工厂扩展接口
type ModelFactoryExtension interface {
	// PreCreateHook 模型创建前钩子
	PreCreateHook(modelType string, config *types.ModelConfig) error

	// PostCreateHook 模型创建后钩子
	PostCreateHook(model Model, modelType string) error

	// ValidationHook 配置验证钩子
	ValidationHook(config *types.ModelConfig) error
}

// ExtendedModelFactory 扩展模型工厂
type ExtendedModelFactory struct {
	*ModelFactoryImpl
	extensions []ModelFactoryExtension
}

// NewExtendedModelFactory 创建扩展模型工厂
func NewExtendedModelFactory(extensions ...ModelFactoryExtension) *ExtendedModelFactory {
	return &ExtendedModelFactory{
		ModelFactoryImpl: NewModelFactory(),
		extensions:       extensions,
	}
}

// CreateModel 扩展创建模型方法
func (f *ExtendedModelFactory) CreateModel(modelType string, config *types.ModelConfig) (Model, error) {
	// 执行前置钩子
	for _, ext := range f.extensions {
		if err := ext.PreCreateHook(modelType, config); err != nil {
			return nil, fmt.Errorf("pre-create hook failed: %w", err)
		}
	}

	// 执行验证钩子
	for _, ext := range f.extensions {
		if err := ext.ValidationHook(config); err != nil {
			return nil, fmt.Errorf("validation hook failed: %w", err)
		}
	}

	// 创建模型
	model, err := f.ModelFactoryImpl.CreateModel(modelType, config)
	if err != nil {
		return nil, err
	}

	// 执行后置钩子
	for _, ext := range f.extensions {
		if err := ext.PostCreateHook(model, modelType); err != nil {
			return nil, fmt.Errorf("post-create hook failed: %w", err)
		}
	}

	return model, nil
}
