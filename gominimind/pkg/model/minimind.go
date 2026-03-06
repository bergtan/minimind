package model

import (
	"fmt"
	"math"
	"sync"

	"gominimind/pkg/types"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// MiniMindModel MiniMind模型实现
type MiniMindModel struct {
	config *types.ModelConfig
	logger *logrus.Logger
	status ModelStatus

	// 模型组件
	embedding EmbeddingLayer
	layers    []TransformerLayer
	lmHead    LinearLayer

	// 缓存管理
	kvCache    map[int]KeyValueCache
	cacheMutex sync.RWMutex

	// 推理状态
	isInitialized bool
	isLoaded      bool
}

// NewMiniMindModel 创建新的MiniMind模型
func NewMiniMindModel(config *types.ModelConfig) (*MiniMindModel, error) {
	model := &MiniMindModel{
		config:  config,
		logger:  logrus.New(),
		status:  ModelStatusCreated,
		kvCache: make(map[int]KeyValueCache),
	}

	// 初始化模型组件
	if err := model.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize MiniMind model: %w", err)
	}

	return model, nil
}

// initialize 初始化模型组件
func (m *MiniMindModel) initialize() error {
	m.logger.Info("Initializing MiniMind model components")

	// 初始化嵌入层
	m.embedding = NewEmbeddingLayer(m.config.VocabSize, m.config.HiddenSize)

	// 初始化Transformer层
	m.layers = make([]TransformerLayer, m.config.NumLayers)
	for i := 0; i < m.config.NumLayers; i++ {
		layer, err := NewTransformerLayer(m.config, i)
		if err != nil {
			return fmt.Errorf("failed to create transformer layer %d: %w", i, err)
		}
		m.layers[i] = layer

		// 初始化KV缓存
		if m.config.UseCache {
			m.kvCache[i] = NewKeyValueCache()
		}
	}

	// 初始化语言模型头
	m.lmHead = NewLinearLayer(m.config.HiddenSize, m.config.VocabSize, false)

	m.isInitialized = true
	m.status = ModelStatusInitialized
	m.logger.Info("MiniMind model initialized successfully")

	return nil
}

// LoadWeights 加载模型权重
func (m *MiniMindModel) LoadWeights(weightsPath string) error {
	if !m.isInitialized {
		return fmt.Errorf("model not initialized")
	}

	m.logger.Infof("Loading weights from: %s", weightsPath)

	// 模拟权重加载过程
	// 在实际实现中，这里会从文件加载权重数据

	// 加载嵌入层权重
	if err := m.embedding.LoadWeights(weightsPath); err != nil {
		return fmt.Errorf("failed to load embedding weights: %w", err)
	}

	// 加载Transformer层权重
	for i, layer := range m.layers {
		if err := layer.LoadWeights(weightsPath); err != nil {
			return fmt.Errorf("failed to load layer %d weights: %w", i, err)
		}
	}

	// 加载LM头权重
	if err := m.lmHead.LoadWeights(weightsPath); err != nil {
		return fmt.Errorf("failed to load lm head weights: %w", err)
	}

	m.isLoaded = true
	m.status = ModelStatusLoaded
	m.logger.Info("MiniMind model weights loaded successfully")

	return nil
}

// Generate 生成文本（实现Model接口）
func (m *MiniMindModel) Generate(prompt string, maxTokens int, temperature, topP float64) (string, error) {
	if !m.isLoaded {
		return "", fmt.Errorf("model weights not loaded")
	}

	m.logger.Debug("Starting text generation")

	input := &types.GenerationInput{
		Prompt:      prompt,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        topP,
	}

	// 验证输入
	if err := m.validateGenerationInput(input); err != nil {
		return "", fmt.Errorf("invalid generation input: %w", err)
	}

	// 准备输入
	inputIds := m.prepareInput(input)

	// 生成文本
	output, err := m.generateText(inputIds, input)
	if err != nil {
		return "", fmt.Errorf("generation failed: %w", err)
	}

	m.logger.Debug("Text generation completed successfully")

	return output.Text, nil
}

// GenerateWithContext 带上下文的文本生成（实现Model接口）
func (m *MiniMindModel) GenerateWithContext(messages []types.Message, maxTokens int, temperature, topP float64) (string, error) {
	if !m.isLoaded {
		return "", fmt.Errorf("model weights not loaded")
	}

	prompt := m.formatChatPrompt(messages)
	return m.Generate(prompt, maxTokens, temperature, topP)
}

// GenerateStream 流式文本生成（实现Model接口）
func (m *MiniMindModel) GenerateStream(prompt string, maxTokens int, temperature, topP float64, callback func(string) error) error {
	text, err := m.Generate(prompt, maxTokens, temperature, topP)
	if err != nil {
		return err
	}
	return callback(text)
}

// ChatCompletion 聊天补全
func (m *MiniMindModel) ChatCompletion(input *types.ChatCompletionInput) (*types.ChatCompletionOutput, error) {
	if !m.isLoaded {
		return nil, fmt.Errorf("model weights not loaded")
	}

	m.logger.Debug("Starting chat completion")

	// 验证输入
	if err := m.validateChatInput(input); err != nil {
		return nil, fmt.Errorf("invalid chat input: %w", err)
	}

	// 格式化对话为文本
	prompt := m.formatChatPrompt(input.Messages)

	// 创建生成输入
	genInput := &types.GenerationInput{
		Prompt:      prompt,
		MaxTokens:   input.MaxTokens,
		Temperature: input.Temperature,
		TopP:        input.TopP,
		StopWords:   input.Stop,
		Stream:      input.Stream,
	}

	// 生成响应
	text, err := m.Generate(genInput.Prompt, genInput.MaxTokens, genInput.Temperature, genInput.TopP)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed: %w", err)
	}

	output := &types.GenerationOutput{Text: text}

	// 格式化响应
	chatOutput := m.formatChatOutput(output, input)

	m.logger.Debug("Chat completion completed successfully")

	return chatOutput, nil
}

// Embedding 文本嵌入
func (m *MiniMindModel) Embedding(input *types.EmbeddingInput) (*types.EmbeddingOutput, error) {
	if !m.isLoaded {
		return nil, fmt.Errorf("model weights not loaded")
	}

	m.logger.Debug("Starting text embedding")

	// 统一使用Input字段，兼容Texts字段
	texts := input.Input
	if len(texts) == 0 {
		texts = input.Texts
	}

	// 验证输入
	if len(texts) == 0 {
		return nil, fmt.Errorf("invalid embedding input: input cannot be empty")
	}

	// 处理输入文本
	embeddings := make([]*mat.VecDense, len(texts))

	for i, text := range texts {
		// Tokenize文本
		tokens := m.tokenize(text)

		// 获取嵌入
		embedding, err := m.getTextEmbedding(tokens)
		if err != nil {
			return nil, fmt.Errorf("failed to get embedding for text %d: %w", i, err)
		}

		embeddings[i] = embedding
	}

	// 创建输出
	output := &types.EmbeddingOutput{
		Object: "list",
		Data:   make([]types.EmbeddingData, len(embeddings)),
		Model:  m.config.Name,
		Usage: &types.Usage{
			PromptTokens:     m.calculateTokenCount(texts),
			CompletionTokens: 0,
			TotalTokens:      m.calculateTokenCount(texts),
		},
	}

	for i, embedding := range embeddings {
		output.Data[i] = types.EmbeddingData{
			Object:    "embedding",
			Embedding: embedding.RawVector().Data,
			Index:     i,
		}
	}

	m.logger.Debug("Text embedding completed successfully")

	return output, nil
}

// GenerateEmbedding 生成单个文本嵌入（实现Model接口）
func (m *MiniMindModel) GenerateEmbedding(text string) ([]float32, error) {
	tokens := m.tokenize(text)
	vec, err := m.getTextEmbedding(tokens)
	if err != nil {
		return nil, err
	}
	data := vec.RawVector().Data
	result := make([]float32, len(data))
	for i, v := range data {
		result[i] = float32(v)
	}
	return result, nil
}

// GenerateEmbeddings 批量生成文本嵌入（实现Model接口）
func (m *MiniMindModel) GenerateEmbeddings(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := m.GenerateEmbedding(text)
		if err != nil {
			return nil, err
		}
		results[i] = emb
	}
	return results, nil
}

// GetEmbeddingDimension 获取嵌入维度（实现Model接口）
func (m *MiniMindModel) GetEmbeddingDimension() int {
	return m.config.HiddenSize
}

// GetStatus 获取模型状态（实现Model接口）
func (m *MiniMindModel) GetStatus() *types.ModelStatus {
	return &types.ModelStatus{
		ModelID: m.config.Name,
		Status:  string(m.status),
	}
}

// SetStatus 设置模型状态（实现Model接口）
func (m *MiniMindModel) SetStatus(status types.ModelStatus) error {
	m.status = ModelStatus(status.Status)
	return nil
}

// GetStatusInfo 获取模型详细状态信息
func (m *MiniMindModel) GetStatusInfo() *types.ModelStatusInfo {
	return &types.ModelStatusInfo{
		Status:    string(m.status),
		IsLoaded:  m.isLoaded,
		ModelType: m.config.ModelType,
		ModelName: m.config.Name,
		Config:    m.config,
	}
}

// Load 加载模型（实现Model接口）
func (m *MiniMindModel) Load(path string, config *types.ModelConfig) error {
	if config != nil {
		m.config = config
	}
	return m.LoadWeights(path)
}

// IsLoaded 检查模型是否已加载（实现Model接口）
func (m *MiniMindModel) IsLoaded() bool {
	return m.isLoaded
}

// GetConfig 获取模型配置（实现Model接口）
func (m *MiniMindModel) GetConfig() *types.ModelConfig {
	return m.config
}

// GetModelInfo 获取模型信息（实现Model接口）
func (m *MiniMindModel) GetModelInfo() *types.ModelInfo {
	return &types.ModelInfo{
		ID:   m.config.Name,
		Name: m.config.Name,
	}
}

// GetParameters 获取模型参数数量（实现Model接口）
func (m *MiniMindModel) GetParameters() float64 {
	return 0
}

// GetContextLength 获取上下文长度（实现Model接口）
func (m *MiniMindModel) GetContextLength() int {
	return m.config.MaxPositionEmbeddings
}

// GetVocabSize 获取词汇表大小（实现Model接口）
func (m *MiniMindModel) GetVocabSize() int {
	return m.config.VocabSize
}

// GetInferenceStats 获取推理统计信息（实现Model接口）
func (m *MiniMindModel) GetInferenceStats() *types.InferenceStats {
	return &types.InferenceStats{}
}

// ResetStats 重置统计信息（实现Model接口）
func (m *MiniMindModel) ResetStats() {}

// Save 保存模型（实现Model接口）
func (m *MiniMindModel) Save(path string) error {
	return nil
}

// Export 导出模型（实现Model接口）
func (m *MiniMindModel) Export(format string, path string) error {
	return nil
}

// Quantize 量化模型（实现Model接口）
func (m *MiniMindModel) Quantize(method string) error {
	return nil
}

// CreateContext 创建推理上下文（实现Model接口）
func (m *MiniMindModel) CreateContext() (string, error) {
	return "default", nil
}

// SetContext 设置当前上下文（实现Model接口）
func (m *MiniMindModel) SetContext(contextID string) error {
	return nil
}

// DeleteContext 删除上下文（实现Model接口）
func (m *MiniMindModel) DeleteContext(contextID string) error {
	return nil
}

// GetContexts 获取所有上下文（实现Model接口）
func (m *MiniMindModel) GetContexts() []string {
	return []string{"default"}
}

// EnableCache 启用KV缓存（实现Model接口）
func (m *MiniMindModel) EnableCache() error {
	m.config.UseCache = true
	return nil
}

// DisableCache 禁用KV缓存（实现Model接口）
func (m *MiniMindModel) DisableCache() error {
	m.config.UseCache = false
	return nil
}

// GetCacheStats 获取缓存统计信息（实现Model接口）
func (m *MiniMindModel) GetCacheStats() *types.CacheStats {
	return &types.CacheStats{}
}

// Tokenize 分词（实现Model接口）
func (m *MiniMindModel) Tokenize(text string) ([]int, error) {
	return m.tokenize(text), nil
}

// Detokenize 反分词（实现Model接口）
func (m *MiniMindModel) Detokenize(tokens []int) (string, error) {
	return m.detokenize(tokens), nil
}

// GetTokenCount 获取token数量（实现Model接口）
func (m *MiniMindModel) GetTokenCount(text string) (int, error) {
	return len(m.tokenize(text)), nil
}

// GenerateAsync 异步生成文本（实现Model接口）
func (m *MiniMindModel) GenerateAsync(prompt string, maxTokens int, temperature, topP float64) (<-chan types.AsyncResult, error) {
	ch := make(chan types.AsyncResult, 1)
	go func() {
		defer close(ch)
		text, err := m.Generate(prompt, maxTokens, temperature, topP)
		ch <- types.AsyncResult{Text: text, Error: err, Done: true}
	}()
	return ch, nil
}

// HealthCheck 健康检查（实现Model接口）
func (m *MiniMindModel) HealthCheck() (*types.ModelHealth, error) {
	status := "healthy"
	if !m.isLoaded {
		status = "not_loaded"
	}
	return &types.ModelHealth{
		Status:  status,
		Message: "ok",
	}, nil
}

// UpdateConfig 更新模型配置（实现Model接口）
func (m *MiniMindModel) UpdateConfig(config *types.ModelConfig) error {
	m.config = config
	return nil
}

// GetDefaultConfig 获取默认配置（实现Model接口）
func (m *MiniMindModel) GetDefaultConfig() *types.ModelConfig {
	return m.config
}

// ValidateConfig 验证配置（实现Model接口）
func (m *MiniMindModel) ValidateConfig(config *types.ModelConfig) []error {
	return nil
}

// SetDevice 设置设备（实现Model接口）
func (m *MiniMindModel) SetDevice(device string) error {
	m.config.Device = device
	return nil
}

// GetDevice 获取当前设备（实现Model接口）
func (m *MiniMindModel) GetDevice() string {
	return m.config.Device
}

// SetMemoryLimit 设置内存限制（实现Model接口）
func (m *MiniMindModel) SetMemoryLimit(limit int) error {
	m.config.MemoryLimit = limit
	return nil
}

// GetMemoryLimit 获取内存限制（实现Model接口）
func (m *MiniMindModel) GetMemoryLimit() int {
	return m.config.MemoryLimit
}

// RegisterExtension 注册扩展（实现Model接口）
func (m *MiniMindModel) RegisterExtension(name string, extension interface{}) error {
	return nil
}

// GetExtension 获取扩展（实现Model接口）
func (m *MiniMindModel) GetExtension(name string) interface{} {
	return nil
}

// ListExtensions 列出所有扩展（实现Model接口）
func (m *MiniMindModel) ListExtensions() []string {
	return nil
}

// SetProgressCallback 设置进度回调（实现Model接口）
func (m *MiniMindModel) SetProgressCallback(callback func(progress float32, message string)) {}

// SetErrorCallback 设置错误回调（实现Model接口）
func (m *MiniMindModel) SetErrorCallback(callback func(error)) {}

// SetLogCallback 设置日志回调（实现Model接口）
func (m *MiniMindModel) SetLogCallback(callback func(level, message string)) {}

// Unload 卸载模型
func (m *MiniMindModel) Unload() error {
	m.logger.Info("Unloading MiniMind model")

	// 清理缓存
	m.clearCache()

	// 重置状态
	m.isLoaded = false
	m.status = ModelStatusUnloaded

	m.logger.Info("MiniMind model unloaded successfully")

	return nil
}

// ========== 私有方法 ==========

// validateGenerationInput 验证生成输入
func (m *MiniMindModel) validateGenerationInput(input *types.GenerationInput) error {
	if input == nil {
		return fmt.Errorf("input cannot be nil")
	}

	if input.Prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}

	if input.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be positive")
	}

	if input.Temperature < 0 || input.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}

	if input.TopP < 0 || input.TopP > 1 {
		return fmt.Errorf("top_p must be between 0 and 1")
	}

	return nil
}

// validateChatInput 验证聊天输入
func (m *MiniMindModel) validateChatInput(input *types.ChatCompletionInput) error {
	if input == nil {
		return fmt.Errorf("input cannot be nil")
	}

	if len(input.Messages) == 0 {
		return fmt.Errorf("messages cannot be empty")
	}

	for i, msg := range input.Messages {
		if msg.Role == "" {
			return fmt.Errorf("message %d role cannot be empty", i)
		}
		if msg.Content == "" {
			return fmt.Errorf("message %d content cannot be empty", i)
		}
	}

	return nil
}

// validateEmbeddingInput 验证嵌入输入
func (m *MiniMindModel) validateEmbeddingInput(input *types.EmbeddingInput) error {
	if input == nil {
		return fmt.Errorf("input cannot be nil")
	}
	texts := input.Input
	if len(texts) == 0 {
		texts = input.Texts
	}
	if len(texts) == 0 {
		return fmt.Errorf("input cannot be empty")
	}
	for i, text := range texts {
		if text == "" {
			return fmt.Errorf("input text %d cannot be empty", i)
		}
	}
	return nil
}

// prepareInput 准备输入数据（内部使用）
func (m *MiniMindModel) prepareInput(input *types.GenerationInput) *mat.VecDense {
	// Tokenize输入文本
	tokens := m.tokenize(input.Prompt)

	// 转换为向量
	tokensFloat := m.intToFloat(tokens)
	inputIds := mat.NewVecDense(len(tokens), tokensFloat)

	return inputIds
}

// generateText 生成文本
func (m *MiniMindModel) generateText(inputIds *mat.VecDense, input *types.GenerationInput) (*types.GenerationOutput, error) {
	// 前向传播（忽略初始隐藏状态，直接进入生成循环）
	_, err := m.forward(inputIds)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %w", err)
	}

	// 生成文本
	generatedTokens := make([]int, 0, input.MaxTokens)
	currentInput := inputIds

	for i := 0; i < input.MaxTokens; i++ {
		// 获取下一个token的logits
		logits, err := m.getNextTokenLogits(currentInput)
		if err != nil {
			return nil, fmt.Errorf("failed to get next token logits: %w", err)
		}

		// 采样下一个token
		nextToken, err := m.sampleNextToken(logits, input.Temperature, input.TopP)
		if err != nil {
			return nil, fmt.Errorf("failed to sample next token: %w", err)
		}

		// 检查停止条件
		if m.shouldStop(nextToken, input.StopWords, generatedTokens) {
			break
		}

		// 添加token到生成序列
		generatedTokens = append(generatedTokens, nextToken)

		// 更新当前输入
		currentInput = mat.NewVecDense(1, []float64{float64(nextToken)})
	}

	// 解码token为文本
	text := m.detokenize(generatedTokens)

	// 创建输出
	usage := &types.Usage{
		PromptTokens:     inputIds.Len(),
		CompletionTokens: len(generatedTokens),
		TotalTokens:      inputIds.Len() + len(generatedTokens),
	}
	output := &types.GenerationOutput{
		Text:   text,
		Tokens: generatedTokens,
		Usage:  usage,
	}

	return output, nil
}

// forward 前向传播
func (m *MiniMindModel) forward(inputIds *mat.VecDense) (*mat.Dense, error) {
	// 获取嵌入
	embeddings, err := m.embedding.Forward(inputIds)
	if err != nil {
		return nil, fmt.Errorf("embedding forward failed: %w", err)
	}

	// 通过Transformer层
	hiddenStates := embeddings
	for i, layer := range m.layers {
		var pastKeyValue KeyValueCache
		if m.config.UseCache {
			pastKeyValue = m.kvCache[i]
		}

		hiddenStates, err = layer.Forward(hiddenStates, pastKeyValue)
		if err != nil {
			return nil, fmt.Errorf("layer %d forward failed: %w", i, err)
		}
	}

	return hiddenStates, nil
}

// getNextTokenLogits 获取下一个token的logits
func (m *MiniMindModel) getNextTokenLogits(inputIds *mat.VecDense) (*mat.VecDense, error) {
	// 前向传播获取隐藏状态
	hiddenStates, err := m.forward(inputIds)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %w", err)
	}

	// 获取最后一个token的隐藏状态
	lastHiddenState := mat.NewVecDense(hiddenStates.RawMatrix().Rows, nil)
	for r := 0; r < hiddenStates.RawMatrix().Rows; r++ {
		lastHiddenState.SetVec(r, hiddenStates.At(r, hiddenStates.RawMatrix().Cols-1))
	}

	// 通过LM头获取logits
	logits, err := m.lmHead.Forward(lastHiddenState)
	if err != nil {
		return nil, fmt.Errorf("lm head forward failed: %w", err)
	}

	return logits, nil
}

// sampleNextToken 采样下一个token
func (m *MiniMindModel) sampleNextToken(logits *mat.VecDense, temperature, topP float64) (int, error) {
	// 应用温度缩放
	if temperature != 1.0 {
		m.applyTemperature(logits, temperature)
	}

	// 应用top-p采样
	if topP < 1.0 {
		m.applyTopP(logits, topP)
	}

	// 应用softmax
	probabilities := m.softmax(logits)

	// 采样
	token := m.sampleFromDistribution(probabilities)

	return token, nil
}

// applyTemperature 应用温度缩放
func (m *MiniMindModel) applyTemperature(logits *mat.VecDense, temperature float64) {
	data := logits.RawVector().Data
	for i := range data {
		data[i] /= temperature
	}
}

// applyTopP 应用top-p采样
func (m *MiniMindModel) applyTopP(logits *mat.VecDense, topP float64) {
	data := logits.RawVector().Data

	// 复制并排序logits
	sortedLogits := make([]float64, len(data))
	copy(sortedLogits, data)
	floats.Argsort(sortedLogits, nil)

	// 计算累积概率
	probabilities := m.softmax(mat.NewVecDense(len(data), data))
	cumulativeProb := 0.0
	cutoffIndex := len(data)

	for i := len(data) - 1; i >= 0; i-- {
		cumulativeProb += probabilities.At(i, 0)
		if cumulativeProb > topP {
			cutoffIndex = i
			break
		}
	}

	// 将低于cutoff的logits设为负无穷
	for i := 0; i < cutoffIndex; i++ {
		data[i] = math.Inf(-1)
	}
}

// softmax 计算softmax
func (m *MiniMindModel) softmax(logits *mat.VecDense) *mat.VecDense {
	data := logits.RawVector().Data

	// 减去最大值以提高数值稳定性
	maxVal := floats.Max(data)
	for i := range data {
		data[i] -= maxVal
	}

	// 计算指数和
	expSum := 0.0
	for i := range data {
		data[i] = math.Exp(data[i])
		expSum += data[i]
	}

	// 归一化
	for i := range data {
		data[i] /= expSum
	}

	return mat.NewVecDense(len(data), data)
}

// sampleFromDistribution 从分布中采样
func (m *MiniMindModel) sampleFromDistribution(probabilities *mat.VecDense) int {
	data := probabilities.RawVector().Data

	// 生成随机数
	randVal := m.randomFloat()

	// 累积概率采样
	cumulativeProb := 0.0
	for i, prob := range data {
		cumulativeProb += prob
		if randVal <= cumulativeProb {
			return i
		}
	}

	// 如果由于浮点误差没有采样到，返回最后一个token
	return len(data) - 1
}

// randomFloat 生成随机浮点数
func (m *MiniMindModel) randomFloat() float64 {
	// 在实际实现中，这里会使用加密安全的随机数生成器
	return 0.5 // 简化实现
}

// shouldStop 检查是否应该停止生成
func (m *MiniMindModel) shouldStop(token int, stopTokens []string, generatedTokens []int) bool {
	// 检查停止token
	for _, stopToken := range stopTokens {
		stopTokenId := m.tokenize(stopToken)[0]
		if token == stopTokenId {
			return true
		}
	}

	// 检查结束token
	if token == m.getEOSToken() {
		return true
	}

	return false
}

// formatChatPrompt 格式化聊天提示
func (m *MiniMindModel) formatChatPrompt(messages []types.Message) string {
	var prompt string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			prompt += fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n", msg.Content)
		case "user":
			prompt += fmt.Sprintf("<|im_start|>user\n%s<|im_start|>\n", msg.Content)
		case "assistant":
			prompt += fmt.Sprintf("<|im_start|>assistant\n%s<|im_end|>\n", msg.Content)
		}
	}

	prompt += "<|im_start|>assistant\n"

	return prompt
}

// formatChatOutput 格式化聊天输出
func (m *MiniMindModel) formatChatOutput(output *types.GenerationOutput, input *types.ChatCompletionInput) *types.ChatCompletionOutput {
	chatOutput := &types.ChatCompletionOutput{
		ID:      m.generateID(),
		Object:  "chat.completion",
		Created: m.getCurrentTimestamp(),
		Model:   m.config.Name,
		Choices: []types.ChatChoice{
			{
				Index: 0,
				Message: types.ChatMessage{
					Role:    "assistant",
					Content: output.Text,
				},
				FinishReason: "stop",
			},
		},
		Usage: output.Usage,
	}

	return chatOutput
}

// validateEmbeddingInput 验证嵌入输入（兼容Input和Texts字段）
func (m *MiniMindModel) validateEmbeddingInputCompat(input *types.EmbeddingInput) error {
	if input == nil {
		return fmt.Errorf("input cannot be nil")
	}
	texts := input.Input
	if len(texts) == 0 {
		texts = input.Texts
	}
	if len(texts) == 0 {
		return fmt.Errorf("input cannot be empty")
	}
	return nil
}

// getTextEmbedding 获取文本嵌入
func (m *MiniMindModel) getTextEmbedding(tokens []int) (*mat.VecDense, error) {
	// 获取token嵌入
	inputIds := mat.NewVecDense(len(tokens), m.intToFloat(tokens))
	embeddings, err := m.embedding.Forward(inputIds)
	if err != nil {
		return nil, fmt.Errorf("embedding forward failed: %w", err)
	}

	// 平均池化获取文本嵌入
	textEmbedding := m.meanPooling(embeddings)

	return textEmbedding, nil
}

// meanPooling 平均池化
func (m *MiniMindModel) meanPooling(embeddings *mat.Dense) *mat.VecDense {
	rows, cols := embeddings.Dims()
	meanEmbedding := mat.NewVecDense(cols, nil)

	for i := 0; i < rows; i++ {
		row := embeddings.RowView(i)
		meanEmbedding.AddVec(meanEmbedding, row)
	}

	meanEmbedding.ScaleVec(1.0/float64(rows), meanEmbedding)

	return meanEmbedding
}

// tokenize Tokenize文本
func (m *MiniMindModel) tokenize(text string) []int {
	// 简化实现：按字符编码
	tokens := make([]int, len(text))
	for i, char := range text {
		tokens[i] = int(char)
	}
	return tokens
}

// detokenize Detokenize tokens
func (m *MiniMindModel) detokenize(tokens []int) string {
	// 简化实现：按字符解码
	text := make([]rune, len(tokens))
	for i, token := range tokens {
		text[i] = rune(token)
	}
	return string(text)
}

// calculateTokenCount 计算token数量
func (m *MiniMindModel) calculateTokenCount(texts []string) int {
	totalTokens := 0
	for _, text := range texts {
		totalTokens += len(m.tokenize(text))
	}
	return totalTokens
}

// getEOSToken 获取结束token
func (m *MiniMindModel) getEOSToken() int {
	return 2 // 简化实现
}

// generateID 生成唯一ID
func (m *MiniMindModel) generateID() string {
	return "chatcmpl-1234567890abcdef" // 简化实现
}

// getCurrentTimestamp 获取当前时间戳
func (m *MiniMindModel) getCurrentTimestamp() int64 {
	return 1677652288 // 简化实现
}

// intToFloat 将int切片转换为float64切片
func (m *MiniMindModel) intToFloat(ints []int) []float64 {
	floats := make([]float64, len(ints))
	for i, val := range ints {
		floats[i] = float64(val)
	}
	return floats
}

// clearCache 清理缓存
func (m *MiniMindModel) clearCache() {
	m.cacheMutex.Lock()
	defer m.cacheMutex.Unlock()

	for i := range m.kvCache {
		m.kvCache[i] = NewKeyValueCache()
	}
}

// ========== 模型组件接口 ==========

// EmbeddingLayer 嵌入层接口
type EmbeddingLayer interface {
	Forward(input *mat.VecDense) (*mat.Dense, error)
	LoadWeights(weightsPath string) error
}

// TransformerLayer Transformer层接口
type TransformerLayer interface {
	Forward(input *mat.Dense, pastKeyValue KeyValueCache) (*mat.Dense, error)
	LoadWeights(weightsPath string) error
}

// LinearLayer 线性层接口
type LinearLayer interface {
	Forward(input *mat.VecDense) (*mat.VecDense, error)
	LoadWeights(weightsPath string) error
}

// KeyValueCache KV缓存接口
type KeyValueCache interface {
	GetKeys() *mat.Dense
	GetValues() *mat.Dense
	Update(keys, values *mat.Dense)
	Clear()
}

// ========== 模型组件实现 ==========

// NewEmbeddingLayer 创建新的嵌入层
func NewEmbeddingLayer(vocabSize, hiddenSize int) *EmbeddingLayerImpl {
	return &EmbeddingLayerImpl{
		vocabSize:  vocabSize,
		hiddenSize: hiddenSize,
		weights:    mat.NewDense(vocabSize, hiddenSize, nil),
	}
}

// EmbeddingLayerImpl 嵌入层实现
type EmbeddingLayerImpl struct {
	vocabSize  int
	hiddenSize int
	weights    *mat.Dense
}

func (e *EmbeddingLayerImpl) Forward(input *mat.VecDense) (*mat.Dense, error) {
	// 嵌入查表: 将token ID映射为嵌入向量
	seqLen := input.Len()
	result := mat.NewDense(seqLen, e.hiddenSize, nil)

	for i := 0; i < seqLen; i++ {
		tokenID := int(input.AtVec(i))
		if tokenID >= 0 && tokenID < e.vocabSize {
			// 从权重矩阵中查找对应行
			for j := 0; j < e.hiddenSize; j++ {
				result.Set(i, j, e.weights.At(tokenID, j))
			}
		}
		// tokenID超出范围时保持零向量
	}

	return result, nil
}

func (e *EmbeddingLayerImpl) LoadWeights(weightsPath string) error {
	// 实现权重加载
	return nil // 简化实现
}

// NewTransformerLayer 创建新的Transformer层
func NewTransformerLayer(config *types.ModelConfig, layerIndex int) (*TransformerLayerImpl, error) {
	return &TransformerLayerImpl{
		config:     config,
		layerIndex: layerIndex,
	}, nil
}

// TransformerLayerImpl Transformer层实现
type TransformerLayerImpl struct {
	config     *types.ModelConfig
	layerIndex int
}

func (t *TransformerLayerImpl) Forward(input *mat.Dense, pastKeyValue KeyValueCache) (*mat.Dense, error) {
	// 简化的Transformer层前向传播
	// 实际实现应包含完整的注意力计算和前馈网络
	// 这里执行一个简单的线性变换+残差连接
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Copy(input)

	// 添加微小的变换以模拟Transformer层效果
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			// 简单的缩放（模拟残差连接后的效果）
			output.Set(i, j, val*0.99)
		}
	}

	return output, nil
}

func (t *TransformerLayerImpl) LoadWeights(weightsPath string) error {
	// 实现权重加载
	return nil // 简化实现
}

// NewLinearLayer 创建新的线性层
func NewLinearLayer(inputSize, outputSize int, useBias bool) *LinearLayerImpl {
	return &LinearLayerImpl{
		inputSize:  inputSize,
		outputSize: outputSize,
		useBias:    useBias,
		weights:    mat.NewDense(inputSize, outputSize, nil),
		bias:       nil,
	}
}

// LinearLayerImpl 线性层实现
type LinearLayerImpl struct {
	inputSize  int
	outputSize int
	useBias    bool
	weights    *mat.Dense
	bias       *mat.VecDense
}

func (l *LinearLayerImpl) Forward(input *mat.VecDense) (*mat.VecDense, error) {
	// 线性变换: output = input @ weights + bias
	outputData := make([]float64, l.outputSize)
	inputData := input.RawVector().Data

	for j := 0; j < l.outputSize; j++ {
		sum := 0.0
		for i := 0; i < l.inputSize && i < len(inputData); i++ {
			sum += inputData[i] * l.weights.At(i, j)
		}
		if l.useBias && l.bias != nil {
			sum += l.bias.AtVec(j)
		}
		outputData[j] = sum
	}

	return mat.NewVecDense(l.outputSize, outputData), nil
}

func (l *LinearLayerImpl) LoadWeights(weightsPath string) error {
	// 实现权重加载
	return nil // 简化实现
}

// NewKeyValueCache 创建新的KV缓存
func NewKeyValueCache() *KeyValueCacheImpl {
	return &KeyValueCacheImpl{
		keys:   mat.NewDense(1, 1, nil),
		values: mat.NewDense(1, 1, nil),
	}
}

// KeyValueCacheImpl KV缓存实现
type KeyValueCacheImpl struct {
	keys   *mat.Dense
	values *mat.Dense
}

func (k *KeyValueCacheImpl) GetKeys() *mat.Dense {
	return k.keys
}

func (k *KeyValueCacheImpl) GetValues() *mat.Dense {
	return k.values
}

func (k *KeyValueCacheImpl) Update(keys, values *mat.Dense) {
	// 实现缓存更新
}

func (k *KeyValueCacheImpl) Clear() {
	k.keys = mat.NewDense(0, 0, nil)
	k.values = mat.NewDense(0, 0, nil)
}

// BatchGenerate 批量生成文本
func (m *MiniMindModel) BatchGenerate(prompts []string, maxTokens int, temperature, topP float64) ([]string, error) {
	results := make([]string, len(prompts))

	for i, prompt := range prompts {
		result, err := m.Generate(prompt, maxTokens, temperature, topP)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}

	return results, nil
}

// BatchGenerateWithContext 带上下文的批量生成文本
func (m *MiniMindModel) BatchGenerateWithContext(messages [][]types.Message, maxTokens int, temperature, topP float64) ([]string, error) {
	results := make([]string, len(messages))

	for i, messageSet := range messages {
		result, err := m.GenerateWithContext(messageSet, maxTokens, temperature, topP)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}

	return results, nil
}

// GetMemoryUsage 获取内存使用情况
func (m *MiniMindModel) GetMemoryUsage() (*types.MemoryUsage, error) {
	// 这里需要实现实际的内存使用统计逻辑
	// 暂时返回一个默认的内存使用信息
	return &types.MemoryUsage{
		TotalMemory:     1024 * 1024 * 1024, // 1GB
		UsedMemory:      512 * 1024 * 1024,  // 512MB
		FreeMemory:      512 * 1024 * 1024,  // 512MB
		MemoryUsageRate: 0.5,                // 50%
		PeakMemoryUsage: 768 * 1024 * 1024,  // 768MB
	}, nil
}

// CancelGeneration 取消正在进行的生成任务
func (m *MiniMindModel) CancelGeneration(taskID string) error {
	// 这里需要实现取消生成任务的逻辑
	// 暂时返回nil表示成功
	return nil
}

// ClearCache 清除缓存
func (m *MiniMindModel) ClearCache() error {
	// 这里需要实现清除缓存的逻辑
	// 暂时返回nil表示成功
	return nil
}
