package model

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/jingyaogong/gominimind/pkg/types"
	"github.com/sirupsen/logrus"
)

// ModelManagerImpl 模型管理器实现
type ModelManagerImpl struct {
	mu sync.RWMutex

	// 模型存储
	models map[string]Model

	// 模型信息存储
	modelInfos map[string]*types.ModelInfo

	// 模型状态存储
	modelStatus map[string]*types.ModelStatus

	// 默认模型ID
	defaultModelID string

	// 模型工厂
	factory ModelFactory

	// 配置
	config *types.ManagerConfig

	// 统计信息
	stats *types.ManagerStats

	// 日志记录器
	logger *logrus.Logger

	// 上下文
	ctx    context.Context
	cancel context.CancelFunc

	// 健康检查间隔
	healthCheckInterval time.Duration

	// 资源监控
	resourceMonitor *ResourceMonitor
}

// ResourceMonitor 资源监控器
type ResourceMonitor struct {
	mu sync.RWMutex

	// 内存使用监控
	memoryUsage map[string]uint64

	// GPU使用监控
	gpuUsage map[string]float32

	// 性能指标
	performanceMetrics map[string]*PerformanceMetrics

	// 告警阈值
	alerts map[string]*AlertThreshold
}

// PerformanceMetrics 性能指标
type PerformanceMetrics struct {
	TotalRequests    int64
	FailedRequests   int64
	AvgResponseTime  float64
	LastResponseTime time.Time
	Throughput       float64
}

// AlertThreshold 告警阈值
type AlertThreshold struct {
	MemoryThreshold       uint64
	GPUThreshold          float32
	ResponseTimeThreshold float64
	ErrorRateThreshold    float64
}

// NewModelManager 创建新的模型管理器
func NewModelManager(config *types.ManagerConfig, factory ModelFactory) (*ModelManagerImpl, error) {
	if config == nil {
		config = getDefaultManagerConfig()
	}

	if factory == nil {
		return nil, fmt.Errorf("model factory is required")
	}

	ctx, cancel := context.WithCancel(context.Background())

	manager := &ModelManagerImpl{
		models:              make(map[string]Model),
		modelInfos:          make(map[string]*types.ModelInfo),
		modelStatus:         make(map[string]*types.ModelStatus),
		config:              config,
		factory:             factory,
		stats:               &types.ManagerStats{},
		logger:              logrus.New(),
		ctx:                 ctx,
		cancel:              cancel,
		healthCheckInterval: time.Duration(config.HealthCheckInterval) * time.Second,
		resourceMonitor: &ResourceMonitor{
			memoryUsage:        make(map[string]uint64),
			gpuUsage:           make(map[string]float32),
			performanceMetrics: make(map[string]*PerformanceMetrics),
			alerts:             make(map[string]*AlertThreshold),
		},
	}

	// 配置日志
	manager.configureLogger()

	// 启动健康检查
	manager.startHealthCheck()

	// 启动资源监控
	manager.startResourceMonitoring()

	manager.logger.Info("Model manager initialized successfully")

	return manager, nil
}

// LoadModel 加载模型
func (m *ModelManagerImpl) LoadModel(modelID, modelType, modelPath string, config *types.ModelConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 检查模型是否已存在
	if _, exists := m.models[modelID]; exists {
		return ErrModelAlreadyLoaded
	}

	// 验证模型类型
	if !m.factory.ValidateModelType(modelType) {
		return fmt.Errorf("unsupported model type: %s", modelType)
	}

	// 创建模型配置
	if config == nil {
		config = GetDefaultModelConfig()
	}

	// 创建模型实例
	model, err := m.factory.CreateModel(modelType, config)
	if err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}

	// 加载模型权重
	m.logger.Infof("Loading model %s from %s", modelID, modelPath)

	startTime := time.Now()
	if err := model.Load(modelPath, config); err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	loadTime := time.Since(startTime)
	m.logger.Infof("Model %s loaded successfully in %v", modelID, loadTime)

	// 存储模型实例
	m.models[modelID] = model

	// 初始化模型信息
	modelInfo := model.GetModelInfo()
	m.modelInfos[modelID] = modelInfo

	// 初始化模型状态
	m.modelStatus[modelID] = &types.ModelStatus{
		ModelID:      modelID,
		Status:       ModelStatusLoaded,
		LoadTime:     startTime,
		LoadDuration: loadTime,
		MemoryUsage:  0,
		LastUsed:     time.Now(),
	}

	// 更新统计信息
	m.stats.TotalModels++
	m.stats.LoadedModels++

	// 如果没有默认模型，设置第一个加载的模型为默认模型
	if m.defaultModelID == "" {
		m.defaultModelID = modelID
		m.logger.Infof("Set default model to %s", modelID)
	}

	// 启动模型健康检查
	m.startModelHealthCheck(modelID)

	return nil
}

// UnloadModel 卸载模型
func (m *ModelManagerImpl) UnloadModel(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 检查模型是否存在
	model, exists := m.models[modelID]
	if !exists {
		return ErrModelNotFound
	}

	m.logger.Infof("Unloading model %s", modelID)

	// 卸载模型
	if err := model.Unload(); err != nil {
		return fmt.Errorf("failed to unload model: %w", err)
	}

	// 从存储中移除
	delete(m.models, modelID)
	delete(m.modelInfos, modelID)
	delete(m.modelStatus, modelID)

	// 更新统计信息
	m.stats.LoadedModels--

	// 如果卸载的是默认模型，重新设置默认模型
	if m.defaultModelID == modelID {
		m.setNewDefaultModel()
	}

	m.logger.Infof("Model %s unloaded successfully", modelID)

	return nil
}

// GetModel 获取模型实例
func (m *ModelManagerImpl) GetModel(modelID string) (Model, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	model, exists := m.models[modelID]
	if !exists {
		return nil, ErrModelNotFound
	}

	// 更新最后使用时间
	if status, exists := m.modelStatus[modelID]; exists {
		status.LastUsed = time.Now()
	}

	return model, nil
}

// ListModels 列出所有已加载模型
func (m *ModelManagerImpl) ListModels() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	models := make([]string, 0, len(m.models))
	for modelID := range m.models {
		models = append(models, modelID)
	}

	return models
}

// SetDefaultModel 设置默认模型
func (m *ModelManagerImpl) SetDefaultModel(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 检查模型是否存在
	if _, exists := m.models[modelID]; !exists {
		return ErrModelNotFound
	}

	m.defaultModelID = modelID
	m.logger.Infof("Default model set to %s", modelID)

	return nil
}

// GetDefaultModel 获取默认模型
func (m *ModelManagerImpl) GetDefaultModel() (Model, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.defaultModelID == "" {
		return nil, fmt.Errorf("no default model set")
	}

	model, exists := m.models[m.defaultModelID]
	if !exists {
		return nil, ErrModelNotFound
	}

	return model, nil
}

// ModelExists 检查模型是否存在
func (m *ModelManagerImpl) ModelExists(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	_, exists := m.models[modelID]
	return exists
}

// GetModelStatus 获取模型状态
func (m *ModelManagerImpl) GetModelStatus(modelID string) (*types.ModelStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	status, exists := m.modelStatus[modelID]
	if !exists {
		return nil, ErrModelNotFound
	}

	// 更新内存使用信息
	if model, exists := m.models[modelID]; exists {
		if memoryUsage, err := model.GetMemoryUsage(); err == nil {
			status.MemoryUsage = memoryUsage.UsedMemory
		}
	}

	return status, nil
}

// GetModelInfo 获取模型信息
func (m *ModelManagerImpl) GetModelInfo(modelID string) (*types.ModelInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	info, exists := m.modelInfos[modelID]
	if !exists {
		return nil, ErrModelNotFound
	}

	return info, nil
}

// UpdateModelConfig 更新模型配置
func (m *ModelManagerImpl) UpdateModelConfig(modelID string, config *types.ModelConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	model, exists := m.models[modelID]
	if !exists {
		return ErrModelNotFound
	}

	return model.UpdateConfig(config)
}

// HealthCheck 健康检查
func (m *ModelManagerImpl) HealthCheck() (*types.ManagerHealth, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	health := &types.ManagerHealth{
		Status:    HealthStatusHealthy,
		Message:   "All models are healthy",
		Timestamp: time.Now().Format(time.RFC3339),
		Models:    make(map[string]string),
	}

	// 检查所有模型状态
	for modelID, model := range m.models {
		modelHealth, err := model.HealthCheck()
		if err != nil {
			health.Status = HealthStatusDegraded
			health.Message = fmt.Sprintf("Model %s is unhealthy", modelID)
			health.Models[modelID] = "unhealthy"
		} else {
			health.Models[modelID] = modelHealth.Status
		}
	}

	// 检查资源使用情况
	if m.isResourceOverloaded() {
		health.Status = HealthStatusUnhealthy
		health.Message = "Resource usage is too high"
	}

	return health, nil
}

// GetStats 获取管理器统计信息
func (m *ModelManagerImpl) GetStats() *types.ManagerStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// 更新统计信息
	m.updateStats()

	return m.stats
}

// Cleanup 清理资源
func (m *ModelManagerImpl) Cleanup() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.logger.Info("Cleaning up model manager")

	// 取消上下文
	m.cancel()

	// 卸载所有模型
	for modelID := range m.models {
		if err := m.models[modelID].Unload(); err != nil {
			m.logger.Warnf("Failed to unload model %s: %v", modelID, err)
		}
	}

	// 清空存储
	m.models = make(map[string]Model)
	m.modelInfos = make(map[string]*types.ModelInfo)
	m.modelStatus = make(map[string]*types.ModelStatus)

	m.logger.Info("Model manager cleanup completed")

	return nil
}

// ========== 私有方法 ==========

// configureLogger 配置日志记录器
func (m *ModelManagerImpl) configureLogger() {
	if m.config.LogLevel != "" {
		level, err := logrus.ParseLevel(m.config.LogLevel)
		if err == nil {
			m.logger.SetLevel(level)
		}
	}

	// 设置日志格式
	m.logger.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: time.RFC3339,
	})
}

// startHealthCheck 启动健康检查
func (m *ModelManagerImpl) startHealthCheck() {
	go func() {
		ticker := time.NewTicker(m.healthCheckInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				m.performHealthCheck()
			case <-m.ctx.Done():
				return
			}
		}
	}()
}

// performHealthCheck 执行健康检查
func (m *ModelManagerImpl) performHealthCheck() {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for modelID, model := range m.models {
		health, err := model.HealthCheck()
		if err != nil {
			m.logger.Warnf("Model %s health check failed: %v", modelID, err)
			m.modelStatus[modelID].Status = ModelStatusError
		} else {
			m.modelStatus[modelID].Status = health.Status
		}
	}
}

// startResourceMonitoring 启动资源监控
func (m *ModelManagerImpl) startResourceMonitoring() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				m.monitorResources()
			case <-m.ctx.Done():
				return
			}
		}
	}()
}

// monitorResources 监控资源使用情况
func (m *ModelManagerImpl) monitorResources() {
	m.resourceMonitor.mu.Lock()
	defer m.resourceMonitor.mu.Unlock()

	// 监控内存使用
	for modelID, model := range m.models {
		if memoryUsage, err := model.GetMemoryUsage(); err == nil {
			m.resourceMonitor.memoryUsage[modelID] = memoryUsage.UsedMemory
		}
	}

	// 检查告警阈值
	m.checkAlerts()
}

// checkAlerts 检查告警阈值
func (m *ModelManagerImpl) checkAlerts() {
	for modelID, memoryUsage := range m.resourceMonitor.memoryUsage {
		if alert, exists := m.resourceMonitor.alerts[modelID]; exists {
			if memoryUsage > alert.MemoryThreshold {
				m.logger.Warnf("Model %s memory usage exceeds threshold: %d > %d",
					modelID, memoryUsage, alert.MemoryThreshold)
			}
		}
	}
}

// startModelHealthCheck 启动模型健康检查
func (m *ModelManagerImpl) startModelHealthCheck(modelID string) {
	go func() {
		ticker := time.NewTicker(m.healthCheckInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				m.checkModelHealth(modelID)
			case <-m.ctx.Done():
				return
			}
		}
	}()
}

// checkModelHealth 检查模型健康状态
func (m *ModelManagerImpl) checkModelHealth(modelID string) {
	m.mu.RLock()
	model, exists := m.models[modelID]
	m.mu.RUnlock()

	if !exists {
		return
	}

	health, err := model.HealthCheck()
	if err != nil {
		m.logger.Warnf("Model %s health check failed: %v", modelID, err)
		m.updateModelStatus(modelID, ModelStatusError)
	} else {
		m.updateModelStatus(modelID, health.Status)
	}
}

// updateModelStatus 更新模型状态
func (m *ModelManagerImpl) updateModelStatus(modelID, status string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if modelStatus, exists := m.modelStatus[modelID]; exists {
		modelStatus.Status = status
		modelStatus.LastUsed = time.Now()
	}
}

// setNewDefaultModel 设置新的默认模型
func (m *ModelManagerImpl) setNewDefaultModel() {
	if len(m.models) == 0 {
		m.defaultModelID = ""
		return
	}

	// 选择第一个可用的模型作为默认模型
	for modelID := range m.models {
		m.defaultModelID = modelID
		m.logger.Infof("Set new default model to %s", modelID)
		break
	}
}

// updateStats 更新统计信息
func (m *ModelManagerImpl) updateStats() {
	m.stats.TotalModels = len(m.models)
	m.stats.LoadedModels = len(m.models)

	// 计算总内存使用
	totalMemory := uint64(0)
	for _, model := range m.models {
		if memoryUsage, err := model.GetMemoryUsage(); err == nil {
			totalMemory += memoryUsage.UsedMemory
		}
	}
	m.stats.MemoryUsageMB = float64(totalMemory) / 1024 / 1024
}

// isResourceOverloaded 检查资源是否过载
func (m *ModelManagerImpl) isResourceOverloaded() bool {
	totalMemory := uint64(0)
	for _, usage := range m.resourceMonitor.memoryUsage {
		totalMemory += usage
	}

	return totalMemory > m.config.MaxMemoryUsage
}

// getDefaultManagerConfig 获取默认管理器配置
func getDefaultManagerConfig() *types.ManagerConfig {
	return &types.ManagerConfig{
		MaxModels:           10,
		MaxMemoryUsage:      8 * 1024 * 1024 * 1024, // 8GB
		HealthCheckInterval: 60,
		LogLevel:            "info",
		EnableMetrics:       true,
		EnableTracing:       false,
	}
}

// ========== 资源监控器方法 ==========

// SetAlertThreshold 设置告警阈值
func (m *ModelManagerImpl) SetAlertThreshold(modelID string, threshold *AlertThreshold) {
	m.resourceMonitor.mu.Lock()
	defer m.resourceMonitor.mu.Unlock()

	m.resourceMonitor.alerts[modelID] = threshold
}

// GetPerformanceMetrics 获取性能指标
func (m *ModelManagerImpl) GetPerformanceMetrics(modelID string) *PerformanceMetrics {
	m.resourceMonitor.mu.RLock()
	defer m.resourceMonitor.mu.RUnlock()

	return m.resourceMonitor.performanceMetrics[modelID]
}

// UpdatePerformanceMetrics 更新性能指标
func (m *ModelManagerImpl) UpdatePerformanceMetrics(modelID string, responseTime float64, success bool) {
	m.resourceMonitor.mu.Lock()
	defer m.resourceMonitor.mu.Unlock()

	metrics, exists := m.resourceMonitor.performanceMetrics[modelID]
	if !exists {
		metrics = &PerformanceMetrics{}
		m.resourceMonitor.performanceMetrics[modelID] = metrics
	}

	metrics.TotalRequests++
	if !success {
		metrics.FailedRequests++
	}

	// 更新平均响应时间
	if metrics.AvgResponseTime == 0 {
		metrics.AvgResponseTime = responseTime
	} else {
		metrics.AvgResponseTime = (metrics.AvgResponseTime*float64(metrics.TotalRequests-1) + responseTime) / float64(metrics.TotalRequests)
	}

	metrics.LastResponseTime = time.Now()

	// 计算吞吐量（最近一分钟）
	if metrics.LastResponseTime.Sub(time.Now().Add(-time.Minute)) < time.Minute {
		metrics.Throughput = float64(metrics.TotalRequests) / 60.0
	}
}

// GetResourceUsage 获取资源使用情况
func (m *ModelManagerImpl) GetResourceUsage() map[string]interface{} {
	m.resourceMonitor.mu.RLock()
	defer m.resourceMonitor.mu.RUnlock()

	usage := make(map[string]interface{})
	usage["memory"] = m.resourceMonitor.memoryUsage
	usage["gpu"] = m.resourceMonitor.gpuUsage
	usage["performance"] = m.resourceMonitor.performanceMetrics

	return usage
}

// ========== 工具函数 ==========

// CreateModelID 创建模型ID
func CreateModelID(modelType, modelPath string) string {
	return fmt.Sprintf("%s_%s", modelType, modelPath)
}

// ValidateModelConfig 验证模型配置
func ValidateModelConfig(config *types.ModelConfig) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	if config.Name == "" {
		return fmt.Errorf("model name is required")
	}

	if config.VocabSize <= 0 {
		return fmt.Errorf("vocab size must be positive")
	}

	if config.HiddenSize <= 0 {
		return fmt.Errorf("hidden size must be positive")
	}

	if config.NumLayers <= 0 {
		return fmt.Errorf("number of layers must be positive")
	}

	if config.NumHeads <= 0 {
		return fmt.Errorf("number of heads must be positive")
	}

	return nil
}

// GetSupportedModelTypes 获取支持的模型类型
func GetSupportedModelTypes() []string {
	return []string{"minimind", "llama", "bert", "gpt"}
}

// IsModelTypeSupported 检查模型类型是否支持
func IsModelTypeSupported(modelType string) bool {
	supportedTypes := GetSupportedModelTypes()
	for _, t := range supportedTypes {
		if t == modelType {
			return true
		}
	}
	return false
}
