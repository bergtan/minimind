package autograd

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// DataType 数据类型
type DataType int

const (
	Float32 DataType = iota
	Float16
	BFloat16
)

// AMPConfig 混合精度训练配置
type AMPConfig struct {
	Enabled          bool    // 是否启用混合精度
	InitScale        float32 // 初始损失缩放因子
	GrowthFactor     float32 // 缩放因子增长系数
	BackoffFactor    float32 // 缩放因子减小系数
	GrowthInterval   int     // 增长间隔
	MaxScale         float32 // 最大缩放因子
	MinScale         float32 // 最小缩放因子
	CastToFP32Set    []Op    // 需要保持FP32的操作
	DynamicLossScale bool    // 是否使用动态损失缩放
}

// DefaultAMPConfig 返回默认AMP配置
func DefaultAMPConfig() *AMPConfig {
	return &AMPConfig{
		Enabled:          true,
		InitScale:        65536.0,
		GrowthFactor:     2.0,
		BackoffFactor:    0.5,
		GrowthInterval:   2000,
		MaxScale:         float32(math.Inf(1)),
		MinScale:         1.0,
		CastToFP32Set:    []Op{OpAdd, OpSum, OpMean},
		DynamicLossScale: true,
	}
}

// LossScaler 损失缩放器
type LossScaler struct {
	config        *AMPConfig
	currentScale  float32
	growthTracker int
	mu            sync.RWMutex
}

// NewLossScaler 创建新的损失缩放器
func NewLossScaler(config *AMPConfig) *LossScaler {
	if config == nil {
		config = DefaultAMPConfig()
	}
	return &LossScaler{
		config:        config,
		currentScale:  config.InitScale,
		growthTracker: 0,
	}
}

// Scale 缩放损失
func (ls *LossScaler) Scale(loss float64) float64 {
	ls.mu.RLock()
	defer ls.mu.RUnlock()

	if !ls.config.Enabled {
		return loss
	}

	return loss * float64(ls.currentScale)
}

// Unscale 反缩放梯度
func (ls *LossScaler) Unscale(gradients map[string]*mat.VecDense) {
	ls.mu.RLock()
	scale := ls.currentScale
	ls.mu.RUnlock()

	if !ls.config.Enabled || scale == 1.0 {
		return
	}

	invScale := 1.0 / float64(scale)
	for _, grad := range gradients {
		grad.ScaleVec(invScale, grad)
	}
}

// Update 更新缩放因子（检查梯度溢出）
func (ls *LossScaler) Update(foundInf bool) {
	ls.mu.Lock()
	defer ls.mu.Unlock()

	if !ls.config.Enabled || !ls.config.DynamicLossScale {
		return
	}

	if foundInf {
		// 发现溢出，减小缩放因子
		ls.currentScale = max(ls.currentScale*ls.config.BackoffFactor, ls.config.MinScale)
		ls.growthTracker = 0
	} else {
		// 无溢出，可能增加缩放因子
		ls.growthTracker++
		if ls.growthTracker >= ls.config.GrowthInterval {
			ls.currentScale = min(ls.currentScale*ls.config.GrowthFactor, ls.config.MaxScale)
			ls.growthTracker = 0
		}
	}
}

// CurrentScale 获取当前缩放因子
func (ls *LossScaler) CurrentScale() float32 {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.currentScale
}

// CheckInfInGradients 检查梯度中是否有Inf或NaN
func CheckInfInGradients(gradients map[string]*mat.VecDense) bool {
	for _, grad := range gradients {
		data := grad.RawVector().Data
		for _, v := range data {
			if math.IsInf(v, 0) || math.IsNaN(v) {
				return true
			}
		}
	}
	return false
}

// FP16Tensor FP16精度张量
type FP16Tensor struct {
	data  []uint16 // FP16数据（半精度浮点）
	shape []int
	dtype DataType
	mu    sync.RWMutex
}

// NewFP16Tensor 创建FP16张量
func NewFP16Tensor(shape []int) *FP16Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &FP16Tensor{
		data:  make([]uint16, size),
		shape: shape,
		dtype: Float16,
	}
}

// ToFP16 将Float32转换为Float16
func ToFP16(data []float32) []uint16 {
	fp16Data := make([]uint16, len(data))
	for i, v := range data {
		fp16Data[i] = float32ToFP16(v)
	}
	return fp16Data
}

// ToFP32 将Float16转换为Float32
func ToFP32(data []uint16) []float32 {
	fp32Data := make([]float32, len(data))
	for i, v := range data {
		fp32Data[i] = fp16ToFloat32(v)
	}
	return fp32Data
}

// float32ToFP16 将float32转换为FP16表示
func float32ToFP16(f float32) uint16 {
	// 简化实现，实际应使用IEEE 754标准转换
	if f == 0 {
		return 0
	}

	// 提取符号位
	sign := uint16(0)
	if f < 0 {
		sign = 1
		f = -f
	}

	// 提取指数
	exp := int32(0)
	mantissa := f
	if mantissa >= 1.0 {
		for mantissa >= 2.0 {
			mantissa /= 2.0
			exp++
		}
	} else {
		for mantissa < 1.0 {
			mantissa *= 2.0
			exp--
		}
	}

	// 偏置指数
	exp += 15
	if exp > 31 {
		return sign<<15 | 0x7C00 // Inf
	}
	if exp < 0 {
		return sign << 15 // 下溢为0
	}

	// 尾数（10位）
	mantissa = (mantissa - 1.0) * 1024
	mant := uint16(mantissa)

	return sign<<15 | uint16(exp)<<10 | (mant & 0x3FF)
}

// fp16ToFloat32 将FP16转换为float32
func fp16ToFloat32(bits uint16) float32 {
	// 提取分量
	sign := float32(1.0)
	if bits&0x8000 != 0 {
		sign = -1.0
	}

	exp := int32((bits >> 10) & 0x1F)
	mant := float32(bits & 0x3FF)

	// 特殊情况
	if exp == 0 {
		if mant == 0 {
			return 0 // 零
		}
		// 次正规数
		return sign * mant * float32(math.Pow(2, -24))
	}
	if exp == 31 {
		if mant == 0 {
			return sign * float32(math.Inf(1)) // Inf
		}
		return float32(math.NaN()) // NaN
	}

	// 正规数
	exp -= 15
	return sign * (1.0 + mant/1024.0) * float32(math.Pow(2, float64(exp)))
}

// AMPGradientScaler 梯度缩放器
type AMPGradientScaler struct {
	lossScaler   *LossScaler
	masterParams map[string]*mat.VecDense // FP32主参数
	modelParams  map[string]*mat.VecDense // FP16模型参数
}

// NewAMPGradientScaler 创建AMP梯度缩放器
func NewAMPGradientScaler(lossScaler *LossScaler) *AMPGradientScaler {
	return &AMPGradientScaler{
		lossScaler:   lossScaler,
		masterParams: make(map[string]*mat.VecDense),
		modelParams:  make(map[string]*mat.VecDense),
	}
}

// RegisterParams 注册参数
func (ags *AMPGradientScaler) RegisterParams(name string, master, model *mat.VecDense) {
	ags.masterParams[name] = master
	ags.modelParams[name] = model
}

// Step 执行梯度缩放和参数更新
func (ags *AMPGradientScaler) Step(gradients map[string]*mat.VecDense) {
	// 1. 检查梯度溢出
	foundInf := CheckInfInGradients(gradients)

	if foundInf {
		// 发现溢出，跳过更新
		ags.lossScaler.Update(true)
		return
	}

	// 2. 反缩放梯度
	ags.lossScaler.Unscale(gradients)

	// 3. 梯度裁剪
	clipGradNorm(gradients, 1.0)

	// 4. 更新主参数（FP32）
	for name, grad := range gradients {
		if master, ok := ags.masterParams[name]; ok {
			master.AddVec(master, grad)
		}
	}

	// 5. 更新缩放因子
	ags.lossScaler.Update(false)

	// 6. 拷贝到FP16参数
	ags.copyMasterToModel()
}

// copyMasterToModel 将FP32主参数拷贝到FP16模型参数
func (ags *AMPGradientScaler) copyMasterToModel() {
	for name, master := range ags.masterParams {
		if model, ok := ags.modelParams[name]; ok {
			masterData := master.RawVector().Data
			modelData := model.RawVector().Data

			for i, v := range masterData {
				if i < len(modelData) {
					// FP32转FP16
					modelData[i] = float64(float32(v))
				}
			}
		}
	}
}

// clipGradNorm 梯度裁剪
func clipGradNorm(gradients map[string]*mat.VecDense, maxNorm float64) {
	totalNorm := 0.0
	for _, grad := range gradients {
		data := grad.RawVector().Data
		for _, v := range data {
			totalNorm += v * v
		}
	}
	totalNorm = math.Sqrt(totalNorm)

	clipCoef := maxNorm / (totalNorm + 1e-6)
	if clipCoef < 1.0 {
		for _, grad := range gradients {
			grad.ScaleVec(clipCoef, grad)
		}
	}
}

// MasterWeightOptimizer 主权重优化器
type MasterWeightOptimizer struct {
	optimizer    Optimizer
	scaler       *AMPGradientScaler
	masterParams map[string]*Tensor
}

// NewMasterWeightOptimizer 创建主权重优化器
func NewMasterWeightOptimizer(opt Optimizer, scaler *AMPGradientScaler) *MasterWeightOptimizer {
	return &MasterWeightOptimizer{
		optimizer:    opt,
		scaler:       scaler,
		masterParams: make(map[string]*Tensor),
	}
}

// Step 执行优化步骤
func (mwo *MasterWeightOptimizer) Step(gradients map[string]*mat.VecDense) {
	mwo.scaler.Step(gradients)
}

// ZeroGrad 清零梯度
func (mwo *MasterWeightOptimizer) ZeroGrad() {
	mwo.optimizer.ZeroGrad()
}

// GetLR 获取学习率
func (mwo *MasterWeightOptimizer) GetLR() float64 {
	return mwo.optimizer.GetLR()
}

// SetLR 设置学习率
func (mwo *MasterWeightOptimizer) SetLR(lr float64) {
	mwo.optimizer.SetLR(lr)
}

// ============ 辅助函数 ============

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// Optimizer 优化器接口（与trainer包兼容）
type Optimizer interface {
	Step(gradients map[string]*mat.VecDense)
	ZeroGrad()
	GetLR() float64
	SetLR(lr float64)
}

// Autocast 自动类型转换上下文管理器
type Autocast struct {
	enabled bool
	dtype   DataType
	cache   map[string]interface{}
}

// NewAutocast 创建自动类型转换上下文
func NewAutocast(enabled bool) *Autocast {
	return &Autocast{
		enabled: enabled,
		dtype:   Float16,
		cache:   make(map[string]interface{}),
	}
}

// Enter 进入上下文
func (a *Autocast) Enter() {
	// 设置全局AMP状态
}

// Exit 退出上下文
func (a *Autocast) Exit() {
	// 恢复AMP状态
}

// ShouldCastToFP16 判断操作是否应使用FP16
func (a *Autocast) ShouldCastToFP16(op Op) bool {
	if !a.enabled {
		return false
	}
	// 某些操作应保持FP32精度
	for _, keepFP32Op := range DefaultAMPConfig().CastToFP32Set {
		if op == keepFP32Op {
			return false
		}
	}
	return true
}

// CastTensor 转换张量精度
func CastTensor(tensor *Tensor, dtype DataType) *Tensor {
	if dtype == Float32 {
		return tensor
	}

	// FP16转换（简化实现）
	r, c := tensor.Dims()
	data := tensor.Data().RawMatrix().Data
	fp16Data := make([]float64, len(data))

	for i, v := range data {
		// 模拟FP16精度损失
		fp16Val := float64(float32(v))
		fp16Data[i] = fp16Val
	}

	return NewTensor(fp16Data, r, c)
}
