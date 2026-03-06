package trainer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Optimizer 优化器接口
type Optimizer interface {
	Step(gradients map[string]*mat.VecDense)
	ZeroGrad()
	GetLR() float64
	SetLR(lr float64)
	GetState() map[string]interface{}
	SetState(state map[string]interface{})
}

// AdamWConfig AdamW优化器配置
type AdamWConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	WeightDecay  float64
}

// DefaultAdamWConfig 返回默认AdamW配置
func DefaultAdamWConfig() *AdamWConfig {
	return &AdamWConfig{
		LearningRate: 5e-4,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.01,
	}
}

// AdamW AdamW优化器实现
type AdamW struct {
	config *AdamWConfig
	params map[string]*mat.VecDense
	m      map[string]*mat.VecDense // 一阶矩估计
	v      map[string]*mat.VecDense // 二阶矩估计
	t      int                      // 时间步
}

// NewAdamW 创建新的AdamW优化器
func NewAdamW(params map[string]*mat.VecDense, config *AdamWConfig) *AdamW {
	if config == nil {
		config = DefaultAdamWConfig()
	}

	m := make(map[string]*mat.VecDense)
	v := make(map[string]*mat.VecDense)

	for name, param := range params {
		size := param.Len()
		m[name] = mat.NewVecDense(size, make([]float64, size))
		v[name] = mat.NewVecDense(size, make([]float64, size))
	}

	return &AdamW{
		config: config,
		params: params,
		m:      m,
		v:      v,
		t:      0,
	}
}

// Step 执行一步优化
func (opt *AdamW) Step(gradients map[string]*mat.VecDense) {
	opt.t++

	lr := opt.config.LearningRate
	beta1 := opt.config.Beta1
	beta2 := opt.config.Beta2
	epsilon := opt.config.Epsilon
	weightDecay := opt.config.WeightDecay

	for name, param := range opt.params {
		grad, exists := gradients[name]
		if !exists {
			continue
		}

		m := opt.m[name]
		v := opt.v[name]

		paramData := param.RawVector().Data
		gradData := grad.RawVector().Data
		mData := m.RawVector().Data
		vData := v.RawVector().Data

		// AdamW: 权重衰减在自适应梯度之前应用
		for i := 0; i < len(paramData); i++ {
			// 权重衰减（与L2正则化不同，直接衰减参数）
			paramData[i] *= (1 - lr*weightDecay)

			// 更新一阶矩估计
			mData[i] = beta1*mData[i] + (1-beta1)*gradData[i]

			// 更新二阶矩估计
			vData[i] = beta2*vData[i] + (1-beta2)*gradData[i]*gradData[i]

			// 偏差修正
			mHat := mData[i] / (1 - math.Pow(beta1, float64(opt.t)))
			vHat := vData[i] / (1 - math.Pow(beta2, float64(opt.t)))

			// 更新参数
			paramData[i] -= lr * mHat / (math.Sqrt(vHat) + epsilon)
		}
	}
}

// ZeroGrad 清零梯度
func (opt *AdamW) ZeroGrad() {
	// 在调用Step之前，调用者应该重新计算梯度
	// 这里不需要显式清零，因为梯度会在下一次反向传播时被覆盖
}

// GetLR 获取当前学习率
func (opt *AdamW) GetLR() float64 {
	return opt.config.LearningRate
}

// SetLR 设置学习率
func (opt *AdamW) SetLR(lr float64) {
	opt.config.LearningRate = lr
}

// GetState 获取优化器状态
func (opt *AdamW) GetState() map[string]interface{} {
	state := make(map[string]interface{})
	state["t"] = opt.t
	state["learning_rate"] = opt.config.LearningRate

	// 保存一阶矩估计
	mData := make(map[string][]float64)
	for name, m := range opt.m {
		mData[name] = append([]float64(nil), m.RawVector().Data...)
	}
	state["m"] = mData

	// 保存二阶矩估计
	vData := make(map[string][]float64)
	for name, v := range opt.v {
		vData[name] = append([]float64(nil), v.RawVector().Data...)
	}
	state["v"] = vData

	return state
}

// SetState 设置优化器状态
func (opt *AdamW) SetState(state map[string]interface{}) {
	if t, ok := state["t"].(int); ok {
		opt.t = t
	}
	if lr, ok := state["learning_rate"].(float64); ok {
		opt.config.LearningRate = lr
	}

	if mData, ok := state["m"].(map[string][]float64); ok {
		for name, data := range mData {
			if m, exists := opt.m[name]; exists {
				copy(m.RawVector().Data, data)
			}
		}
	}

	if vData, ok := state["v"].(map[string][]float64); ok {
		for name, data := range vData {
			if v, exists := opt.v[name]; exists {
				copy(v.RawVector().Data, data)
			}
		}
	}
}

// SGDConfig SGD优化器配置
type SGDConfig struct {
	LearningRate float64
	Momentum     float64
	WeightDecay  float64
	Nesterov     bool
}

// DefaultSGDConfig 返回默认SGD配置
func DefaultSGDConfig() *SGDConfig {
	return &SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.9,
		WeightDecay:  0.0,
		Nesterov:     false,
	}
}

// SGD SGD优化器实现
type SGD struct {
	config     *SGDConfig
	params     map[string]*mat.VecDense
	velocities map[string]*mat.VecDense
}

// NewSGD 创建新的SGD优化器
func NewSGD(params map[string]*mat.VecDense, config *SGDConfig) *SGD {
	if config == nil {
		config = DefaultSGDConfig()
	}

	velocities := make(map[string]*mat.VecDense)
	for name, param := range params {
		size := param.Len()
		velocities[name] = mat.NewVecDense(size, make([]float64, size))
	}

	return &SGD{
		config:     config,
		params:     params,
		velocities: velocities,
	}
}

// Step 执行一步优化
func (opt *SGD) Step(gradients map[string]*mat.VecDense) {
	lr := opt.config.LearningRate
	momentum := opt.config.Momentum
	weightDecay := opt.config.WeightDecay
	nesterov := opt.config.Nesterov

	for name, param := range opt.params {
		grad, exists := gradients[name]
		if !exists {
			continue
		}

		velocity := opt.velocities[name]
		paramData := param.RawVector().Data
		gradData := grad.RawVector().Data
		velocityData := velocity.RawVector().Data

		for i := 0; i < len(paramData); i++ {
			// 权重衰减
			g := gradData[i] + weightDecay*paramData[i]

			// 更新速度
			velocityData[i] = momentum*velocityData[i] + g

			// 更新参数
			if nesterov {
				paramData[i] -= lr * (momentum*velocityData[i] + g)
			} else {
				paramData[i] -= lr * velocityData[i]
			}
		}
	}
}

// ZeroGrad 清零梯度
func (opt *SGD) ZeroGrad() {}

// GetLR 获取当前学习率
func (opt *SGD) GetLR() float64 {
	return opt.config.LearningRate
}

// SetLR 设置学习率
func (opt *SGD) SetLR(lr float64) {
	opt.config.LearningRate = lr
}

// GetState 获取优化器状态
func (opt *SGD) GetState() map[string]interface{} {
	state := make(map[string]interface{})
	state["learning_rate"] = opt.config.LearningRate

	velocityData := make(map[string][]float64)
	for name, v := range opt.velocities {
		velocityData[name] = append([]float64(nil), v.RawVector().Data...)
	}
	state["velocities"] = velocityData

	return state
}

// SetState 设置优化器状态
func (opt *SGD) SetState(state map[string]interface{}) {
	if lr, ok := state["learning_rate"].(float64); ok {
		opt.config.LearningRate = lr
	}

	if velocityData, ok := state["velocities"].(map[string][]float64); ok {
		for name, data := range velocityData {
			if v, exists := opt.velocities[name]; exists {
				copy(v.RawVector().Data, data)
			}
		}
	}
}
