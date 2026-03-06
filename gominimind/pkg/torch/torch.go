package torch

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Tensor 张量类型（简化的torch.Tensor兼容层）
type Tensor struct {
	Data   *mat.Dense
	Shape  []int
	Dtype  string
	Device string
}

// NewTensor 从float64切片创建张量
func NewTensor(data []float64, shape []int) *Tensor {
	if len(shape) != 2 {
		// 简化为2D
		rows := shape[0]
		cols := 1
		for i := 1; i < len(shape); i++ {
			cols *= shape[i]
		}
		shape = []int{rows, cols}
	}
	return &Tensor{
		Data:   mat.NewDense(shape[0], shape[1], data),
		Shape:  shape,
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Zeros 创建零张量
func Zeros(shape ...int) *Tensor {
	rows, cols := shape[0], 1
	if len(shape) > 1 {
		cols = shape[1]
	}
	return &Tensor{
		Data:   mat.NewDense(rows, cols, nil),
		Shape:  []int{rows, cols},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Ones 创建全1张量
func Ones(shape ...int) *Tensor {
	rows, cols := shape[0], 1
	if len(shape) > 1 {
		cols = shape[1]
	}
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return &Tensor{
		Data:   mat.NewDense(rows, cols, data),
		Shape:  []int{rows, cols},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Randn 创建正态分布随机张量
func Randn(shape ...int) *Tensor {
	rows, cols := shape[0], 1
	if len(shape) > 1 {
		cols = shape[1]
	}
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return &Tensor{
		Data:   mat.NewDense(rows, cols, data),
		Shape:  []int{rows, cols},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Rand 创建均匀分布随机张量
func Rand(shape ...int) *Tensor {
	rows, cols := shape[0], 1
	if len(shape) > 1 {
		cols = shape[1]
	}
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()
	}
	return &Tensor{
		Data:   mat.NewDense(rows, cols, data),
		Shape:  []int{rows, cols},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Arange 创建等差序列张量
func Arange(start, end, step float64) *Tensor {
	n := int(math.Ceil((end - start) / step))
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = start + float64(i)*step
	}
	return &Tensor{
		Data:   mat.NewDense(1, n, data),
		Shape:  []int{1, n},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Linspace 创建等间隔序列张量
func Linspace(start, end float64, steps int) *Tensor {
	data := make([]float64, steps)
	if steps == 1 {
		data[0] = start
	} else {
		step := (end - start) / float64(steps-1)
		for i := 0; i < steps; i++ {
			data[i] = start + float64(i)*step
		}
	}
	return &Tensor{
		Data:   mat.NewDense(1, steps, data),
		Shape:  []int{1, steps},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// ============ 张量运算 ============

// MatMul 矩阵乘法
func MatMul(a, b *Tensor) *Tensor {
	r1, _ := a.Data.Dims()
	_, c2 := b.Data.Dims()
	result := mat.NewDense(r1, c2, nil)
	result.Mul(a.Data, b.Data)
	return &Tensor{
		Data:   result,
		Shape:  []int{r1, c2},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Add 张量加法
func Add(a, b *Tensor) *Tensor {
	r, c := a.Data.Dims()
	result := mat.NewDense(r, c, nil)
	result.Add(a.Data, b.Data)
	return &Tensor{
		Data:   result,
		Shape:  []int{r, c},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Sub 张量减法
func Sub(a, b *Tensor) *Tensor {
	r, c := a.Data.Dims()
	result := mat.NewDense(r, c, nil)
	result.Sub(a.Data, b.Data)
	return &Tensor{
		Data:   result,
		Shape:  []int{r, c},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// MulElem 逐元素乘法
func MulElem(a, b *Tensor) *Tensor {
	r, c := a.Data.Dims()
	result := mat.NewDense(r, c, nil)
	result.MulElem(a.Data, b.Data)
	return &Tensor{
		Data:   result,
		Shape:  []int{r, c},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// ============ 损失函数 ============

// CrossEntropyLoss 交叉熵损失
func CrossEntropyLoss(logits, targets *Tensor) float64 {
	r, c := logits.Data.Dims()
	totalLoss := 0.0

	for i := 0; i < r; i++ {
		// 找到目标类别
		targetIdx := int(targets.Data.At(i, 0))
		if targetIdx < 0 || targetIdx >= c {
			continue
		}

		// 计算log-softmax
		maxVal := -math.MaxFloat64
		for j := 0; j < c; j++ {
			if logits.Data.At(i, j) > maxVal {
				maxVal = logits.Data.At(i, j)
			}
		}

		sumExp := 0.0
		for j := 0; j < c; j++ {
			sumExp += math.Exp(logits.Data.At(i, j) - maxVal)
		}

		logSoftmax := logits.Data.At(i, targetIdx) - maxVal - math.Log(sumExp)
		totalLoss -= logSoftmax
	}

	return totalLoss / float64(r)
}

// MSELoss 均方误差损失
func MSELoss(predictions, targets *Tensor) float64 {
	r, c := predictions.Data.Dims()
	totalLoss := 0.0
	n := float64(r * c)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			diff := predictions.Data.At(i, j) - targets.Data.At(i, j)
			totalLoss += diff * diff
		}
	}

	return totalLoss / n
}

// ============ 激活函数 ============

// Softmax 计算softmax
func Softmax(t *Tensor, dim int) *Tensor {
	r, c := t.Data.Dims()
	result := mat.NewDense(r, c, nil)

	if dim == 1 || dim == -1 {
		for i := 0; i < r; i++ {
			maxVal := -math.MaxFloat64
			for j := 0; j < c; j++ {
				if t.Data.At(i, j) > maxVal {
					maxVal = t.Data.At(i, j)
				}
			}
			sumExp := 0.0
			for j := 0; j < c; j++ {
				sumExp += math.Exp(t.Data.At(i, j) - maxVal)
			}
			for j := 0; j < c; j++ {
				result.Set(i, j, math.Exp(t.Data.At(i, j)-maxVal)/sumExp)
			}
		}
	}

	return &Tensor{
		Data:   result,
		Shape:  []int{r, c},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// Sigmoid Sigmoid激活
func Sigmoid(t *Tensor) *Tensor {
	r, c := t.Data.Dims()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, 1.0/(1.0+math.Exp(-t.Data.At(i, j))))
		}
	}

	return &Tensor{
		Data:   result,
		Shape:  []int{r, c},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// ReLU ReLU激活
func ReLU(t *Tensor) *Tensor {
	r, c := t.Data.Dims()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := t.Data.At(i, j)
			if val > 0 {
				result.Set(i, j, val)
			}
		}
	}

	return &Tensor{
		Data:   result,
		Shape:  []int{r, c},
		Dtype:  "float32",
		Device: "cpu",
	}
}

// ============ 工具函数 ============

// ManualSeed 设置随机种子
func ManualSeed(seed int64) {
	rand.Seed(seed)
}

// NoGrad 无梯度上下文（在Go中简化为空操作）
func NoGrad(fn func()) {
	fn()
}

// String 张量字符串表示
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%s)", t.Shape, t.Dtype, t.Device)
}

// To 将张量移动到指定设备
func (t *Tensor) To(device string) *Tensor {
	// 在纯Go实现中，只更新device标记
	t.Device = device
	return t
}

// Clone 克隆张量
func (t *Tensor) Clone() *Tensor {
	r, c := t.Data.Dims()
	data := make([]float64, r*c)
	copy(data, t.Data.RawMatrix().Data)
	return &Tensor{
		Data:   mat.NewDense(r, c, data),
		Shape:  []int{r, c},
		Dtype:  t.Dtype,
		Device: t.Device,
	}
}

// Numel 返回元素总数
func (t *Tensor) Numel() int {
	r, c := t.Data.Dims()
	return r * c
}
