package autograd

import (
	"fmt"
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// Op 操作类型
type Op int

const (
	OpNoOp Op = iota
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpMatMul
	OpPow
	OpExp
	OpLog
	OpSqrt
	OpRelu
	OpSigmoid
	OpSoftmax
	OpTanh
	OpTranspose
	OpSum
	OpMean
	OpMax
	OpMin
	OpReshape
	OpSlice
)

// Tensor 支持自动微分的张量
type Tensor struct {
	data         *mat.Dense
	grad         *mat.Dense
	op           Op
	prev         []*Tensor
	requiresGrad bool
	gradFn       func()
	name         string
	id           int64

	mu sync.RWMutex
}

var (
	tensorIDCounter int64
	tensorIDMu      sync.Mutex
)

func newTensorID() int64 {
	tensorIDMu.Lock()
	defer tensorIDMu.Unlock()
	tensorIDCounter++
	return tensorIDCounter
}

// NewTensor 创建新张量
func NewTensor(data []float64, rows, cols int) *Tensor {
	if len(data) == 0 {
		data = make([]float64, rows*cols)
	}

	dense := mat.NewDense(rows, cols, data)
	return &Tensor{
		data:         dense,
		grad:         mat.NewDense(rows, cols, nil),
		op:           OpNoOp,
		prev:         nil,
		requiresGrad: false,
		id:           newTensorID(),
	}
}

// NewTensorFromDense 从Dense创建张量
func NewTensorFromDense(dense *mat.Dense) *Tensor {
	r, c := dense.Dims()
	return &Tensor{
		data:         dense,
		grad:         mat.NewDense(r, c, nil),
		op:           OpNoOp,
		prev:         nil,
		requiresGrad: false,
		id:           newTensorID(),
	}
}

// Zeros 创建零张量
func Zeros(rows, cols int) *Tensor {
	return NewTensor(nil, rows, cols)
}

// Ones 创建全1张量
func Ones(rows, cols int) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return NewTensor(data, rows, cols)
}

// Randn 创建正态分布随机张量
func Randn(rows, cols int) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		// Box-Muller变换
		u1 := randFloat()
		u2 := randFloat()
		data[i] = math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	}
	return NewTensor(data, rows, cols)
}

// Rand 创建均匀分布随机张量
func Rand(rows, cols int) *Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = randFloat()
	}
	return NewTensor(data, rows, cols)
}

func randFloat() float64 {
	// 简化实现，实际应使用更好的随机数生成器
	return float64(newTensorID()%1000) / 1000.0
}

// SetRequiresGrad 设置是否需要梯度
func (t *Tensor) SetRequiresGrad(requires bool) *Tensor {
	t.requiresGrad = requires
	return t
}

// Dims 返回维度
func (t *Tensor) Dims() (int, int) {
	return t.data.Dims()
}

// Shape 返回形状
func (t *Tensor) Shape() []int {
	r, c := t.data.Dims()
	return []int{r, c}
}

// Numel 返回元素数量
func (t *Tensor) Numel() int {
	r, c := t.data.Dims()
	return r * c
}

// Data 获取数据
func (t *Tensor) Data() *mat.Dense {
	return t.data
}

// Grad 获取梯度
func (t *Tensor) Grad() *mat.Dense {
	return t.grad
}

// ZeroGrad 清零梯度
func (t *Tensor) ZeroGrad() {
	r, c := t.grad.Dims()
	t.grad = mat.NewDense(r, c, nil)
}

// String 返回字符串表示
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(%v, requires_grad=%v)", t.Shape(), t.requiresGrad)
}

// ==================== 基础运算 ====================

// Add 加法
func (t *Tensor) Add(other *Tensor) *Tensor {
	r1, c1 := t.data.Dims()
	r2, c2 := other.data.Dims()

	// 广播处理
	if r1 != r2 || c1 != c2 {
		// 简化处理，实际应实现完整广播逻辑
		return nil
	}

	result := mat.NewDense(r1, c1, nil)
	result.Add(t.data, other.data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r1, c1, nil),
		op:           OpAdd,
		prev:         []*Tensor{t, other},
		requiresGrad: t.requiresGrad || other.requiresGrad,
		id:           newTensorID(),
	}

	// 定义梯度函数
	out.gradFn = func() {
		if t.requiresGrad {
			t.grad.Add(t.grad, out.grad)
		}
		if other.requiresGrad {
			other.grad.Add(other.grad, out.grad)
		}
	}

	return out
}

// Sub 减法
func (t *Tensor) Sub(other *Tensor) *Tensor {
	r1, c1 := t.data.Dims()
	r2, c2 := other.data.Dims()

	if r1 != r2 || c1 != c2 {
		return nil
	}

	result := mat.NewDense(r1, c1, nil)
	result.Sub(t.data, other.data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r1, c1, nil),
		op:           OpSub,
		prev:         []*Tensor{t, other},
		requiresGrad: t.requiresGrad || other.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			t.grad.Add(t.grad, out.grad)
		}
		if other.requiresGrad {
			other.grad.Sub(other.grad, out.grad)
		}
	}

	return out
}

// Mul 元素乘法
func (t *Tensor) Mul(other *Tensor) *Tensor {
	r1, c1 := t.data.Dims()
	r2, c2 := other.data.Dims()

	if r1 != r2 || c1 != c2 {
		return nil
	}

	result := mat.NewDense(r1, c1, nil)
	result.MulElem(t.data, other.data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r1, c1, nil),
		op:           OpMul,
		prev:         []*Tensor{t, other},
		requiresGrad: t.requiresGrad || other.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			gradTemp := mat.NewDense(r1, c1, nil)
			gradTemp.MulElem(out.grad, other.data)
			t.grad.Add(t.grad, gradTemp)
		}
		if other.requiresGrad {
			gradTemp := mat.NewDense(r1, c1, nil)
			gradTemp.MulElem(out.grad, t.data)
			other.grad.Add(other.grad, gradTemp)
		}
	}

	return out
}

// MatMul 矩阵乘法
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	r1, c1 := t.data.Dims()
	r2, c2 := other.data.Dims()

	if c1 != r2 {
		return nil
	}

	result := mat.NewDense(r1, c2, nil)
	result.Mul(t.data, other.data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r1, c2, nil),
		op:           OpMatMul,
		prev:         []*Tensor{t, other},
		requiresGrad: t.requiresGrad || other.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			// grad_t = grad_out @ other^T
			otherT := other.data.T()
			gradTemp := mat.NewDense(r1, c1, nil)
			gradTemp.Mul(out.grad, otherT)
			t.grad.Add(t.grad, gradTemp)
		}
		if other.requiresGrad {
			// grad_other = t^T @ grad_out
			tT := t.data.T()
			gradTemp := mat.NewDense(c1, c2, nil)
			gradTemp.Mul(tT, out.grad)
			other.grad.Add(other.grad, gradTemp)
		}
	}

	return out
}

// Div 除法
func (t *Tensor) Div(other *Tensor) *Tensor {
	r1, c1 := t.data.Dims()

	result := mat.NewDense(r1, c1, nil)
	result.DivElem(t.data, other.data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r1, c1, nil),
		op:           OpDiv,
		prev:         []*Tensor{t, other},
		requiresGrad: t.requiresGrad || other.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			// grad_t = grad_out / other
			gradTemp := mat.NewDense(r1, c1, nil)
			gradTemp.DivElem(out.grad, other.data)
			t.grad.Add(t.grad, gradTemp)
		}
		if other.requiresGrad {
			// grad_other = -grad_out * t / other^2
			gradTemp := mat.NewDense(r1, c1, nil)
			gradTemp.MulElem(out.grad, t.data)
			other2 := mat.NewDense(r1, c1, nil)
			other2.MulElem(other.data, other.data)
			gradTemp.DivElem(gradTemp, other2)
			gradTemp.Scale(-1, gradTemp)
			other.grad.Add(other.grad, gradTemp)
		}
	}

	return out
}

// Pow 幂运算
func (t *Tensor) Pow(exponent float64) *Tensor {
	r, c := t.data.Dims()

	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = math.Pow(t.data.At(i, j), exponent)
		}
	}
	result := mat.NewDense(r, c, data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r, c, nil),
		op:           OpPow,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			gradTemp := mat.NewDense(r, c, nil)
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					val := t.data.At(i, j)
					grad := out.grad.At(i, j)
					gradTemp.Set(i, j, grad*exponent*math.Pow(val, exponent-1))
				}
			}
			t.grad.Add(t.grad, gradTemp)
		}
	}

	return out
}

// ==================== 激活函数 ====================

// Relu ReLU激活函数
func (t *Tensor) Relu() *Tensor {
	r, c := t.data.Dims()

	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := t.data.At(i, j)
			if val > 0 {
				result.Set(i, j, val)
			}
		}
	}

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r, c, nil),
		op:           OpRelu,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					if t.data.At(i, j) > 0 {
						t.grad.Set(i, j, t.grad.At(i, j)+out.grad.At(i, j))
					}
				}
			}
		}
	}

	return out
}

// Sigmoid Sigmoid激活函数
func (t *Tensor) Sigmoid() *Tensor {
	r, c := t.data.Dims()

	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := t.data.At(i, j)
			result.Set(i, j, 1.0/(1.0+math.Exp(-val)))
		}
	}

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r, c, nil),
		op:           OpSigmoid,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					sig := out.data.At(i, j)
					grad := out.grad.At(i, j) * sig * (1 - sig)
					t.grad.Set(i, j, t.grad.At(i, j)+grad)
				}
			}
		}
	}

	return out
}

// Tanh Tanh激活函数
func (t *Tensor) Tanh() *Tensor {
	r, c := t.data.Dims()

	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Tanh(t.data.At(i, j)))
		}
	}

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r, c, nil),
		op:           OpTanh,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					tanh := out.data.At(i, j)
					grad := out.grad.At(i, j) * (1 - math.Pow(tanh, 2))
					t.grad.Set(i, j, t.grad.At(i, j)+grad)
				}
			}
		}
	}

	return out
}

// Softmax Softmax激活函数
func (t *Tensor) Softmax(dim int) *Tensor {
	r, c := t.data.Dims()

	result := mat.NewDense(r, c, nil)

	if dim == 1 { // 按行
		for i := 0; i < r; i++ {
			row := make([]float64, c)
			maxVal := -math.MaxFloat64
			for j := 0; j < c; j++ {
				if t.data.At(i, j) > maxVal {
					maxVal = t.data.At(i, j)
				}
			}

			expSum := 0.0
			for j := 0; j < c; j++ {
				row[j] = math.Exp(t.data.At(i, j) - maxVal)
				expSum += row[j]
			}

			for j := 0; j < c; j++ {
				result.Set(i, j, row[j]/expSum)
			}
		}
	}

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r, c, nil),
		op:           OpSoftmax,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	// Softmax梯度较复杂，简化处理
	out.gradFn = func() {}

	return out
}

// ==================== 反向传播 ====================

// Backward 反向传播
func (t *Tensor) Backward() {
	if !t.requiresGrad {
		return
	}

	// 设置输出梯度为1
	r, c := t.data.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t.grad.Set(i, j, 1.0)
		}
	}

	// 拓扑排序
	topOrder := topologicalSort(t)

	// 反向计算梯度
	for i := len(topOrder) - 1; i >= 0; i-- {
		node := topOrder[i]
		if node.gradFn != nil {
			node.gradFn()
		}
	}
}

// topologicalSort 拓扑排序
func topologicalSort(root *Tensor) []*Tensor {
	visited := make(map[int64]bool)
	order := make([]*Tensor, 0)

	var visit func(*Tensor)
	visit = func(t *Tensor) {
		if visited[t.id] {
			return
		}
		visited[t.id] = true

		for _, prev := range t.prev {
			if prev != nil {
				visit(prev)
			}
		}

		order = append(order, t)
	}

	visit(root)
	return order
}

// ==================== 标量操作 ====================

// Item 获取标量值（仅适用于单元素张量）
func (t *Tensor) Item() float64 {
	r, c := t.data.Dims()
	if r != 1 || c != 1 {
		panic("Item() called on non-scalar tensor")
	}
	return t.data.At(0, 0)
}

// Sum 求和
func (t *Tensor) Sum() *Tensor {
	r, c := t.data.Dims()
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += t.data.At(i, j)
		}
	}

	result := mat.NewDense(1, 1, []float64{sum})
	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(1, 1, nil),
		op:           OpSum,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			gradVal := out.grad.At(0, 0)
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					t.grad.Set(i, j, t.grad.At(i, j)+gradVal)
				}
			}
		}
	}

	return out
}

// Mean 求平均
func (t *Tensor) Mean() *Tensor {
	sum := t.Sum()
	count := float64(t.Numel())

	result := mat.NewDense(1, 1, nil)
	result.Scale(1.0/count, sum.data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(1, 1, nil),
		op:           OpMean,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			gradVal := out.grad.At(0, 0) / count
			r, c := t.data.Dims()
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					t.grad.Set(i, j, t.grad.At(i, j)+gradVal)
				}
			}
		}
	}

	return out
}

// Scale 缩放
func (t *Tensor) Scale(alpha float64) *Tensor {
	r, c := t.data.Dims()

	result := mat.NewDense(r, c, nil)
	result.Scale(alpha, t.data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(r, c, nil),
		op:           OpMul,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	out.gradFn = func() {
		if t.requiresGrad {
			gradTemp := mat.NewDense(r, c, nil)
			gradTemp.Scale(alpha, out.grad)
			t.grad.Add(t.grad, gradTemp)
		}
	}

	return out
}

// ==================== 其他操作 ====================

// View 重塑视图
func (t *Tensor) View(rows, cols int) *Tensor {
	if t.Numel() != rows*cols {
		panic("View: incompatible size")
	}

	data := t.data.RawMatrix().Data
	result := mat.NewDense(rows, cols, data)

	out := &Tensor{
		data:         result,
		grad:         mat.NewDense(rows, cols, nil),
		op:           OpReshape,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	return out
}

// T 转置
func (t *Tensor) T() *Tensor {
	r, c := t.data.Dims()

	resultMat := mat.NewDense(c, r, nil)
	resultMat.Copy(t.data.T())

	out := &Tensor{
		data:         resultMat,
		grad:         mat.NewDense(c, r, nil),
		op:           OpTranspose,
		prev:         []*Tensor{t},
		requiresGrad: t.requiresGrad,
		id:           newTensorID(),
	}

	return out
}

// Clone 克隆张量
func (t *Tensor) Clone() *Tensor {
	r, c := t.data.Dims()
	data := make([]float64, r*c)
	copy(data, t.data.RawMatrix().Data)

	return NewTensor(data, r, c)
}

// Detach 分离张量（不计算梯度）
func (t *Tensor) Detach() *Tensor {
	return NewTensorFromDense(t.data)
}
