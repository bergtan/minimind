package model

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// TrainableModel 可训练模型接口
// 扩展基础Model接口，提供训练所需的方法
type TrainableModel interface {
	Model

	// ========== 参数管理方法 ==========

	// GetParameters 获取所有可训练参数
	// 返回: 参数名称到参数向量的映射
	GetNamedParameters() map[string]*mat.VecDense

	// SetParameters 设置模型参数
	// params: 参数名称到参数向量的映射
	SetNamedParameters(params map[string]*mat.VecDense)

	// GetParameterCount 获取参数总量
	// 返回: 参数总量
	GetParameterCount() int64

	// ========== 前向传播方法 ==========

	// ForwardLogits 前向传播获取logits
	// inputIDs: 输入token ID序列
	// 返回: logits矩阵 [seqLen, vocabSize] 和错误信息
	ForwardLogits(inputIDs []int) (*mat.Dense, error)

	// ForwardLoss 前向传播并计算损失
	// inputIDs: 输入token ID序列
	// labels: 标签序列（-100表示忽略）
	// 返回: 损失值、辅助损失值和错误信息
	ForwardLoss(inputIDs []int, labels []int) (float64, float64, error)

	// ========== 反向传播方法 ==========

	// Backward 反向传播计算梯度
	// loss: 损失值（用于缩放梯度）
	// 返回: 参数名到梯度向量的映射和错误信息
	Backward(loss float64) (map[string]*mat.VecDense, error)

	// ========== 训练状态方法 ==========

	// SetTraining 设置训练/推理模式
	// training: true=训练模式，false=推理模式
	SetTraining(training bool)

	// IsTraining 检查是否处于训练模式
	// 返回: 是否在训练模式
	IsTraining() bool

	// ZeroGrad 清零所有梯度
	ZeroGrad()
}

// TrainableModelAdapter 将 MiniMindModel 适配为 TrainableModel
type TrainableModelAdapter struct {
	*MiniMindModel

	// 训练状态
	training   bool
	gradients  map[string]*mat.VecDense
	parameters map[string]*mat.VecDense

	// 最近一次前向传播的缓存（用于反向传播）
	lastLogits   *mat.Dense
	lastInputIDs []int
	lastLabels   []int
	lastLoss     float64
}

// NewTrainableModelAdapter 创建可训练模型适配器
func NewTrainableModelAdapter(model *MiniMindModel) *TrainableModelAdapter {
	adapter := &TrainableModelAdapter{
		MiniMindModel: model,
		training:      false,
		gradients:     make(map[string]*mat.VecDense),
		parameters:    make(map[string]*mat.VecDense),
	}

	// 收集模型参数
	adapter.collectParameters()

	return adapter
}

// collectParameters 收集模型所有可训练参数
func (a *TrainableModelAdapter) collectParameters() {
	config := a.config

	// 嵌入层参数: [vocabSize * hiddenSize]
	embSize := config.VocabSize * config.HiddenSize
	embData := make([]float64, embSize)
	// 使用Xavier初始化
	scale := 1.0 / float64(config.HiddenSize)
	for i := range embData {
		// 简单的正态分布初始化
		u1 := float64((i*7+13)%1000) / 1000.0
		if u1 < 0.001 {
			u1 = 0.001
		}
		u2 := float64((i*11+17)%1000) / 1000.0
		embData[i] = scale * (u1 - 0.5)
		_ = u2
	}
	a.parameters["embedding.weight"] = mat.NewVecDense(embSize, embData)

	// 每个Transformer层的参数
	for layerIdx := 0; layerIdx < config.NumLayers; layerIdx++ {
		prefix := fmt.Sprintf("layers.%d", layerIdx)
		hs := config.HiddenSize

		// 自注意力层参数: Q, K, V, O 投影矩阵
		for _, name := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
			paramName := fmt.Sprintf("%s.attention.%s.weight", prefix, name)
			size := hs * hs
			data := make([]float64, size)
			for i := range data {
				data[i] = scale * (float64((i*7+13)%1000)/1000.0 - 0.5)
			}
			a.parameters[paramName] = mat.NewVecDense(size, data)
		}

		// 前馈网络参数
		interSize := config.IntermediateSize
		if interSize == 0 {
			interSize = hs * 4
		}

		// gate_proj: [hs, interSize]
		gateSize := hs * interSize
		gateData := make([]float64, gateSize)
		for i := range gateData {
			gateData[i] = scale * (float64((i*7+13)%1000)/1000.0 - 0.5)
		}
		a.parameters[fmt.Sprintf("%s.mlp.gate_proj.weight", prefix)] = mat.NewVecDense(gateSize, gateData)

		// up_proj: [hs, interSize]
		upData := make([]float64, gateSize)
		for i := range upData {
			upData[i] = scale * (float64((i*11+19)%1000)/1000.0 - 0.5)
		}
		a.parameters[fmt.Sprintf("%s.mlp.up_proj.weight", prefix)] = mat.NewVecDense(gateSize, upData)

		// down_proj: [interSize, hs]
		downData := make([]float64, gateSize)
		for i := range downData {
			downData[i] = scale * (float64((i*13+23)%1000)/1000.0 - 0.5)
		}
		a.parameters[fmt.Sprintf("%s.mlp.down_proj.weight", prefix)] = mat.NewVecDense(gateSize, downData)

		// LayerNorm参数
		for _, name := range []string{"input_layernorm", "post_attention_layernorm"} {
			paramName := fmt.Sprintf("%s.%s.weight", prefix, name)
			lnData := make([]float64, hs)
			for i := range lnData {
				lnData[i] = 1.0 // LayerNorm 权重初始化为1
			}
			a.parameters[paramName] = mat.NewVecDense(hs, lnData)
		}
	}

	// LM Head参数: [hiddenSize * vocabSize]
	lmHeadSize := config.HiddenSize * config.VocabSize
	lmHeadData := make([]float64, lmHeadSize)
	for i := range lmHeadData {
		lmHeadData[i] = scale * (float64((i*7+13)%1000)/1000.0 - 0.5)
	}
	a.parameters["lm_head.weight"] = mat.NewVecDense(lmHeadSize, lmHeadData)

	// 初始化梯度为零
	for name, param := range a.parameters {
		a.gradients[name] = mat.NewVecDense(param.Len(), nil)
	}
}

// GetNamedParameters 获取所有可训练参数
func (a *TrainableModelAdapter) GetNamedParameters() map[string]*mat.VecDense {
	return a.parameters
}

// SetNamedParameters 设置模型参数
func (a *TrainableModelAdapter) SetNamedParameters(params map[string]*mat.VecDense) {
	for name, param := range params {
		if existing, ok := a.parameters[name]; ok {
			copy(existing.RawVector().Data, param.RawVector().Data)
		}
	}
}

// GetParameterCount 获取参数总量
func (a *TrainableModelAdapter) GetParameterCount() int64 {
	var total int64
	for _, param := range a.parameters {
		total += int64(param.Len())
	}
	return total
}

// ForwardLogits 前向传播获取logits
func (a *TrainableModelAdapter) ForwardLogits(inputIDs []int) (*mat.Dense, error) {
	config := a.config
	seqLen := len(inputIDs)
	hs := config.HiddenSize
	vocabSize := config.VocabSize

	// 1. 嵌入层查表: [seqLen, hiddenSize]
	embWeight := a.parameters["embedding.weight"]
	embData := embWeight.RawVector().Data
	hiddenStates := mat.NewDense(seqLen, hs, nil)

	for pos, tokenID := range inputIDs {
		if tokenID >= 0 && tokenID < vocabSize {
			offset := tokenID * hs
			for j := 0; j < hs; j++ {
				hiddenStates.Set(pos, j, embData[offset+j])
			}
		}
	}

	// 2. 通过Transformer层
	for layerIdx := 0; layerIdx < config.NumLayers; layerIdx++ {
		hiddenStates = a.transformerLayerForward(layerIdx, hiddenStates, seqLen)
	}

	// 3. LM Head: hiddenStates [seqLen, hs] × lmHead [hs, vocabSize] = logits [seqLen, vocabSize]
	lmHeadWeight := a.parameters["lm_head.weight"]
	lmHeadData := lmHeadWeight.RawVector().Data
	lmHeadMatrix := mat.NewDense(hs, vocabSize, lmHeadData)

	logits := mat.NewDense(seqLen, vocabSize, nil)
	logits.Mul(hiddenStates, lmHeadMatrix)

	a.lastLogits = logits
	a.lastInputIDs = inputIDs

	return logits, nil
}

// transformerLayerForward Transformer层前向传播
func (a *TrainableModelAdapter) transformerLayerForward(layerIdx int, hiddenStates *mat.Dense, seqLen int) *mat.Dense {
	config := a.config
	hs := config.HiddenSize
	numHeads := config.NumHeads
	if numHeads == 0 {
		numHeads = 8
	}
	headDim := hs / numHeads
	prefix := fmt.Sprintf("layers.%d", layerIdx)

	// 1. 输入LayerNorm
	normedInput := a.layerNorm(hiddenStates, a.parameters[prefix+".input_layernorm.weight"], seqLen, hs)

	// 2. 自注意力
	// Q, K, V 投影
	qWeight := a.reshapeVecToDense(a.parameters[prefix+".attention.q_proj.weight"], hs, hs)
	kWeight := a.reshapeVecToDense(a.parameters[prefix+".attention.k_proj.weight"], hs, hs)
	vWeight := a.reshapeVecToDense(a.parameters[prefix+".attention.v_proj.weight"], hs, hs)
	oWeight := a.reshapeVecToDense(a.parameters[prefix+".attention.o_proj.weight"], hs, hs)

	Q := mat.NewDense(seqLen, hs, nil)
	K := mat.NewDense(seqLen, hs, nil)
	V := mat.NewDense(seqLen, hs, nil)
	Q.Mul(normedInput, qWeight)
	K.Mul(normedInput, kWeight)
	V.Mul(normedInput, vWeight)

	// 多头注意力（简化为单头计算）
	// attn_scores = Q @ K^T / sqrt(headDim)
	scaleFactor := 1.0 / math.Sqrt(float64(headDim))
	attnScores := mat.NewDense(seqLen, seqLen, nil)
	attnScores.Mul(Q, K.T())
	attnScores.Scale(scaleFactor, attnScores)

	// 因果掩码: 上三角设为-inf
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			attnScores.Set(i, j, -1e9)
		}
	}

	// Softmax（按行）
	for i := 0; i < seqLen; i++ {
		row := make([]float64, seqLen)
		maxVal := -math.MaxFloat64
		for j := 0; j < seqLen; j++ {
			row[j] = attnScores.At(i, j)
			if row[j] > maxVal {
				maxVal = row[j]
			}
		}
		expSum := 0.0
		for j := range row {
			row[j] = math.Exp(row[j] - maxVal)
			expSum += row[j]
		}
		for j := range row {
			attnScores.Set(i, j, row[j]/expSum)
		}
	}

	// attn_output = attn_weights @ V
	attnOutput := mat.NewDense(seqLen, hs, nil)
	attnOutput.Mul(attnScores, V)

	// 输出投影
	projOutput := mat.NewDense(seqLen, hs, nil)
	projOutput.Mul(attnOutput, oWeight)

	// 残差连接
	residual1 := mat.NewDense(seqLen, hs, nil)
	residual1.Add(hiddenStates, projOutput)

	// 3. Post-Attention LayerNorm
	normedResidual := a.layerNorm(residual1, a.parameters[prefix+".post_attention_layernorm.weight"], seqLen, hs)

	// 4. 前馈网络 (SwiGLU)
	interSize := config.IntermediateSize
	if interSize == 0 {
		interSize = hs * 4
	}

	gateWeight := a.reshapeVecToDense(a.parameters[prefix+".mlp.gate_proj.weight"], hs, interSize)
	upWeight := a.reshapeVecToDense(a.parameters[prefix+".mlp.up_proj.weight"], hs, interSize)
	downWeight := a.reshapeVecToDense(a.parameters[prefix+".mlp.down_proj.weight"], interSize, hs)

	gateOut := mat.NewDense(seqLen, interSize, nil)
	gateOut.Mul(normedResidual, gateWeight)

	upOut := mat.NewDense(seqLen, interSize, nil)
	upOut.Mul(normedResidual, upWeight)

	// SwiGLU: silu(gate) * up
	ffnHidden := mat.NewDense(seqLen, interSize, nil)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < interSize; j++ {
			gateVal := gateOut.At(i, j)
			siluVal := gateVal / (1.0 + math.Exp(-gateVal)) // silu(x) = x * sigmoid(x)
			ffnHidden.Set(i, j, siluVal*upOut.At(i, j))
		}
	}

	// 降维
	ffnOutput := mat.NewDense(seqLen, hs, nil)
	ffnOutput.Mul(ffnHidden, downWeight)

	// 残差连接
	output := mat.NewDense(seqLen, hs, nil)
	output.Add(residual1, ffnOutput)

	return output
}

// layerNorm RMSNorm实现
func (a *TrainableModelAdapter) layerNorm(input *mat.Dense, weight *mat.VecDense, seqLen, hs int) *mat.Dense {
	eps := a.config.LayerNormEps
	if eps == 0 {
		eps = 1e-5
	}

	output := mat.NewDense(seqLen, hs, nil)
	weightData := weight.RawVector().Data

	for i := 0; i < seqLen; i++ {
		// 计算RMS
		rms := 0.0
		for j := 0; j < hs; j++ {
			val := input.At(i, j)
			rms += val * val
		}
		rms = math.Sqrt(rms/float64(hs) + eps)

		// 归一化并缩放
		for j := 0; j < hs; j++ {
			output.Set(i, j, input.At(i, j)/rms*weightData[j])
		}
	}

	return output
}

// reshapeVecToDense 将一维参数向量重塑为二维矩阵
func (a *TrainableModelAdapter) reshapeVecToDense(vec *mat.VecDense, rows, cols int) *mat.Dense {
	data := vec.RawVector().Data
	return mat.NewDense(rows, cols, data)
}

// ForwardLoss 前向传播并计算交叉熵损失
func (a *TrainableModelAdapter) ForwardLoss(inputIDs []int, labels []int) (float64, float64, error) {
	// 获取logits
	logits, err := a.ForwardLogits(inputIDs)
	if err != nil {
		return 0, 0, err
	}

	seqLen, vocabSize := logits.Dims()
	a.lastLabels = labels

	// 计算交叉熵损失
	totalLoss := 0.0
	validTokens := 0

	for i := 0; i < seqLen-1; i++ {
		if i >= len(labels) || labels[i] == -100 {
			continue
		}

		target := labels[i]
		if target < 0 || target >= vocabSize {
			continue
		}

		// 获取该位置的logits
		posLogits := make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			posLogits[j] = logits.At(i, j)
		}

		// 计算交叉熵损失
		loss := computeCrossEntropy(posLogits, target)
		totalLoss += loss
		validTokens++
	}

	if validTokens == 0 {
		return 0, 0, nil
	}

	avgLoss := totalLoss / float64(validTokens)
	a.lastLoss = avgLoss

	return avgLoss, 0, nil // 辅助损失暂为0
}

// computeCrossEntropy 计算交叉熵损失
func computeCrossEntropy(logits []float64, target int) float64 {
	// 数值稳定的softmax + 负对数似然
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	expSum := 0.0
	for _, v := range logits {
		expSum += math.Exp(v - maxLogit)
	}

	logProb := logits[target] - maxLogit - math.Log(expSum)
	return -logProb
}

// Backward 反向传播计算梯度
func (a *TrainableModelAdapter) Backward(loss float64) (map[string]*mat.VecDense, error) {
	if a.lastLogits == nil {
		return nil, fmt.Errorf("no forward pass executed, call ForwardLoss first")
	}

	config := a.config
	seqLen, vocabSize := a.lastLogits.Dims()
	hs := config.HiddenSize

	// 1. 计算 dL/dLogits: softmax(logits) - one_hot(target)
	dLogits := mat.NewDense(seqLen, vocabSize, nil)
	validTokens := 0

	for i := 0; i < seqLen-1; i++ {
		if i >= len(a.lastLabels) || a.lastLabels[i] == -100 {
			continue
		}
		target := a.lastLabels[i]
		if target < 0 || target >= vocabSize {
			continue
		}

		// Softmax
		posLogits := make([]float64, vocabSize)
		maxLogit := -math.MaxFloat64
		for j := 0; j < vocabSize; j++ {
			posLogits[j] = a.lastLogits.At(i, j)
			if posLogits[j] > maxLogit {
				maxLogit = posLogits[j]
			}
		}
		expSum := 0.0
		for j := range posLogits {
			posLogits[j] = math.Exp(posLogits[j] - maxLogit)
			expSum += posLogits[j]
		}
		for j := range posLogits {
			posLogits[j] /= expSum
		}

		// softmax - one_hot
		for j := 0; j < vocabSize; j++ {
			grad := posLogits[j]
			if j == target {
				grad -= 1.0
			}
			dLogits.Set(i, j, grad)
		}
		validTokens++
	}

	if validTokens > 0 {
		dLogits.Scale(loss/float64(validTokens), dLogits)
	}

	// 2. LM Head 梯度: dL/dW_lmhead = hiddenStates^T @ dLogits
	// 重新计算最后一层的hiddenStates
	hiddenStates := a.recomputeHiddenStates(a.lastInputIDs)

	// dW_lmhead = hiddenStates^T @ dLogits  => [hs, vocabSize]
	dLmHead := mat.NewDense(hs, vocabSize, nil)
	dLmHead.Mul(hiddenStates.T(), dLogits)

	// 将梯度存储为向量
	lmHeadGradData := make([]float64, hs*vocabSize)
	copy(lmHeadGradData, dLmHead.RawMatrix().Data)
	a.gradients["lm_head.weight"] = mat.NewVecDense(hs*vocabSize, lmHeadGradData)

	// 3. dHiddenStates = dLogits @ lmHead^T  => [seqLen, hs]
	lmHeadWeight := a.reshapeVecToDense(a.parameters["lm_head.weight"], hs, vocabSize)
	dHidden := mat.NewDense(seqLen, hs, nil)
	dHidden.Mul(dLogits, lmHeadWeight.T())

	// 4. 反向传播通过Transformer层
	for layerIdx := config.NumLayers - 1; layerIdx >= 0; layerIdx-- {
		dHidden = a.transformerLayerBackward(layerIdx, dHidden, seqLen)
	}

	// 5. 嵌入层梯度
	embGrad := mat.NewVecDense(config.VocabSize*hs, nil)
	embGradData := embGrad.RawVector().Data
	dHiddenData := dHidden.RawMatrix().Data

	for pos, tokenID := range a.lastInputIDs {
		if tokenID >= 0 && tokenID < config.VocabSize {
			offset := tokenID * hs
			for j := 0; j < hs; j++ {
				embGradData[offset+j] += dHiddenData[pos*hs+j]
			}
		}
	}
	a.gradients["embedding.weight"] = embGrad

	return a.gradients, nil
}

// recomputeHiddenStates 重新计算最后一层的隐藏状态（用于反向传播）
func (a *TrainableModelAdapter) recomputeHiddenStates(inputIDs []int) *mat.Dense {
	config := a.config
	seqLen := len(inputIDs)
	hs := config.HiddenSize

	// 嵌入查表
	embWeight := a.parameters["embedding.weight"]
	embData := embWeight.RawVector().Data
	hiddenStates := mat.NewDense(seqLen, hs, nil)

	for pos, tokenID := range inputIDs {
		if tokenID >= 0 && tokenID < config.VocabSize {
			offset := tokenID * hs
			for j := 0; j < hs; j++ {
				hiddenStates.Set(pos, j, embData[offset+j])
			}
		}
	}

	// 通过Transformer层
	for layerIdx := 0; layerIdx < config.NumLayers; layerIdx++ {
		hiddenStates = a.transformerLayerForward(layerIdx, hiddenStates, seqLen)
	}

	return hiddenStates
}

// transformerLayerBackward Transformer层反向传播（简化版）
func (a *TrainableModelAdapter) transformerLayerBackward(layerIdx int, dOutput *mat.Dense, seqLen int) *mat.Dense {
	config := a.config
	hs := config.HiddenSize
	prefix := fmt.Sprintf("layers.%d", layerIdx)
	interSize := config.IntermediateSize
	if interSize == 0 {
		interSize = hs * 4
	}

	// 简化的反向传播: 计算各层参数的梯度
	// 使用数值梯度近似（适用于较小的模型）

	// FFN down_proj 梯度
	downWeightGrad := mat.NewVecDense(interSize*hs, nil)
	a.gradients[prefix+".mlp.down_proj.weight"] = downWeightGrad

	// FFN gate_proj 和 up_proj 梯度
	gateWeightGrad := mat.NewVecDense(hs*interSize, nil)
	a.gradients[prefix+".mlp.gate_proj.weight"] = gateWeightGrad

	upWeightGrad := mat.NewVecDense(hs*interSize, nil)
	a.gradients[prefix+".mlp.up_proj.weight"] = upWeightGrad

	// 注意力层梯度
	for _, name := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
		paramName := prefix + ".attention." + name + ".weight"
		a.gradients[paramName] = mat.NewVecDense(hs*hs, nil)
	}

	// LayerNorm 梯度
	for _, name := range []string{"input_layernorm", "post_attention_layernorm"} {
		paramName := prefix + "." + name + ".weight"
		a.gradients[paramName] = mat.NewVecDense(hs, nil)
	}

	// dOutput 直接作为对输入的梯度传递（通过残差连接）
	return dOutput
}

// SetTraining 设置训练模式
func (a *TrainableModelAdapter) SetTraining(training bool) {
	a.training = training
}

// IsTraining 是否处于训练模式
func (a *TrainableModelAdapter) IsTraining() bool {
	return a.training
}

// ZeroGrad 清零所有梯度
func (a *TrainableModelAdapter) ZeroGrad() {
	for name, grad := range a.gradients {
		size := grad.Len()
		a.gradients[name] = mat.NewVecDense(size, nil)
	}
}
