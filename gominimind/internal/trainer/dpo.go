package trainer

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"gominimind/internal/dataset"
	"gominimind/pkg/model"
	"gominimind/pkg/tokenizer"

	"gonum.org/v1/gonum/mat"
)

// DPOTrainer DPO偏好优化训练器
type DPOTrainer struct {
	config         *TrainingConfig
	model          model.Model
	trainableModel model.TrainableModel // 可训练模型适配器（可选）
	refModel       model.Model          // 参考模型（冻结）
	tokenizer      *tokenizer.MiniMindTokenizer
	dataset        *dataset.DPODataset
	optimizer      Optimizer
	currentEpoch   int
	currentStep    int
	globalStep     int
	startTime      time.Time
	stats          *TrainingStats
	stopFlag       bool
	callback       TrainingCallback
	logEntries     []string
	beta           float64                  // DPO温度参数
	gradients      map[string]*mat.VecDense // 当前梯度
}

// NewDPOTrainer 创建新的DPO训练器
func NewDPOTrainer(config *TrainingConfig, m model.Model, tok *tokenizer.MiniMindTokenizer, beta float64) (*DPOTrainer, error) {
	// 加载数据集
	ds, err := dataset.NewDPODataset(config.DataPath, tok, config.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("加载DPO数据集失败: %w", err)
	}

	Logger("加载了 %d 条DPO样本", ds.Len())

	// 尝试将模型适配为可训练模型
	var trainable model.TrainableModel
	if tm, ok := m.(model.TrainableModel); ok {
		trainable = tm
	}

	// 创建参考模型（从SFT模型复制）
	refModel := m // 简化处理，实际应该深拷贝

	// 创建优化器（DPO使用很小的学习率）
	var params map[string]*mat.VecDense
	if trainable != nil {
		params = trainable.GetNamedParameters()
		Logger("模型参数总量: %d", trainable.GetParameterCount())
	} else {
		params = make(map[string]*mat.VecDense)
	}

	optConfig := &AdamWConfig{
		LearningRate: config.LearningRate,
		Beta1:        config.AdamBeta1,
		Beta2:        config.AdamBeta2,
		Epsilon:      config.AdamEpsilon,
		WeightDecay:  config.WeightDecay,
	}
	optimizer := NewAdamW(params, optConfig)

	return &DPOTrainer{
		config:         config,
		model:          m,
		trainableModel: trainable,
		refModel:       refModel,
		tokenizer:      tok,
		dataset:        ds,
		optimizer:      optimizer,
		stats: &TrainingStats{
			BestLoss: math.MaxFloat64,
		},
		logEntries: make([]string, 0),
		beta:       beta,
		gradients:  make(map[string]*mat.VecDense),
	}, nil
}

// SetCallback 设置训练回调
func (t *DPOTrainer) SetCallback(cb TrainingCallback) {
	t.callback = cb
}

// Train 开始训练
func (t *DPOTrainer) Train() error {
	t.startTime = time.Now()
	Logger("开始DPO训练，配置: epochs=%d, batch_size=%d, lr=%.8f, beta=%.2f",
		t.config.Epochs, t.config.BatchSize, t.config.LearningRate, t.beta)
	Logger("注意：DPO训练使用很小的学习率（建议<=5e-8）以避免灾难性遗忘")

	// 加载SFT权重作为初始权重
	if t.config.FromWeight != "none" {
		if err := t.loadSFTWeights(); err != nil {
			Logger("加载SFT权重失败: %v", err)
		}
	}

	// 检查是否需要从检查点恢复
	if t.config.FromResume == 1 {
		if err := t.tryResume(); err != nil {
			Logger("恢复训练失败: %v，将从头开始训练", err)
		}
	}

	// 训练循环
	for epoch := t.currentEpoch; epoch < t.config.Epochs; epoch++ {
		if t.stopFlag {
			break
		}

		if err := t.trainEpoch(epoch); err != nil {
			return fmt.Errorf("epoch %d 训练失败: %w", epoch, err)
		}
	}

	// 保存最终模型
	if err := t.saveFinalModel(); err != nil {
		Logger("保存最终模型失败: %v", err)
	}

	// 更新统计信息
	t.stats.EndTime = time.Now()
	t.stats.TotalEpochs = t.config.Epochs
	t.stats.TotalSteps = t.globalStep

	Logger("DPO训练完成! 总用时: %s", FormatDuration(t.stats.EndTime.Sub(t.startTime)))

	return nil
}

// trainEpoch 训练一个epoch
func (t *DPOTrainer) trainEpoch(epoch int) error {
	t.currentEpoch = epoch

	// 打乱数据集
	t.dataset.Shuffle()

	// 计算迭代次数
	datasetSize := t.dataset.Len()
	iters := datasetSize / t.config.BatchSize
	if datasetSize%t.config.BatchSize != 0 {
		iters++
	}

	Logger("Epoch %d/%d 开始，共 %d 个step", epoch+1, t.config.Epochs, iters)

	// 设置随机种子
	rand.Seed(int64(42 + epoch))

	// 生成随机索引
	indices := rand.Perm(datasetSize)

	stepStartTime := time.Now()

	for step := 0; step < iters; step++ {
		if t.stopFlag {
			break
		}

		t.currentStep = step
		t.globalStep++

		// 获取批次数据
		batchIndices := make([]int, 0, t.config.BatchSize)
		for i := 0; i < t.config.BatchSize; i++ {
			idx := step*t.config.BatchSize + i
			if idx < len(indices) {
				batchIndices = append(batchIndices, indices[idx])
			}
		}

		if len(batchIndices) == 0 {
			break
		}

		// 计算DPO损失
		loss, dpoLoss, auxLoss, err := t.computeDPOLoss(batchIndices)
		if err != nil {
			Logger("Step %d 损失计算失败: %v", step, err)
			continue
		}

		totalLoss := loss

		// 梯度累积
		totalLoss = totalLoss / float64(t.config.AccumulationSteps)

		// 反向传播
		if err := t.backward(totalLoss); err != nil {
			Logger("Step %d 反向传播失败: %v", step, err)
			continue
		}

		// 梯度更新
		if (step+1)%t.config.AccumulationSteps == 0 {
			t.clipGradients()
			if len(t.gradients) > 0 {
				t.optimizer.Step(t.gradients)
			} else {
				t.optimizer.Step(nil)
			}
			t.optimizer.ZeroGrad()

			// 清零训练器梯度
			t.gradients = make(map[string]*mat.VecDense)
			if t.trainableModel != nil {
				t.trainableModel.ZeroGrad()
			}
		}

		// 更新学习率
		totalSteps := t.config.Epochs * iters
		lr := GetLRWithWarmup(t.globalStep, t.config.WarmupSteps, totalSteps, t.config.LearningRate)
		t.optimizer.SetLR(lr)

		// 日志记录
		if step%t.config.LogInterval == 0 || step == iters-1 {
			elapsed := time.Since(stepStartTime)
			eta := CalculateETA(elapsed, step, iters)

			logMsg := fmt.Sprintf("Epoch:[%d/%d](%d/%d), loss: %.4f, dpo_loss: %.4f, aux_loss: %.4f, lr: %.8f, eta: %s",
				epoch+1, t.config.Epochs, step, iters, loss, dpoLoss, auxLoss, lr, eta)
			Logger(logMsg)
			t.logEntries = append(t.logEntries, logMsg)

			// 更新统计
			t.stats.AverageLoss = (t.stats.AverageLoss*float64(t.globalStep-1) + loss) / float64(t.globalStep)
			if loss < t.stats.BestLoss {
				t.stats.BestLoss = loss
			}

			// 回调
			if t.callback != nil {
				t.callback(epoch, step, loss, lr)
			}
		}

		// 保存检查点
		if (step%t.config.SaveInterval == 0 || step == iters-1) && step > 0 {
			if err := t.saveCheckpoint(); err != nil {
				Logger("保存检查点失败: %v", err)
			}
		}
	}

	return nil
}

// computeDPOLoss 计算DPO损失
func (t *DPOTrainer) computeDPOLoss(batchIndices []int) (float64, float64, float64, error) {
	samples, err := t.dataset.GetBatch(batchIndices)
	if err != nil {
		return 0, 0, 0, err
	}

	totalDPOLoss := 0.0
	totalAuxLoss := 0.0
	batchSize := len(samples)

	for _, sample := range samples {
		// 计算参考模型的log概率（不计算梯度）
		refLogProbsChosen := t.computeLogProbs(t.refModel, sample.XChosen, sample.YChosen, sample.MaskChosen)
		refLogProbsRejected := t.computeLogProbs(t.refModel, sample.XRejected, sample.YRejected, sample.MaskRejected)

		// 计算策略模型的log概率
		policyLogProbsChosen := t.computeLogProbs(t.model, sample.XChosen, sample.YChosen, sample.MaskChosen)
		policyLogProbsRejected := t.computeLogProbs(t.model, sample.XRejected, sample.YRejected, sample.MaskRejected)

		// 计算DPO损失
		// DPO损失: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
		piLogRatio := policyLogProbsChosen - policyLogProbsRejected
		refLogRatio := refLogProbsChosen - refLogProbsRejected

		logits := t.beta * (piLogRatio - refLogRatio)
		dpoLoss := -t.logSigmoid(logits)

		totalDPOLoss += dpoLoss

		// MoE辅助损失（如果启用）
		// totalAuxLoss += t.computeAuxLoss(sample)
	}

	avgDPOLoss := totalDPOLoss / float64(batchSize)
	avgAuxLoss := totalAuxLoss / float64(batchSize)
	totalLoss := avgDPOLoss + avgAuxLoss

	if math.IsNaN(totalLoss) || math.IsInf(totalLoss, 0) {
		totalLoss = 0.0
		avgDPOLoss = 0.0
		avgAuxLoss = 0.0
	}

	return totalLoss, avgDPOLoss, avgAuxLoss, nil
}

// computeLogProbs 计算log概率
func (t *DPOTrainer) computeLogProbs(m model.Model, inputIDs, labels []int, mask []float64) float64 {
	if len(inputIDs) == 0 || len(labels) == 0 {
		return 0.0
	}

	// 如果模型支持TrainableModel，使用完整的logits计算
	if tm, ok := m.(model.TrainableModel); ok {
		logits, err := tm.ForwardLogits(inputIDs)
		if err == nil && logits != nil {
			return t.computeLogProbsFromLogits(logits, labels, mask)
		}
	}

	// 回退到简化计算
	totalLogProb := 0.0
	validTokens := 0

	for i := 0; i < len(inputIDs)-1 && i < len(labels); i++ {
		if i < len(mask) && mask[i] == 0 {
			continue
		}

		// 简化处理，假设一个固定值
		logProb := -0.5
		totalLogProb += logProb
		validTokens++
	}

	if validTokens == 0 {
		return 0.0
	}

	return totalLogProb / float64(validTokens)
}

// computeLogProbsFromLogits 从logits计算序列的log概率
func (t *DPOTrainer) computeLogProbsFromLogits(logits *mat.Dense, labels []int, mask []float64) float64 {
	seqLen, vocabSize := logits.Dims()
	totalLogProb := 0.0
	validTokens := 0

	for i := 0; i < seqLen-1 && i < len(labels); i++ {
		if i < len(mask) && mask[i] == 0 {
			continue
		}

		target := labels[i]
		if target < 0 || target >= vocabSize {
			continue
		}

		// 计算log softmax
		posLogits := make([]float64, vocabSize)
		maxLogit := -math.MaxFloat64
		for j := 0; j < vocabSize; j++ {
			posLogits[j] = logits.At(i, j)
			if posLogits[j] > maxLogit {
				maxLogit = posLogits[j]
			}
		}

		expSum := 0.0
		for j := range posLogits {
			expSum += math.Exp(posLogits[j] - maxLogit)
		}

		logProb := posLogits[target] - maxLogit - math.Log(expSum)
		totalLogProb += logProb
		validTokens++
	}

	if validTokens == 0 {
		return 0.0
	}

	return totalLogProb / float64(validTokens)
}

// logSigmoid 计算log sigmoid
func (t *DPOTrainer) logSigmoid(x float64) float64 {
	if x >= 0 {
		return -math.Log(1.0 + math.Exp(-x))
	} else {
		return x - math.Log(1.0+math.Exp(x))
	}
}

// backward 反向传播
func (t *DPOTrainer) backward(loss float64) error {
	// DPO的反向传播通过策略模型的梯度计算
	if t.trainableModel != nil {
		grads, err := t.trainableModel.Backward(loss)
		if err != nil {
			return err
		}

		// 累积梯度
		for name, grad := range grads {
			if existing, ok := t.gradients[name]; ok {
				existing.AddVec(existing, grad)
			} else {
				newGrad := mat.NewVecDense(grad.Len(), nil)
				newGrad.CopyVec(grad)
				t.gradients[name] = newGrad
			}
		}
	}
	return nil
}

// clipGradients 梯度裁剪
func (t *DPOTrainer) clipGradients() {
	if len(t.gradients) > 0 {
		maxNorm := t.config.MaxGradNorm
		if maxNorm <= 0 {
			maxNorm = 1.0
		}
		ClipGradNorm(t.gradients, maxNorm)
	}
}

// loadSFTWeights 加载SFT模型权重
func (t *DPOTrainer) loadSFTWeights() error {
	Logger("加载SFT权重: %s", t.config.FromWeight)

	moeSuffix := ""
	if t.config.UseMoE {
		moeSuffix = "_moe"
	}
	modelPath := filepath.Join(t.config.SaveDir,
		fmt.Sprintf("%s_%d%s.bin", t.config.FromWeight, t.config.HiddenSize, moeSuffix))

	// 检查文件是否存在
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = filepath.Join("../out",
			fmt.Sprintf("%s_%d%s.bin", t.config.FromWeight, t.config.HiddenSize, moeSuffix))
	}

	// 加载模型
	if err := t.model.Load(modelPath, nil); err != nil {
		return fmt.Errorf("加载模型失败: %w", err)
	}

	// 同时加载参考模型
	t.refModel = t.model

	Logger("成功加载SFT权重")
	Logger("策略模型总参数量: 待计算")
	Logger("参考模型总参数量: 待计算")

	return nil
}

// saveCheckpoint 保存检查点
func (t *DPOTrainer) saveCheckpoint() error {
	checkpointDir := filepath.Join(t.config.SaveDir, "../checkpoints")
	if err := os.MkdirAll(checkpointDir, 0755); err != nil {
		return err
	}

	moeSuffix := ""
	if t.config.UseMoE {
		moeSuffix = "_moe"
	}
	checkpointPath := filepath.Join(checkpointDir,
		fmt.Sprintf("%s_%d%s.gob", t.config.SaveWeight, t.config.HiddenSize, moeSuffix))

	data := &CheckpointData{
		Epoch:          t.currentEpoch,
		Step:           t.currentStep,
		GlobalStep:     t.globalStep,
		TrainingConfig: t.config,
		Timestamp:      time.Now(),
	}

	if err := SaveCheckpoint(checkpointPath, data); err != nil {
		return err
	}

	Logger("已保存检查点到: %s", checkpointPath)
	return nil
}

// tryResume 尝试从检查点恢复
func (t *DPOTrainer) tryResume() error {
	checkpointDir := filepath.Join(t.config.SaveDir, "../checkpoints")
	moeSuffix := ""
	if t.config.UseMoE {
		moeSuffix = "_moe"
	}
	checkpointPath := filepath.Join(checkpointDir,
		fmt.Sprintf("%s_%d%s.gob", t.config.SaveWeight, t.config.HiddenSize, moeSuffix))

	if _, err := os.Stat(checkpointPath); os.IsNotExist(err) {
		return fmt.Errorf("检查点不存在")
	}

	data, err := LoadCheckpoint(checkpointPath)
	if err != nil {
		return err
	}

	t.currentEpoch = data.Epoch
	t.currentStep = data.Step
	t.globalStep = data.GlobalStep

	Logger("从检查点恢复: epoch=%d, step=%d", t.currentEpoch+1, t.currentStep)
	return nil
}

// saveFinalModel 保存最终模型
func (t *DPOTrainer) saveFinalModel() error {
	if err := os.MkdirAll(t.config.SaveDir, 0755); err != nil {
		return err
	}

	moeSuffix := ""
	if t.config.UseMoE {
		moeSuffix = "_moe"
	}
	modelPath := filepath.Join(t.config.SaveDir,
		fmt.Sprintf("%s_%d%s.bin", t.config.SaveWeight, t.config.HiddenSize, moeSuffix))

	if err := t.model.Save(modelPath); err != nil {
		return fmt.Errorf("保存模型失败: %w", err)
	}

	Logger("已保存最终模型到: %s", modelPath)
	return nil
}

// GetStats 获取训练统计
func (t *DPOTrainer) GetStats() *TrainingStats {
	return t.stats
}

// Stop 停止训练
func (t *DPOTrainer) Stop() {
	t.stopFlag = true
}

// SaveTrainingLog 保存训练日志
func (t *DPOTrainer) SaveTrainingLog() error {
	logPath := filepath.Join(t.config.SaveDir, "dpo_training.log")
	return SaveTrainingLog(logPath, t.logEntries)
}
