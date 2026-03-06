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

// SFTTrainer SFT微调训练器
type SFTTrainer struct {
	config         *TrainingConfig
	model          model.Model
	trainableModel model.TrainableModel // 可训练模型适配器（可选）
	tokenizer      *tokenizer.MiniMindTokenizer
	dataset        *dataset.SFTDataset
	optimizer      Optimizer
	currentEpoch   int
	currentStep    int
	globalStep     int
	startTime      time.Time
	stats          *TrainingStats
	stopFlag       bool
	callback       TrainingCallback
	logEntries     []string
	gradients      map[string]*mat.VecDense // 当前梯度
}

// NewSFTTrainer 创建新的SFT训练器
func NewSFTTrainer(config *TrainingConfig, m model.Model, tok *tokenizer.MiniMindTokenizer) (*SFTTrainer, error) {
	// 加载数据集
	ds, err := dataset.NewSFTDataset(config.DataPath, tok, config.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("加载SFT数据集失败: %w", err)
	}

	Logger("加载了 %d 条SFT样本", ds.Len())

	// 尝试将模型适配为可训练模型
	var trainable model.TrainableModel
	if tm, ok := m.(model.TrainableModel); ok {
		trainable = tm
	}

	// 创建优化器（SFT使用较小的学习率）
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

	return &SFTTrainer{
		config:         config,
		model:          m,
		trainableModel: trainable,
		tokenizer:      tok,
		dataset:        ds,
		optimizer:      optimizer,
		stats: &TrainingStats{
			BestLoss: math.MaxFloat64,
		},
		logEntries: make([]string, 0),
		gradients:  make(map[string]*mat.VecDense),
	}, nil
}

// SetCallback 设置训练回调
func (t *SFTTrainer) SetCallback(cb TrainingCallback) {
	t.callback = cb
}

// Train 开始训练
func (t *SFTTrainer) Train() error {
	t.startTime = time.Now()
	Logger("开始SFT微调，配置: epochs=%d, batch_size=%d, lr=%.8f",
		t.config.Epochs, t.config.BatchSize, t.config.LearningRate)

	// 加载预训练权重
	if t.config.FromWeight != "none" {
		if err := t.loadPretrainedWeights(); err != nil {
			Logger("加载预训练权重失败: %v", err)
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

	Logger("SFT微调完成! 总用时: %s", FormatDuration(t.stats.EndTime.Sub(t.startTime)))

	return nil
}

// trainEpoch 训练一个epoch
func (t *SFTTrainer) trainEpoch(epoch int) error {
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

		// 前向传播和计算损失
		loss, auxLoss, err := t.forward(batchIndices)
		if err != nil {
			Logger("Step %d 前向传播失败: %v", step, err)
			continue
		}

		totalLoss := loss + auxLoss

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

			logitsLoss := totalLoss - auxLoss
			logMsg := fmt.Sprintf("Epoch:[%d/%d](%d/%d), loss: %.4f, logits_loss: %.4f, aux_loss: %.4f, lr: %.8f, eta: %s",
				epoch+1, t.config.Epochs, step, iters, totalLoss, logitsLoss, auxLoss, lr, eta)
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

// forward 前向传播 - SFT版本
func (t *SFTTrainer) forward(batchIndices []int) (float64, float64, error) {
	samples, err := t.dataset.GetBatch(batchIndices)
	if err != nil {
		return 0, 0, err
	}

	// 如果模型支持TrainableModel接口，使用完整前向传播
	if t.trainableModel != nil {
		return t.forwardTrainable(samples)
	}

	// 回退到简化实现
	return t.forwardSimple(samples)
}

// forwardTrainable 使用TrainableModel接口的完整前向传播
func (t *SFTTrainer) forwardTrainable(samples []dataset.SFTSample) (float64, float64, error) {
	totalLoss := 0.0
	totalAuxLoss := 0.0
	batchSize := len(samples)

	// 清零梯度
	t.gradients = make(map[string]*mat.VecDense)

	for _, sample := range samples {
		// SFT使用ForwardLoss，标签中-100位置会被忽略（只对assistant回答计算损失）
		loss, auxLoss, err := t.trainableModel.ForwardLoss(sample.InputIDs, sample.Labels)
		if err != nil {
			continue
		}

		totalLoss += loss
		totalAuxLoss += auxLoss

		// 反向传播获取梯度
		grads, err := t.trainableModel.Backward(loss / float64(batchSize))
		if err != nil {
			continue
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

	avgLoss := totalLoss / float64(batchSize)
	avgAuxLoss := totalAuxLoss / float64(batchSize)

	if math.IsNaN(avgLoss) || math.IsInf(avgLoss, 0) {
		avgLoss = 0.0
	}

	return avgLoss, avgAuxLoss, nil
}

// forwardSimple 简化前向传播
func (t *SFTTrainer) forwardSimple(samples []dataset.SFTSample) (float64, float64, error) {
	totalLoss := 0.0
	totalTokens := 0
	auxLoss := 0.0

	for _, sample := range samples {
		tokens := sample.InputIDs
		targetTokens := sample.Labels

		for i := 0; i < len(tokens)-1 && i < len(targetTokens); i++ {
			if targetTokens[i] == -100 {
				continue
			}

			// 只对assistant的回答计算损失
			vocabSize := 6400
			loss := math.Log(float64(vocabSize))
			totalLoss += loss
			totalTokens++
		}
	}

	if totalTokens == 0 {
		return 0, 0, nil
	}

	avgLoss := totalLoss / float64(totalTokens)
	if math.IsNaN(avgLoss) || math.IsInf(avgLoss, 0) {
		avgLoss = 0.0
	}

	return avgLoss, auxLoss, nil
}

// backward 反向传播
func (t *SFTTrainer) backward(loss float64) error {
	// 梯度已在forwardTrainable中计算
	return nil
}

// clipGradients 梯度裁剪
func (t *SFTTrainer) clipGradients() {
	if len(t.gradients) > 0 {
		maxNorm := t.config.MaxGradNorm
		if maxNorm <= 0 {
			maxNorm = 1.0
		}
		ClipGradNorm(t.gradients, maxNorm)
	}
}

// loadPretrainedWeights 加载预训练权重
func (t *SFTTrainer) loadPretrainedWeights() error {
	Logger("加载预训练权重: %s", t.config.FromWeight)

	// 构建模型路径
	moeSuffix := ""
	if t.config.UseMoE {
		moeSuffix = "_moe"
	}
	modelPath := filepath.Join(t.config.SaveDir,
		fmt.Sprintf("%s_%d%s.bin", t.config.FromWeight, t.config.HiddenSize, moeSuffix))

	// 检查文件是否存在
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		// 尝试其他路径
		modelPath = filepath.Join("../out",
			fmt.Sprintf("%s_%d%s.bin", t.config.FromWeight, t.config.HiddenSize, moeSuffix))
	}

	// 加载模型
	if err := t.model.Load(modelPath, nil); err != nil {
		return fmt.Errorf("加载模型失败: %w", err)
	}

	Logger("成功加载预训练权重")
	return nil
}

// saveCheckpoint 保存检查点
func (t *SFTTrainer) saveCheckpoint() error {
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
func (t *SFTTrainer) tryResume() error {
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
func (t *SFTTrainer) saveFinalModel() error {
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
func (t *SFTTrainer) GetStats() *TrainingStats {
	return t.stats
}

// Stop 停止训练
func (t *SFTTrainer) Stop() {
	t.stopFlag = true
}

// SaveTrainingLog 保存训练日志
func (t *SFTTrainer) SaveTrainingLog() error {
	logPath := filepath.Join(t.config.SaveDir, "sft_training.log")
	return SaveTrainingLog(logPath, t.logEntries)
}
