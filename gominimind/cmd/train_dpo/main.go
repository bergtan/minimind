package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"gominimind/internal/trainer"
	"gominimind/pkg/config"
	"gominimind/pkg/logger"
	"gominimind/pkg/model"
	"gominimind/pkg/tokenizer"
	"gominimind/pkg/types"
)

func main() {
	// 定义命令行参数
	var (
		saveDir           = flag.String("save_dir", "../out", "模型保存目录")
		saveWeight        = flag.String("save_weight", "rlhf", "保存权重的前缀名")
		epochs            = flag.Int("epochs", 6, "训练轮数")
		batchSize         = flag.Int("batch_size", 16, "batch size")
		learningRate      = flag.Float64("learning_rate", 4e-8, "初始学习率（DPO使用极小学习率）")
		device            = flag.String("device", "cpu", "训练设备 (cpu)")
		numWorkers        = flag.Int("num_workers", 4, "数据加载线程数")
		accumulationSteps = flag.Int("accumulation_steps", 1, "梯度累积步数")
		gradClip          = flag.Float64("grad_clip", 1.0, "梯度裁剪阈值")
		logInterval       = flag.Int("log_interval", 50, "日志打印间隔")
		saveInterval      = flag.Int("save_interval", 500, "模型保存间隔")
		hiddenSize        = flag.Int("hidden_size", 512, "隐藏层维度")
		numHiddenLayers   = flag.Int("num_hidden_layers", 8, "隐藏层数量")
		maxSeqLen         = flag.Int("max_seq_len", 340, "训练的最大截断长度")
		useMoE            = flag.Bool("use_moe", false, "是否使用MoE架构")
		numAttentionHeads = flag.Int("num_attention_heads", 8, "注意力头数")
		dataPath          = flag.String("data_path", "../dataset/dpo.jsonl", "DPO数据路径")
		fromWeight        = flag.String("from_weight", "full_sft", "基于哪个权重训练（建议从SFT权重开始）")
		fromResume        = flag.Int("from_resume", 0, "是否自动检测&续训（0=否，1=是）")
		beta              = flag.Float64("beta", 0.1, "DPO温度参数beta")
		warmupSteps       = flag.Int("warmup_steps", 100, "warmup步数")
		weightDecay       = flag.Float64("weight_decay", 0.01, "权重衰减")
		beta1             = flag.Float64("beta1", 0.9, "Adam beta1")
		beta2             = flag.Float64("beta2", 0.999, "Adam beta2")
		epsilon           = flag.Float64("epsilon", 1e-8, "Adam epsilon")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "用法: %s [选项]\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "\n选项:")
		flag.PrintDefaults()
		fmt.Fprintln(os.Stderr, "\n示例:")
		fmt.Fprintln(os.Stderr, "  从SFT权重开始DPO训练:")
		fmt.Fprintln(os.Stderr, "    go run cmd/train_dpo/main.go -from_weight full_sft")
		fmt.Fprintln(os.Stderr, "  从检查点恢复训练:")
		fmt.Fprintln(os.Stderr, "    go run cmd/train_dpo/main.go -from_weight full_sft -from_resume 1")
		fmt.Fprintln(os.Stderr, "  使用自定义参数:")
		fmt.Fprintln(os.Stderr, "    go run cmd/train_dpo/main.go -beta 0.2 -learning_rate 5e-8")
	}

	flag.Parse()

	// 初始化日志
	logger.Init(logger.Config{
		Level:  "info",
		Output: "console",
	})

	trainer.Logger("========================================")
	trainer.Logger("MiniMind Go DPO训练启动")
	trainer.Logger("========================================")
	trainer.Logger("注意: DPO训练使用极小的学习率（建议<=4e-8）")
	trainer.Logger("      以避免对预训练知识产生灾难性遗忘")

	// 创建训练配置
	cfg := trainer.DefaultTrainingConfig()
	cfg.SaveDir = *saveDir
	cfg.SaveWeight = *saveWeight
	cfg.Epochs = *epochs
	cfg.BatchSize = *batchSize
	cfg.LearningRate = *learningRate
	cfg.Device = *device
	cfg.NumWorkers = *numWorkers
	cfg.AccumulationSteps = *accumulationSteps
	cfg.GradClip = *gradClip
	cfg.LogInterval = *logInterval
	cfg.SaveInterval = *saveInterval
	cfg.HiddenSize = *hiddenSize
	cfg.NumHiddenLayers = *numHiddenLayers
	cfg.MaxSeqLen = *maxSeqLen
	cfg.UseMoE = *useMoE
	cfg.NumAttentionHeads = *numAttentionHeads
	cfg.DataPath = *dataPath
	cfg.FromWeight = *fromWeight
	cfg.FromResume = *fromResume
	cfg.WarmupSteps = *warmupSteps
	cfg.WeightDecay = *weightDecay
	cfg.AdamBeta1 = *beta1
	cfg.AdamBeta2 = *beta2
	cfg.AdamEpsilon = *epsilon

	// 打印配置
	trainer.Logger("DPO配置:")
	trainer.Logger("  数据路径: %s", cfg.DataPath)
	trainer.Logger("  保存目录: %s", cfg.SaveDir)
	trainer.Logger("  权重前缀: %s", cfg.SaveWeight)
	trainer.Logger("  SFT权重: %s", cfg.FromWeight)
	trainer.Logger("  训练轮数: %d", cfg.Epochs)
	trainer.Logger("  Batch Size: %d", cfg.BatchSize)
	trainer.Logger("  学习率: %.8f", cfg.LearningRate)
	trainer.Logger("  DPO Beta: %.2f", *beta)
	trainer.Logger("  梯度累积: %d", cfg.AccumulationSteps)
	trainer.Logger("  隐藏层维度: %d", cfg.HiddenSize)
	trainer.Logger("  层数: %d", cfg.NumHiddenLayers)
	trainer.Logger("  最大序列长度: %d", cfg.MaxSeqLen)
	trainer.Logger("  使用MoE: %v", cfg.UseMoE)

	// 加载配置
	configPath := filepath.Join("..", "config", "config.yaml")
	_, err := config.LoadConfig(configPath)
	if err != nil {
		trainer.Logger("使用默认配置")
	}

	// 加载tokenizer
	tokenizerPath := filepath.Join("..", "weights", "tokenizer.json")
	tok, err := tokenizer.Load(tokenizerPath)
	if err != nil {
		trainer.Logger("加载tokenizer失败: %v，创建新tokenizer", err)
		tok = tokenizer.NewWithBPE(nil, nil, 0, 0)
	}

	// 创建模型
	modelConfig := &types.ModelConfig{
		HiddenSize:            cfg.HiddenSize,
		NumLayers:             cfg.NumHiddenLayers,
		NumHeads:              cfg.NumAttentionHeads,
		MaxPositionEmbeddings: cfg.MaxSeqLen,
		VocabSize:             tok.VocabSize(),
	}

	m, err := model.NewMiniMindModel(modelConfig)
	if err != nil {
		trainer.Logger("创建DPO模型失败: %v", err)
		os.Exit(1)
	}

	// 创建DPO训练器
	t, err := trainer.NewDPOTrainer(cfg, m, tok, *beta)
	if err != nil {
		trainer.Logger("创建DPO训练器失败: %v", err)
		os.Exit(1)
	}

	// 开始训练
	if err := t.Train(); err != nil {
		trainer.Logger("DPO训练失败: %v", err)
		os.Exit(1)
	}

	// 保存训练日志
	if err := t.SaveTrainingLog(); err != nil {
		trainer.Logger("保存训练日志失败: %v", err)
	}

	// 打印统计信息
	stats := t.GetStats()
	trainer.Logger("\nDPO训练统计:")
	trainer.Logger("  总训练时长: %s", trainer.FormatDuration(stats.EndTime.Sub(stats.StartTime)))
	trainer.Logger("  总步数: %d", stats.TotalSteps)
	trainer.Logger("  总轮数: %d", stats.TotalEpochs)
	trainer.Logger("  平均损失: %.4f", stats.AverageLoss)
	trainer.Logger("  最佳损失: %.4f", stats.BestLoss)

	trainer.Logger("\nDPO训练完成！")
}
