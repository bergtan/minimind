package trainer

import (
	"time"

	"gonum.org/v1/gonum/mat"
)

// TrainingConfig 训练配置
type TrainingConfig struct {
	// 基本配置
	SaveDir           string  `json:"save_dir" yaml:"save_dir"`
	SaveWeight        string  `json:"save_weight" yaml:"save_weight"`
	Epochs            int     `json:"epochs" yaml:"epochs"`
	BatchSize         int     `json:"batch_size" yaml:"batch_size"`
	LearningRate      float64 `json:"learning_rate" yaml:"learning_rate"`
	Device            string  `json:"device" yaml:"device"`
	NumWorkers        int     `json:"num_workers" yaml:"num_workers"`
	AccumulationSteps int     `json:"accumulation_steps" yaml:"accumulation_steps"`
	GradClip          float64 `json:"grad_clip" yaml:"grad_clip"`
	LogInterval       int     `json:"log_interval" yaml:"log_interval"`
	SaveInterval      int     `json:"save_interval" yaml:"save_interval"`

	// 模型配置
	HiddenSize        int  `json:"hidden_size" yaml:"hidden_size"`
	NumHiddenLayers   int  `json:"num_hidden_layers" yaml:"num_hidden_layers"`
	MaxSeqLen         int  `json:"max_seq_len" yaml:"max_seq_len"`
	UseMoE            bool `json:"use_moe" yaml:"use_moe"`
	NumAttentionHeads int  `json:"num_attention_heads" yaml:"num_attention_heads"`

	// 数据配置
	DataPath   string `json:"data_path" yaml:"data_path"`
	FromWeight string `json:"from_weight" yaml:"from_weight"`
	FromResume int    `json:"from_resume" yaml:"from_resume"`

	// DPO专用配置
	Beta float64 `json:"beta" yaml:"beta"`

	// 高级配置
	WarmupSteps int     `json:"warmup_steps" yaml:"warmup_steps"`
	WeightDecay float64 `json:"weight_decay" yaml:"weight_decay"`
	AdamBeta1   float64 `json:"adam_beta1" yaml:"adam_beta1"`
	AdamBeta2   float64 `json:"adam_beta2" yaml:"adam_beta2"`
	AdamEpsilon float64 `json:"adam_epsilon" yaml:"adam_epsilon"`
	MaxGradNorm float64 `json:"max_grad_norm" yaml:"max_grad_norm"`
}

// DefaultTrainingConfig 返回默认训练配置
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		SaveDir:           "../out",
		SaveWeight:        "pretrain",
		Epochs:            1,
		BatchSize:         32,
		LearningRate:      5e-4,
		Device:            "cpu",
		NumWorkers:        4,
		AccumulationSteps: 8,
		GradClip:          1.0,
		LogInterval:       100,
		SaveInterval:      1000,
		HiddenSize:        512,
		NumHiddenLayers:   8,
		MaxSeqLen:         340,
		UseMoE:            false,
		NumAttentionHeads: 8,
		DataPath:          "../dataset/pretrain_hq.jsonl",
		FromWeight:        "none",
		FromResume:        0,
		Beta:              0.1,
		WarmupSteps:       100,
		WeightDecay:       0.01,
		AdamBeta1:         0.9,
		AdamBeta2:         0.999,
		AdamEpsilon:       1e-8,
		MaxGradNorm:       1.0,
	}
}

// TrainingState 训练状态
type TrainingState struct {
	Epoch        int       `json:"epoch"`
	Step         int       `json:"step"`
	GlobalStep   int       `json:"global_step"`
	Loss         float64   `json:"loss"`
	LearningRate float64   `json:"learning_rate"`
	Timestamp    time.Time `json:"timestamp"`
}

// TrainingStats 训练统计信息
type TrainingStats struct {
	StartTime       time.Time `json:"start_time"`
	EndTime         time.Time `json:"end_time"`
	TotalSteps      int       `json:"total_steps"`
	TotalEpochs     int       `json:"total_epochs"`
	AverageLoss     float64   `json:"average_loss"`
	BestLoss        float64   `json:"best_loss"`
	TotalTokens     int64     `json:"total_tokens"`
	TokensPerSecond float64   `json:"tokens_per_second"`
}

// CheckpointData 检查点数据
type CheckpointData struct {
	Epoch          int                    `json:"epoch"`
	Step           int                    `json:"step"`
	GlobalStep     int                    `json:"global_step"`
	ModelWeights   map[string]*mat.Dense  `json:"model_weights"`
	OptimizerState map[string]interface{} `json:"optimizer_state"`
	RandomState    interface{}            `json:"random_state"`
	TrainingConfig *TrainingConfig        `json:"training_config"`
	Timestamp      time.Time              `json:"timestamp"`
}

// BatchData 批次数据
type BatchData struct {
	InputIDs      [][]int `json:"input_ids"`
	Labels        [][]int `json:"labels"`
	AttentionMask [][]int `json:"attention_mask,omitempty"`
}

// DPOBatchData DPO专用批次数据
type DPOBatchData struct {
	XChosen      [][]int     `json:"x_chosen"`
	XRejected    [][]int     `json:"x_rejected"`
	YChosen      [][]int     `json:"y_chosen"`
	YRejected    [][]int     `json:"y_rejected"`
	MaskChosen   [][]float64 `json:"mask_chosen"`
	MaskRejected [][]float64 `json:"mask_rejected"`
}

// LossOutput 损失输出
type LossOutput struct {
	TotalLoss  float64 `json:"total_loss"`
	LogitsLoss float64 `json:"logits_loss"`
	AuxLoss    float64 `json:"aux_loss"`
	DPOLoss    float64 `json:"dpo_loss,omitempty"`
}

// Trainer 训练器接口
type Trainer interface {
	// Train 开始训练
	Train() error

	// TrainEpoch 训练一个epoch
	TrainEpoch(epoch int) error

	// SaveCheckpoint 保存检查点
	SaveCheckpoint(path string) error

	// LoadCheckpoint 加载检查点
	LoadCheckpoint(path string) error

	// GetStats 获取训练统计
	GetStats() *TrainingStats

	// Stop 停止训练
	Stop()
}

// TrainingCallback 训练回调函数类型
type TrainingCallback func(epoch, step int, loss float64, lr float64)

// ProgressCallback 进度回调函数类型
type ProgressCallback func(progress float64, message string)
