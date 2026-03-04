package config

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/jingyaogong/gominimind/pkg/types"
	"gopkg.in/yaml.v3"
)

// Config 全局配置结构
type Config struct {
	Server   ServerConfig   `yaml:"server" json:"server"`
	Model    ModelConfig    `yaml:"model" json:"model"`
	Training TrainingConfig `yaml:"training" json:"training"`
	Dataset  DatasetConfig  `yaml:"dataset" json:"dataset"`
	Logging  LoggingConfig  `yaml:"logging" json:"logging"`
	Cache    CacheConfig    `yaml:"cache" json:"cache"`
}

// ServerConfig 服务器配置
type ServerConfig struct {
	Host        string   `yaml:"host" json:"host"`
	Port        int      `yaml:"port" json:"port"`
	APIKey      string   `yaml:"api_key" json:"api_key"`
	MaxTokens   int      `yaml:"max_tokens" json:"max_tokens"`
	Temperature float64  `yaml:"temperature" json:"temperature"`
	TopP        float64  `yaml:"top_p" json:"top_p"`
	Workers     int      `yaml:"workers" json:"workers"`
	LogLevel    string   `yaml:"log_level" json:"log_level"`
	UseCache    bool     `yaml:"use_cache" json:"use_cache"`
	CacheSize   int      `yaml:"cache_size" json:"cache_size"`
	RateLimit   int      `yaml:"rate_limit" json:"rate_limit"`
	AllowedIPs  []string `yaml:"allowed_ips" json:"allowed_ips"`
	SSLEnabled  bool     `yaml:"ssl_enabled" json:"ssl_enabled"`
	SSLCertFile string   `yaml:"ssl_cert_file" json:"ssl_cert_file"`
	SSLKeyFile  string   `yaml:"ssl_key_file" json:"ssl_key_file"`
}

// ModelConfig 模型配置
type ModelConfig struct {
	Name                  string  `yaml:"name" json:"name"`
	Path                  string  `yaml:"path" json:"path"`
	VocabSize             int     `yaml:"vocab_size" json:"vocab_size"`
	HiddenSize            int     `yaml:"hidden_size" json:"hidden_size"`
	NumHiddenLayers       int     `yaml:"num_hidden_layers" json:"num_hidden_layers"`
	NumAttentionHeads     int     `yaml:"num_attention_heads" json:"num_attention_heads"`
	MaxPositionEmbeddings int     `yaml:"max_position_embeddings" json:"max_position_embeddings"`
	UseFlashAttention     bool    `yaml:"use_flash_attention" json:"use_flash_attention"`
	RopeTheta             float64 `yaml:"rope_theta" json:"rope_theta"`
	HiddenAct             string  `yaml:"hidden_act" json:"hidden_act"`
	IntermediateSize      int     `yaml:"intermediate_size" json:"intermediate_size"`
	LayerNormEps          float64 `yaml:"layer_norm_eps" json:"layer_norm_eps"`
	InitializerRange      float64 `yaml:"initializer_range" json:"initializer_range"`
	Precision             string  `yaml:"precision" json:"precision"`
	UseQuantization       bool    `yaml:"use_quantization" json:"use_quantization"`
	QuantizationBits      int     `yaml:"quantization_bits" json:"quantization_bits"`
}

// TrainingConfig 训练配置
type TrainingConfig struct {
	Epochs                    int     `yaml:"epochs" json:"epochs"`
	BatchSize                 int     `yaml:"batch_size" json:"batch_size"`
	LearningRate              float64 `yaml:"learning_rate" json:"learning_rate"`
	GradientAccumulationSteps int     `yaml:"gradient_accumulation_steps" json:"gradient_accumulation_steps"`
	MaxSeqLen                 int     `yaml:"max_seq_len" json:"max_seq_len"`
	UseAmp                    bool    `yaml:"use_amp" json:"use_amp"`
	AmpDtype                  string  `yaml:"amp_dtype" json:"amp_dtype"`
	GradientCheckpointing     bool    `yaml:"gradient_checkpointing" json:"gradient_checkpointing"`
	SaveSteps                 int     `yaml:"save_steps" json:"save_steps"`
	EvalSteps                 int     `yaml:"eval_steps" json:"eval_steps"`
	LogSteps                  int     `yaml:"log_steps" json:"log_steps"`
	WarmupSteps               int     `yaml:"warmup_steps" json:"warmup_steps"`
	WeightDecay               float64 `yaml:"weight_decay" json:"weight_decay"`
	MaxGradNorm               float64 `yaml:"max_grad_norm" json:"max_grad_norm"`
	UseDDP                    bool    `yaml:"use_ddp" json:"use_ddp"`
	DDPBackend                string  `yaml:"ddp_backend" json:"ddp_backend"`
	UseDeepspeed              bool    `yaml:"use_deepspeed" json:"use_deepspeed"`
	DeepspeedConfig           string  `yaml:"deepspeed_config" json:"deepspeed_config"`
}

// DatasetConfig 数据集配置
type DatasetConfig struct {
	DataPath       string  `yaml:"data_path" json:"data_path"`
	MaxLength      int     `yaml:"max_length" json:"max_length"`
	Shuffle        bool    `yaml:"shuffle" json:"shuffle"`
	NumWorkers     int     `yaml:"num_workers" json:"num_workers"`
	PrefetchFactor int     `yaml:"prefetch_factor" json:"prefetch_factor"`
	PinMemory      bool    `yaml:"pin_memory" json:"pin_memory"`
	CacheDir       string  `yaml:"cache_dir" json:"cache_dir"`
	SplitRatio     float64 `yaml:"split_ratio" json:"split_ratio"`
	Seed           int64   `yaml:"seed" json:"seed"`
}

// LoggingConfig 日志配置
type LoggingConfig struct {
	Level      string `yaml:"level" json:"level"`
	Format     string `yaml:"format" json:"format"`
	Output     string `yaml:"output" json:"output"`
	MaxSize    int    `yaml:"max_size" json:"max_size"`
	MaxBackups int    `yaml:"max_backups" json:"max_backups"`
	MaxAge     int    `yaml:"max_age" json:"max_age"`
	Compress   bool   `yaml:"compress" json:"compress"`
}

// CacheConfig 缓存配置
type CacheConfig struct {
	Enabled       bool   `yaml:"enabled" json:"enabled"`
	Type          string `yaml:"type" json:"type"`
	Size          int    `yaml:"size" json:"size"`
	Expiration    int    `yaml:"expiration" json:"expiration"`
	RedisHost     string `yaml:"redis_host" json:"redis_host"`
	RedisPort     int    `yaml:"redis_port" json:"redis_port"`
	RedisDB       int    `yaml:"redis_db" json:"redis_db"`
	RedisPassword string `yaml:"redis_password" json:"redis_password"`
}

// DefaultConfig 默认配置
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:        "0.0.0.0",
			Port:        8000,
			APIKey:      "",
			MaxTokens:   2048,
			Temperature: 0.7,
			TopP:        0.9,
			Workers:     4,
			LogLevel:    "info",
			UseCache:    true,
			CacheSize:   1000,
			RateLimit:   100,
			AllowedIPs:  []string{"0.0.0.0/0"},
			SSLEnabled:  false,
		},
		Model: ModelConfig{
			Name:                  "minimind",
			Path:                  "./MiniMind2",
			VocabSize:             6400,
			HiddenSize:            512,
			NumHiddenLayers:       8,
			NumAttentionHeads:     8,
			MaxPositionEmbeddings: 32768,
			UseFlashAttention:     true,
			RopeTheta:             10000.0,
			HiddenAct:             "swiglu",
			IntermediateSize:      2048,
			LayerNormEps:          1e-5,
			InitializerRange:      0.02,
			Precision:             "float32",
			UseQuantization:       false,
			QuantizationBits:      8,
		},
		Training: TrainingConfig{
			Epochs:                    1,
			BatchSize:                 32,
			LearningRate:              5e-4,
			GradientAccumulationSteps: 1,
			MaxSeqLen:                 340,
			UseAmp:                    true,
			AmpDtype:                  "float16",
			GradientCheckpointing:     false,
			SaveSteps:                 1000,
			EvalSteps:                 500,
			LogSteps:                  100,
			WarmupSteps:               100,
			WeightDecay:               0.01,
			MaxGradNorm:               1.0,
			UseDDP:                    false,
			DDPBackend:                "nccl",
			UseDeepspeed:              false,
		},
		Dataset: DatasetConfig{
			DataPath:       "./dataset",
			MaxLength:      512,
			Shuffle:        true,
			NumWorkers:     4,
			PrefetchFactor: 2,
			PinMemory:      true,
			CacheDir:       "./cache",
			SplitRatio:     0.9,
			Seed:           42,
		},
		Logging: LoggingConfig{
			Level:      "info",
			Format:     "json",
			Output:     "stdout",
			MaxSize:    100,
			MaxBackups: 10,
			MaxAge:     30,
			Compress:   true,
		},
		Cache: CacheConfig{
			Enabled:       true,
			Type:          "memory",
			Size:          1000,
			Expiration:    300,
			RedisHost:     "localhost",
			RedisPort:     6379,
			RedisDB:       0,
			RedisPassword: "",
		},
	}
}

// LoadConfig 从文件加载配置
func LoadConfig(path string) (*Config, error) {
	if path == "" {
		return DefaultConfig(), nil
	}

	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	config := DefaultConfig()
	ext := filepath.Ext(path)

	switch strings.ToLower(ext) {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, config); err != nil {
			return nil, fmt.Errorf("failed to parse YAML config: %v", err)
		}
	case ".json":
		if err := json.Unmarshal(data, config); err != nil {
			return nil, fmt.Errorf("failed to parse JSON config: %v", err)
		}
	default:
		return nil, fmt.Errorf("unsupported config file format: %s", ext)
	}

	return config, nil
}

// SaveConfig 保存配置到文件
func SaveConfig(config *Config, path string) error {
	var data []byte
	var err error

	ext := filepath.Ext(path)
	switch strings.ToLower(ext) {
	case ".yaml", ".yml":
		data, err = yaml.Marshal(config)
	case ".json":
		data, err = json.MarshalIndent(config, "", "  ")
	default:
		return fmt.Errorf("unsupported config file format: %s", ext)
	}

	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	if err := ioutil.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %v", err)
	}

	return nil
}

// MergeEnvVars 合并环境变量到配置
func MergeEnvVars(config *Config) {
	// 服务器配置
	if host := os.Getenv("MINIMIND_HOST"); host != "" {
		config.Server.Host = host
	}
	if port := os.Getenv("MINIMIND_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			config.Server.Port = p
		}
	}
	if apiKey := os.Getenv("MINIMIND_API_KEY"); apiKey != "" {
		config.Server.APIKey = apiKey
	}

	// 模型配置
	if modelPath := os.Getenv("MINIMIND_MODEL_PATH"); modelPath != "" {
		config.Model.Path = modelPath
	}
	if precision := os.Getenv("MINIMIND_PRECISION"); precision != "" {
		config.Model.Precision = precision
	}

	// 训练配置
	if batchSize := os.Getenv("MINIMIND_BATCH_SIZE"); batchSize != "" {
		if bs, err := strconv.Atoi(batchSize); err == nil {
			config.Training.BatchSize = bs
		}
	}
	if learningRate := os.Getenv("MINIMIND_LEARNING_RATE"); learningRate != "" {
		if lr, err := strconv.ParseFloat(learningRate, 64); err == nil {
			config.Training.LearningRate = lr
		}
	}
}

// ValidateConfig 验证配置有效性
func ValidateConfig(config *Config) error {
	// 验证服务器配置
	if config.Server.Port < 1 || config.Server.Port > 65535 {
		return fmt.Errorf("invalid port number: %d", config.Server.Port)
	}
	if config.Server.MaxTokens < 1 {
		return fmt.Errorf("max_tokens must be positive: %d", config.Server.MaxTokens)
	}
	if config.Server.Temperature < 0 || config.Server.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2: %f", config.Server.Temperature)
	}
	if config.Server.TopP < 0 || config.Server.TopP > 1 {
		return fmt.Errorf("top_p must be between 0 and 1: %f", config.Server.TopP)
	}

	// 验证模型配置
	if config.Model.VocabSize < 1 {
		return fmt.Errorf("vocab_size must be positive: %d", config.Model.VocabSize)
	}
	if config.Model.HiddenSize < 1 {
		return fmt.Errorf("hidden_size must be positive: %d", config.Model.HiddenSize)
	}
	if config.Model.NumHiddenLayers < 1 {
		return fmt.Errorf("num_hidden_layers must be positive: %d", config.Model.NumHiddenLayers)
	}
	if config.Model.NumAttentionHeads < 1 {
		return fmt.Errorf("num_attention_heads must be positive: %d", config.Model.NumAttentionHeads)
	}
	if config.Model.HiddenSize%config.Model.NumAttentionHeads != 0 {
		return fmt.Errorf("hidden_size must be divisible by num_attention_heads")
	}

	// 验证训练配置
	if config.Training.Epochs < 1 {
		return fmt.Errorf("epochs must be positive: %d", config.Training.Epochs)
	}
	if config.Training.BatchSize < 1 {
		return fmt.Errorf("batch_size must be positive: %d", config.Training.BatchSize)
	}
	if config.Training.LearningRate <= 0 {
		return fmt.Errorf("learning_rate must be positive: %f", config.Training.LearningRate)
	}

	return nil
}

// ToTypesConfig 转换为类型包中的配置
func (c *Config) ToTypesConfig() *types.MiniMindConfig {
	return &types.MiniMindConfig{
		VocabSize:             c.Model.VocabSize,
		HiddenSize:            c.Model.HiddenSize,
		NumHiddenLayers:       c.Model.NumHiddenLayers,
		NumAttentionHeads:     c.Model.NumAttentionHeads,
		MaxPositionEmbeddings: c.Model.MaxPositionEmbeddings,
		UseFlashAttention:     c.Model.UseFlashAttention,
		RopeTheta:             c.Model.RopeTheta,
		HiddenAct:             c.Model.HiddenAct,
		IntermediateSize:      c.Model.IntermediateSize,
		LayerNormEps:          c.Model.LayerNormEps,
		InitializerRange:      c.Model.InitializerRange,
	}
}

// GetModelPath 获取模型路径
func (c *Config) GetModelPath() string {
	if c.Model.Path != "" {
		return c.Model.Path
	}
	return "./MiniMind2"
}

// GetDataPath 获取数据路径
func (c *Config) GetDataPath() string {
	if c.Dataset.DataPath != "" {
		return c.Dataset.DataPath
	}
	return "./dataset"
}

// GetCacheDir 获取缓存目录
func (c *Config) GetCacheDir() string {
	if c.Dataset.CacheDir != "" {
		return c.Dataset.CacheDir
	}
	return "./cache"
}
