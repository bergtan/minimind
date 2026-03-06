package model

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gominimind/pkg/types"
)

// createTestModelConfig 创建测试用的模型配置
func createTestModelConfig() *types.ModelConfig {
	return &types.ModelConfig{
		Name:                  "test-minimind",
		ModelType:             "minimind",
		VocabSize:             6400,
		HiddenSize:            512,
		NumLayers:             8,
		NumHeads:              8,
		MaxPositionEmbeddings: 32768,
		IntermediateSize:      2048,
		HiddenAct:             "swiglu",
		InitializerRange:      0.02,
		LayerNormEps:          1e-5,
		UseCache:              true,
		RopeTheta:             10000.0,
	}
}

// TestMiniMindConfigValidation 测试配置验证
func TestMiniMindConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      *types.ModelConfig
		shouldError bool
	}{
		{
			name:        "Valid configuration",
			config:      createTestModelConfig(),
			shouldError: false,
		},
		{
			name: "Minimal valid config",
			config: &types.ModelConfig{
				VocabSize:  100,
				HiddenSize: 64,
				NumLayers:  2,
				NumHeads:   2,
			},
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := NewMiniMindModel(tt.config)
			if tt.shouldError {
				assert.Error(t, err)
				assert.Nil(t, model)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, model)
			}
		})
	}
}

// TestMiniMindInitialization 测试模型初始化
func TestMiniMindInitialization(t *testing.T) {
	config := createTestModelConfig()

	model, err := NewMiniMindModel(config)
	require.NoError(t, err)
	require.NotNil(t, model)

	// 验证模型状态
	assert.Equal(t, ModelStatusInitialized, model.status)
	assert.NotNil(t, model.kvCache)
}

// TestMiniMindGenerate 测试模型生成
func TestMiniMindGenerate(t *testing.T) {
	config := &types.ModelConfig{
		VocabSize:             1000,
		HiddenSize:            64,
		NumLayers:             2,
		NumHeads:              2,
		MaxPositionEmbeddings: 128,
	}

	model, err := NewMiniMindModel(config)
	require.NoError(t, err)

	t.Run("Generate with simple input", func(t *testing.T) {
		output, err := model.Generate("hello world", 10, 1.0, 0.9)
		// 模型未加载权重，应返回错误
		assert.Error(t, err)
		assert.Empty(t, output)
	})
}

// TestMiniMindModelInfo 测试模型信息
func TestMiniMindModelInfo(t *testing.T) {
	config := createTestModelConfig()

	model, err := NewMiniMindModel(config)
	require.NoError(t, err)

	info := model.GetModelInfo()
	assert.NotNil(t, info)
}
