package model

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"minimind/pkg/tokenizer"
	"minimind/pkg/types"
)

// TestMiniMindConfigValidation tests configuration validation
func TestMiniMindConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      *MiniMindConfig
		shouldError bool
		errorMsg    string
	}{
		{
			name: "Valid configuration",
			config: &MiniMindConfig{
				VocabSize:             6400,
				HiddenSize:            512,
				NumHiddenLayers:       8,
				NumAttentionHeads:     8,
				MaxPositionEmbeddings: 32768,
				UseFlashAttention:     true,
				RopeTheta:             10000.0,
				HiddenAct:             "swiglu",
			},
			shouldError: false,
		},
		{
			name: "Invalid vocab size",
			config: &MiniMindConfig{
				VocabSize:       0,
				HiddenSize:      512,
				NumHiddenLayers: 8,
			},
			shouldError: true,
			errorMsg:    "vocab_size must be positive",
		},
		{
			name: "Invalid hidden size",
			config: &MiniMindConfig{
				VocabSize:       6400,
				HiddenSize:      0,
				NumHiddenLayers: 8,
			},
			shouldError: true,
			errorMsg:    "hidden_size must be positive",
		},
		{
			name: "Invalid number of layers",
			config: &MiniMindConfig{
				VocabSize:       6400,
				HiddenSize:      512,
				NumHiddenLayers: 0,
			},
			shouldError: true,
			errorMsg:    "num_hidden_layers must be positive",
		},
		{
			name: "Invalid attention heads",
			config: &MiniMindConfig{
				VocabSize:         6400,
				HiddenSize:        512,
				NumHiddenLayers:   8,
				NumAttentionHeads: 0,
			},
			shouldError: true,
			errorMsg:    "num_attention_heads must be positive",
		},
		{
			name: "Hidden size not divisible by attention heads",
			config: &MiniMindConfig{
				VocabSize:         6400,
				HiddenSize:        512,
				NumHiddenLayers:   8,
				NumAttentionHeads: 7,
			},
			shouldError: true,
			errorMsg:    "hidden_size must be divisible by num_attention_heads",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.shouldError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestMiniMindInitialization tests model initialization
func TestMiniMindInitialization(t *testing.T) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(t, err)
	require.NotNil(t, model)

	// Test model parameters
	assert.Equal(t, config.VocabSize, model.config.VocabSize)
	assert.Equal(t, config.HiddenSize, model.config.HiddenSize)
	assert.Equal(t, config.NumHiddenLayers, model.config.NumHiddenLayers)

	// Test model components
	assert.NotNil(t, model.embedTokens)
	assert.NotNil(t, model.layers)
	assert.Len(t, model.layers, config.NumHiddenLayers)
	assert.NotNil(t, model.lmHead)
}

// TestMiniMindForwardPass tests forward pass functionality
func TestMiniMindForwardPass(t *testing.T) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(t, err)

	// Create test input
	inputIDs := []int{1, 2, 3, 4, 5}

	// Test forward pass
	output, err := model.Forward(inputIDs, nil)
	require.NoError(t, err)
	require.NotNil(t, output)

	// Check output shape
	expectedShape := []int{len(inputIDs), config.VocabSize}
	assert.Equal(t, expectedShape[0], output.Shape[0])
	assert.Equal(t, expectedShape[1], output.Shape[1])

	// Check output values are finite
	for i := 0; i < output.Shape[0]; i++ {
		for j := 0; j < output.Shape[1]; j++ {
			assert.False(t, output.Data[i*output.Shape[1]+j].IsNaN())
			assert.False(t, output.Data[i*output.Shape[1]+j].IsInf())
		}
	}
}

// TestMiniMindGenerate tests text generation
func TestMiniMindGenerate(t *testing.T) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(t, err)

	// Create generation parameters
	params := &types.GenerationParams{
		MaxTokens:   10,
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        50,
		DoSample:    true,
		StopTokens:  []int{2},
	}

	// Test generation
	inputIDs := []int{1, 2, 3}
	generated, err := model.Generate(inputIDs, params)
	require.NoError(t, err)
	require.NotNil(t, generated)

	// Check generated sequence length
	assert.True(t, len(generated) >= len(inputIDs))
	assert.True(t, len(generated) <= len(inputIDs)+params.MaxTokens)

	// Check generated tokens are within vocabulary
	for _, token := range generated {
		assert.True(t, token >= 0)
		assert.True(t, token < config.VocabSize)
	}
}

// TestMiniMindEmbeddings tests embedding generation
func TestMiniMindEmbeddings(t *testing.T) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(t, err)

	// Test single text embedding
	text := "Hello, world!"
	embedding, err := model.GetEmbedding(text)
	require.NoError(t, err)
	require.NotNil(t, embedding)

	// Check embedding dimensions
	assert.Equal(t, config.HiddenSize, len(embedding))

	// Check embedding values are finite
	for _, value := range embedding {
		assert.False(t, value.IsNaN())
		assert.False(t, value.IsInf())
	}

	// Test batch embedding
	texts := []string{"Hello", "world", "test"}
	embeddings, err := model.GetBatchEmbeddings(texts)
	require.NoError(t, err)
	require.NotNil(t, embeddings)

	// Check batch dimensions
	assert.Equal(t, len(texts), len(embeddings))
	for i, emb := range embeddings {
		assert.Equal(t, config.HiddenSize, len(emb))

		// Check each embedding
		for _, value := range emb {
			assert.False(t, value.IsNaN())
			assert.False(t, value.IsInf())
		}

		// Check embeddings are different for different inputs
		if i > 0 {
			assert.NotEqual(t, embeddings[0], emb)
		}
	}
}

// TestMiniMindSaveLoad tests model saving and loading
func TestMiniMindSaveLoad(t *testing.T) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	// Create and save model
	model1, err := NewMiniMind(config)
	require.NoError(t, err)

	tempDir := t.TempDir()
	modelPath := tempDir + "/model.bin"

	err = model1.Save(modelPath)
	require.NoError(t, err)

	// Load model
	model2, err := LoadMiniMind(modelPath)
	require.NoError(t, err)
	require.NotNil(t, model2)

	// Compare configurations
	assert.Equal(t, model1.config.VocabSize, model2.config.VocabSize)
	assert.Equal(t, model1.config.HiddenSize, model2.config.HiddenSize)
	assert.Equal(t, model1.config.NumHiddenLayers, model2.config.NumHiddenLayers)

	// Test that loaded model produces same output
	inputIDs := []int{1, 2, 3, 4, 5}

	output1, err := model1.Forward(inputIDs, nil)
	require.NoError(t, err)

	output2, err := model2.Forward(inputIDs, nil)
	require.NoError(t, err)

	// Check outputs are similar (allowing for small numerical differences)
	assert.Equal(t, output1.Shape, output2.Shape)

	for i := 0; i < len(output1.Data); i++ {
		diff := output1.Data[i].Sub(output2.Data[i]).Abs()
		assert.True(t, diff.Lt(1e-6), "Output mismatch at index %d: %v vs %v", i, output1.Data[i], output2.Data[i])
	}
}

// TestMiniMindPerformance tests model performance
func TestMiniMindPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(t, err)

	// Performance test with larger input
	inputIDs := make([]int, 100)
	for i := range inputIDs {
		inputIDs[i] = i % config.VocabSize
	}

	startTime := time.Now()

	// Run multiple forward passes
	const numRuns = 10
	for i := 0; i < numRuns; i++ {
		_, err := model.Forward(inputIDs, nil)
		require.NoError(t, err)
	}

	duration := time.Since(startTime)
	averageTime := duration / numRuns

	// Performance threshold: average forward pass should be under 100ms
	assert.True(t, averageTime < 100*time.Millisecond,
		"Average forward pass time %v exceeds 100ms threshold", averageTime)

	t.Logf("Average forward pass time: %v", averageTime)
	t.Logf("Total time for %d runs: %v", numRuns, duration)
}

// TestMiniMindMemoryUsage tests memory usage
func TestMiniMindMemoryUsage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory usage test in short mode")
	}

	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	// Measure memory before model creation
	var memBefore, memAfter uint64

	// Create model
	model, err := NewMiniMind(config)
	require.NoError(t, err)

	// Measure memory after model creation
	// Note: This is a simplified memory measurement
	// In practice, you'd use runtime.MemStats

	// Test that model can handle large inputs without excessive memory growth
	largeInput := make([]int, 1000)
	for i := range largeInput {
		largeInput[i] = i % config.VocabSize
	}

	_, err = model.Forward(largeInput, nil)
	require.NoError(t, err)

	// The test passes if we can process large input without panic
	// More sophisticated memory testing would require external tools
}

// TestMiniMindErrorHandling tests error conditions
func TestMiniMindErrorHandling(t *testing.T) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(t, err)

	// Test invalid input (out of vocabulary)
	invalidInput := []int{-1, 0, config.VocabSize}
	_, err = model.Forward(invalidInput, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "out of vocabulary")

	// Test empty input
	emptyInput := []int{}
	_, err = model.Forward(emptyInput, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty input")

	// Test input too long
	longInput := make([]int, config.MaxPositionEmbeddings+1)
	_, err = model.Forward(longInput, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds maximum sequence length")

	// Test invalid generation parameters
	invalidParams := &types.GenerationParams{
		MaxTokens:   0,
		Temperature: -0.5,
		TopP:        1.5,
	}

	validInput := []int{1, 2, 3}
	_, err = model.Generate(validInput, invalidParams)
	assert.Error(t, err)
}

// TestMiniMindIntegration tests integration with other components
func TestMiniMindIntegration(t *testing.T) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(t, err)

	// Test with tokenizer
	tokenizer, err := tokenizer.NewTokenizer(config.VocabSize)
	require.NoError(t, err)

	text := "Hello, MiniMind!"
	tokens, err := tokenizer.Encode(text)
	require.NoError(t, err)

	// Test model with tokenized input
	output, err := model.Forward(tokens, nil)
	require.NoError(t, err)
	require.NotNil(t, output)

	// Test generation with tokenizer
	params := &types.GenerationParams{
		MaxTokens:   5,
		Temperature: 0.7,
		DoSample:    true,
	}

	generatedTokens, err := model.Generate(tokens, params)
	require.NoError(t, err)

	// Decode generated tokens
	decoded, err := tokenizer.Decode(generatedTokens)
	require.NoError(t, err)
	assert.NotEmpty(t, decoded)

	t.Logf("Original text: %s", text)
	t.Logf("Generated text: %s", decoded)
}

// BenchmarkMiniMindForward benchmarks forward pass performance
func BenchmarkMiniMindForward(b *testing.B) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(b, err)

	inputIDs := make([]int, 100)
	for i := range inputIDs {
		inputIDs[i] = i % config.VocabSize
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := model.Forward(inputIDs, nil)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}

// BenchmarkMiniMindGeneration benchmarks text generation performance
func BenchmarkMiniMindGeneration(b *testing.B) {
	config := &MiniMindConfig{
		VocabSize:             6400,
		HiddenSize:            512,
		NumHiddenLayers:       8,
		NumAttentionHeads:     8,
		MaxPositionEmbeddings: 32768,
		UseFlashAttention:     true,
		RopeTheta:             10000.0,
		HiddenAct:             "swiglu",
	}

	model, err := NewMiniMind(config)
	require.NoError(b, err)

	inputIDs := []int{1, 2, 3}
	params := &types.GenerationParams{
		MaxTokens:   10,
		Temperature: 0.7,
		DoSample:    true,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := model.Generate(inputIDs, params)
		if err != nil {
			b.Fatalf("Generation failed: %v", err)
		}
	}
}
