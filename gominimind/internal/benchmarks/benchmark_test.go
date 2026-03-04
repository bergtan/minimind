package benchmarks

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"minimind/pkg/model"
	"minimind/pkg/tokenizer"
	"minimind/pkg/types"
)

// BenchmarkModelInference - 基准测试模型推理性能
func BenchmarkModelInference(b *testing.B) {
	cfg := &types.ModelConfig{
		ModelType:         "minimind",
		VocabSize:         5000,
		ModelDim:          512,
		NumHeads:          8,
		NumLayers:         8,
		MaxSeqLength:      128,
		RoPEDim:           64,
		FFNMultiplier:     4,
		UseFlashAttention: true,
	}

	m, err := model.NewMiniMind(cfg)
	if err != nil {
		b.Fatal(err)
	}

	inputIDs := []int{1, 2, 3, 4, 5}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err = m.Generate(inputIDs, 10, 1.0)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkModelInferenceBatch - 批量推理基准测试
func BenchmarkModelInferenceBatch(b *testing.B) {
	cfg := &types.ModelConfig{
		ModelType:    "minimind",
		VocabSize:    5000,
		ModelDim:     512,
		NumHeads:     8,
		NumLayers:    8,
		MaxSeqLength: 128,
	}

	m, _ := model.NewMiniMind(cfg)

	batchSizes := []int{1, 4, 8, 16}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("BatchSize_%d", batchSize), func(b *testing.B) {
			inputs := make([][]int, batchSize)
			for i := 0; i < batchSize; i++ {
				inputs[i] = []int{1, 2, 3, 4, 5}
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				m.BatchGenerate(inputs, 10, 1.0)
			}
		})
	}
}

// BenchmarkModelDifferentSizes - 不同模型尺寸的性能测试
func BenchmarkModelDifferentSizes(b *testing.B) {
	sizes := []struct {
		name      string
		dim       int
		numLayers int
		numHeads  int
	}{
		{"Small", 256, 4, 4},
		{"Medium", 512, 8, 8},
		{"Large", 768, 12, 12},
		{"XLarge", 1024, 16, 16},
	}

	inputIDs := []int{1, 2, 3, 4, 5}

	for _, size := range sizes {
		cfg := &types.ModelConfig{
			VocabSize:    5000,
			ModelDim:     size.dim,
			NumHeads:     size.numHeads,
			NumLayers:    size.numLayers,
			MaxSeqLength: 128,
		}

		b.Run(size.name, func(b *testing.B) {
			m, err := model.NewMiniMind(cfg)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				m.Generate(inputIDs, 10, 1.0)
			}
		})
	}
}

// BenchmarkTokenizerEncode - Tokenizer编码性能测试
func BenchmarkTokenizerEncode(b *testing.B) {
	cfg := &tokenizer.Config{
		VocabSize:    50000,
		MaxSeqLength: 512,
		ModelDim:     512,
	}

	tok, _ := tokenizer.NewBPETokenizer(cfg)

	texts := []struct {
		name string
		text string
	}{
		{"Short", "Hello world"},
		{"Medium", "The quick brown fox jumps over the lazy dog"},
		{"Long", "In the heart of the bustling city, where skyscrapers pierce the clouds and streets pulse with endless energy, there exists a small cafe that has become a sanctuary for those seeking refuge from the urban chaos."},
	}

	for _, tc := range texts {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				tok.Encode(tc.text, true)
			}
		})
	}
}

// BenchmarkTokenizerDecode - Tokenizer解码性能测试
func BenchmarkTokenizerDecode(b *testing.B) {
	cfg := &tokenizer.Config{
		VocabSize:    50000,
		MaxSeqLength: 512,
		ModelDim:     512,
	}

	tok, _ := tokenizer.NewBPETokenizer(cfg)

	// Pre-encode tokens for decoding benchmark
	tokens := tok.Encode("The quick brown fox jumps over the lazy dog", true)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tok.Decode(tokens)
	}
}

// BenchmarkMemoryAllocation - 内存分配基准测试
func BenchmarkMemoryAllocation(b *testing.B) {
	b.Run("ModelCreate", func(b *testing.B) {
		cfg := &types.ModelConfig{
			ModelType:    "minimind",
			VocabSize:    5000,
			ModelDim:     512,
			NumHeads:     8,
			NumLayers:    8,
			MaxSeqLength: 128,
		}

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := model.NewMiniMind(cfg)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("InferenceAllocation", func(b *testing.B) {
		cfg := &types.ModelConfig{
			ModelType:    "minimind",
			VocabSize:    5000,
			ModelDim:     512,
			NumHeads:     8,
			NumLayers:    8,
			MaxSeqLength: 128,
		}

		m, _ := model.NewMiniMind(cfg)
		inputIDs := []int{1, 2, 3, 4, 5}

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			m.Generate(inputIDs, 10, 1.0)
		}
	})
}

// BenchmarkMatrixOperations - 矩阵运算性能测试
func BenchmarkMatrixOperations(b *testing.B) {
	sizes := []int{64, 128, 256, 512, 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Multiply_%dx%d", size, size), func(b *testing.B) {
			a := createRandomMatrix(size, size)
			c := createRandomMatrix(size, size)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				matrixMultiply(a, c, size)
			}
		})
	}
}

// BenchmarkSoftmax - Softmax计算性能测试
func BenchmarkSoftmax(b *testing.B) {
	sizes := []int{64, 128, 256, 512, 1024, 2048}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			input := make([]float32, size)
			for i := range input {
				input[i] = rand.Float32() * 10
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				softmax(input)
			}
		})
	}
}

// BenchmarkEmbeddingLookup - 嵌入查找性能测试
func BenchmarkEmbeddingLookup(b *testing.B) {
	tokenCounts := []int{1000, 5000, 10000, 50000}
	dim := 512

	for _, vocabSize := range tokenCounts {
		b.Run(fmt.Sprintf("Vocab_%d", vocabSize), func(b *testing.B) {
			embeddings := createRandomMatrix(vocabSize, dim)
			inputIDs := []int{1, 2, 3, 4, 5}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				embeddingLookup(embeddings, inputIDs, dim)
			}
		})
	}
}

// BenchmarkJSONSerialization - JSON序列化性能测试
func BenchmarkJSONSerialization(b *testing.B) {
	response := types.ChatCompletionResponse{
		ID:      "benchmark-id",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "minimind",
		Choices: []types.ChatCompletionChoice{
			{
				Index: 0,
				Message: types.ChatMessage{
					Role:    "assistant",
					Content: "This is a benchmark response for testing JSON serialization performance. It contains some text to make the payload larger and more realistic. The quick brown fox jumps over the lazy dog.",
				},
				FinishReason: "stop",
			},
		},
		Usage: types.ChatCompletionUsage{
			PromptTokens:     20,
			CompletionTokens: 30,
			TotalTokens:      50,
		},
	}

	b.Run("Marshal", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := json.Marshal(response)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Unmarshal", func(b *testing.B) {
		data, _ := json.Marshal(response)

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			var resp types.ChatCompletionResponse
			err := json.Unmarshal(data, &resp)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkConcurrentRequests - 并发请求性能测试
func BenchmarkConcurrentRequests(b *testing.B) {
	concurrencyLevels := []int{1, 10, 50, 100, 200}

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrency_%d", concurrency), func(b *testing.B) {
			cfg := &types.ModelConfig{
				ModelType:    "minimind",
				VocabSize:    5000,
				ModelDim:     512,
				NumHeads:     8,
				NumLayers:    8,
				MaxSeqLength: 128,
			}

			m, _ := model.NewMiniMind(cfg)
			inputIDs := []int{1, 2, 3, 4, 5}

			sem := make(chan struct{}, concurrency)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				sem <- struct{}{}
				go func() {
					defer func() { <-sem }()
					m.Generate(inputIDs, 10, 1.0)
				}()
			}
		})
	}
}

// BenchmarkMemoryUsage - 内存使用基准测试
func BenchmarkMemoryUsage(b *testing.B) {
	b.Run("ModelMemory", func(b *testing.B) {
		cfg := &types.ModelConfig{
			ModelType:    "minimind",
			VocabSize:    5000,
			ModelDim:     512,
			NumHeads:     8,
			NumLayers:    8,
			MaxSeqLength: 128,
		}

		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		beforeAlloc := m.Alloc

		model, _ := model.NewMiniMind(cfg)

		runtime.ReadMemStats(&m)
		afterAlloc := m.Alloc

		modelBytes := afterAlloc - beforeAlloc
		b.ReportMetric(float64(modelBytes)/1024/1024, "MB")

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			// Simulate model usage
			_ = model
		}
	})
}

// Helper functions

func createRandomMatrix(rows, cols int) [][]float32 {
	matrix := make([][]float32, rows)
	for i := range matrix {
		matrix[i] = make([]float32, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float32()
		}
	}
	return matrix
}

func matrixMultiply(a, c [][]float32, size int) [][]float32 {
	result := make([][]float32, size)
	for i := range result {
		result[i] = make([]float32, size)
		for j := 0; j < size; j++ {
			sum := float32(0)
			for k := 0; k < size; k++ {
				sum += a[i][k] * c[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

func softmax(input []float32) []float32 {
	maxVal := input[0]
	for _, v := range input[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	expSum := float32(0)
	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = float32(math.Exp(float64(v - maxVal)))
		expSum += output[i]
	}

	for i := range output {
		output[i] /= expSum
	}

	return output
}

func embeddingLookup(embeddings [][]float32, inputIDs []int, dim int) [][]float32 {
	result := make([][]float32, len(inputIDs))
	for i, id := range inputIDs {
		if id < len(embeddings) {
			result[i] = make([]float32, dim)
			copy(result[i], embeddings[id])
		}
	}
	return result
}

// BenchmarkReport - 生成性能报告
func BenchmarkReport(b *testing.B) {
	if !b.ReportAllocs {
		b.Log("Warning: ReportAllocs not enabled")
	}

	b.Logf("Running benchmark on %s architecture", runtime.GOARCH)
	b.Logf("Go version: %s", runtime.Version())
	b.Logf("NumCPU: %d", runtime.NumCPU())

	// Run memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	b.Logf("TotalAlloc: %d MB", m.TotalAlloc/1024/1024)
	b.Logf("Sys: %d MB", m.Sys/1024/1024)
	b.Logf("NumGC: %d", m.NumGC)
}

// Example benchmark showing typical usage
func ExampleBenchmark() {
	// This is an example showing how to run benchmarks
	// Run with: go test -bench=. -benchmem ./internal/benchmarks/
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
