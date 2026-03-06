package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"gominimind/pkg/model"
	"gominimind/pkg/tokenizer"
	"gominimind/pkg/types"
)

// BenchmarkModelInference - 基准测试模型推理性能
func BenchmarkModelInference(b *testing.B) {
	cfg := &types.ModelConfig{
		ModelType:             "minimind",
		VocabSize:             5000,
		HiddenSize:            512,
		NumHeads:              8,
		NumLayers:             8,
		MaxPositionEmbeddings: 128,
	}

	m, err := model.NewMiniMindModel(cfg)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err = m.Generate("hello world", 10, 1.0, 0.9)
		if err != nil {
			// 预期可能失败（模型未加载权重），不中断基准测试
			break
		}
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
	}

	for _, size := range sizes {
		cfg := &types.ModelConfig{
			VocabSize:             5000,
			HiddenSize:            size.dim,
			NumHeads:              size.numHeads,
			NumLayers:             size.numLayers,
			MaxPositionEmbeddings: 128,
		}

		b.Run(size.name, func(b *testing.B) {
			m, err := model.NewMiniMindModel(cfg)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, _ = m.Generate("test input", 10, 1.0, 0.9)
			}
		})
	}
}

// BenchmarkTokenizer - 分词器性能基准测试
func BenchmarkTokenizer(b *testing.B) {
	tok, err := tokenizer.CreateDefaultTokenizer()
	if err != nil {
		b.Fatal(err)
	}

	texts := []string{
		"Hello world, this is a simple test.",
		"The quick brown fox jumps over the lazy dog.",
		"你好世界，这是一个测试。",
		"Machine learning is a subset of artificial intelligence.",
	}

	b.Run("Encode", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			text := texts[i%len(texts)]
			_, _ = tok.Encode(text)
		}
	})

	b.Run("EncodeSimple", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			text := texts[i%len(texts)]
			_ = tok.EncodeSimple(text)
		}
	})

	b.Run("EncodeBatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = tok.EncodeBatch(texts)
		}
	})
}

// BenchmarkMemoryUsage - 内存使用基准测试
func BenchmarkMemoryUsage(b *testing.B) {
	var memBefore, memAfter runtime.MemStats

	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	cfg := &types.ModelConfig{
		VocabSize:             5000,
		HiddenSize:            512,
		NumHeads:              8,
		NumLayers:             8,
		MaxPositionEmbeddings: 128,
	}

	_, err := model.NewMiniMindModel(cfg)
	if err != nil {
		b.Fatal(err)
	}

	runtime.GC()
	runtime.ReadMemStats(&memAfter)

	allocBytes := memAfter.TotalAlloc - memBefore.TotalAlloc
	b.ReportMetric(float64(allocBytes), "bytes/model")
}

// BenchmarkRandomDataGeneration - 随机数据生成基准
func BenchmarkRandomDataGeneration(b *testing.B) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	vocabSize := 5000

	b.Run("GenerateTokenIDs", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tokens := make([]int, 128)
			for j := range tokens {
				tokens[j] = rng.Intn(vocabSize)
			}
		}
	})
}

// BenchmarkModelCreation - 模型创建性能基准
func BenchmarkModelCreation(b *testing.B) {
	configs := []struct {
		name string
		cfg  *types.ModelConfig
	}{
		{
			name: "Small",
			cfg: &types.ModelConfig{
				VocabSize:             1000,
				HiddenSize:            128,
				NumHeads:              2,
				NumLayers:             2,
				MaxPositionEmbeddings: 64,
			},
		},
		{
			name: "Medium",
			cfg: &types.ModelConfig{
				VocabSize:             5000,
				HiddenSize:            512,
				NumHeads:              8,
				NumLayers:             8,
				MaxPositionEmbeddings: 128,
			},
		},
	}

	for _, c := range configs {
		b.Run(fmt.Sprintf("Create_%s", c.name), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = model.NewMiniMindModel(c.cfg)
			}
		})
	}
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
	b.ReportAllocs()

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
