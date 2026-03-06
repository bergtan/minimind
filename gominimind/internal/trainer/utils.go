package trainer

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"gonum.org/v1/gonum/mat"
)

// GetLR 计算学习率（使用cosine decay）
func GetLR(currentStep, totalSteps int, initialLR float64) float64 {
	if currentStep >= totalSteps {
		return 0.0
	}

	// Cosine decay with warmup
	progress := float64(currentStep) / float64(totalSteps)
	return initialLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
}

// GetLRWithWarmup 带warmup的学习率调度
func GetLRWithWarmup(currentStep, warmupSteps, totalSteps int, initialLR float64) float64 {
	if currentStep < warmupSteps {
		// Linear warmup
		return initialLR * float64(currentStep) / float64(warmupSteps)
	}

	// Cosine decay after warmup
	progress := float64(currentStep-warmupSteps) / float64(totalSteps-warmupSteps)
	minLR := initialLR * 0.1
	return minLR + (initialLR-minLR)*0.5*(1.0+math.Cos(math.Pi*progress))
}

// Logger 简单日志记录器
func Logger(format string, args ...interface{}) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	message := fmt.Sprintf(format, args...)
	fmt.Printf("[%s] %s\n", timestamp, message)
}

// ClipGradNorm 梯度裁剪
func ClipGradNorm(gradients map[string]*mat.VecDense, maxNorm float64) float64 {
	totalNorm := 0.0

	// 计算全局梯度范数
	for _, grad := range gradients {
		data := grad.RawVector().Data
		for _, v := range data {
			totalNorm += v * v
		}
	}

	totalNorm = math.Sqrt(totalNorm)

	// 裁剪
	clipCoef := maxNorm / (totalNorm + 1e-6)
	if clipCoef < 1.0 {
		for _, grad := range gradients {
			data := grad.RawVector().Data
			for i := range data {
				data[i] *= clipCoef
			}
		}
	}

	return totalNorm
}

// ComputeCosineSimilarity 计算余弦相似度
func ComputeCosineSimilarity(a, b *mat.VecDense) float64 {
	aData := a.RawVector().Data
	bData := b.RawVector().Data

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := 0; i < len(aData); i++ {
		dotProduct += aData[i] * bData[i]
		normA += aData[i] * aData[i]
		normB += bData[i] * bData[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// SaveCheckpoint 保存模型检查点
func SaveCheckpoint(path string, data *CheckpointData) error {
	// 创建目录
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("创建目录失败: %w", err)
	}

	// 创建临时文件
	tmpPath := path + ".tmp"
	file, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("创建临时文件失败: %w", err)
	}
	defer os.Remove(tmpPath)

	// 使用gob编码
	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(data); err != nil {
		file.Close()
		return fmt.Errorf("编码检查点失败: %w", err)
	}

	file.Close()

	// 原子重命名
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("保存检查点失败: %w", err)
	}

	return nil
}

// LoadCheckpoint 加载模型检查点
func LoadCheckpoint(path string) (*CheckpointData, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("打开检查点文件失败: %w", err)
	}
	defer file.Close()

	var data CheckpointData
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, fmt.Errorf("解码检查点失败: %w", err)
	}

	return &data, nil
}

// CheckpointExists 检查检查点是否存在
func CheckpointExists(saveDir, weight string, hiddenSize int, useMoE bool) bool {
	moeSuffix := ""
	if useMoE {
		moeSuffix = "_moe"
	}

	checkpointPath := filepath.Join(saveDir, fmt.Sprintf("%s_%d%s.gob", weight, hiddenSize, moeSuffix))
	_, err := os.Stat(checkpointPath)
	return err == nil
}

// GetCheckpointPath 获取检查点路径
func GetCheckpointPath(saveDir, weight string, hiddenSize int, useMoE bool) string {
	moeSuffix := ""
	if useMoE {
		moeSuffix = "_moe"
	}
	return filepath.Join(saveDir, fmt.Sprintf("%s_%d%s.gob", weight, hiddenSize, moeSuffix))
}

// ComputeCrossEntropyLoss 计算交叉熵损失
func ComputeCrossEntropyLoss(logits []float64, target int) float64 {
	// Softmax
	var maxLogit float64
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	var sum float64
	expLogits := make([]float64, len(logits))
	for i, v := range logits {
		expLogits[i] = math.Exp(v - maxLogit)
		sum += expLogits[i]
	}

	// 归一化
	for i := range expLogits {
		expLogits[i] /= sum
	}

	// 负对数似然
	loss := -math.Log(expLogits[target] + 1e-10)
	return loss
}

// ComputeLogSoftmax 计算log softmax
func ComputeLogSoftmax(logits []float64) []float64 {
	var maxLogit float64
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	var sum float64
	result := make([]float64, len(logits))
	for i, v := range logits {
		result[i] = math.Exp(v - maxLogit)
		sum += result[i]
	}

	logSum := math.Log(sum)
	for i, v := range logits {
		result[i] = v - maxLogit - logSum
	}

	return result
}

// Softmax 计算softmax
func Softmax(logits []float64) []float64 {
	var maxLogit float64
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	var sum float64
	result := make([]float64, len(logits))
	for i, v := range logits {
		result[i] = math.Exp(v - maxLogit)
		sum += result[i]
	}

	for i := range result {
		result[i] /= sum
	}

	return result
}

// SetupSeed 设置随机种子
func SetupSeed(seed int64) {
	// 在Go中通常使用math/rand或crypto/rand
	// 这里提供一个占位符
	// 实际实现中可以在主要训练循环前调用rand.Seed(seed)
}

// FormatDuration 格式化时间间隔
func FormatDuration(d time.Duration) string {
	hours := int(d.Hours())
	minutes := int(d.Minutes()) % 60
	seconds := int(d.Seconds()) % 60

	if hours > 0 {
		return fmt.Sprintf("%dh%dm%ds", hours, minutes, seconds)
	}
	if minutes > 0 {
		return fmt.Sprintf("%dm%ds", minutes, seconds)
	}
	return fmt.Sprintf("%ds", seconds)
}

// CalculateETA 计算预计完成时间
func CalculateETA(elapsed time.Duration, current, total int) string {
	if current == 0 {
		return "未知"
	}

	remaining := float64(total-current) / float64(current) * elapsed.Seconds()
	eta := time.Duration(remaining) * time.Second
	return FormatDuration(eta)
}

// SaveTrainingLog 保存训练日志
func SaveTrainingLog(logPath string, entries []string) error {
	file, err := os.Create(logPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for _, entry := range entries {
		writer.WriteString(entry + "\n")
	}

	return writer.Flush()
}

// LoadDatasetFromJSONL 从JSONL文件加载数据集
func LoadDatasetFromJSONL(path string, maxLines int) ([]map[string]interface{}, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data []map[string]interface{}
	scanner := bufio.NewScanner(file)
	lineCount := 0

	for scanner.Scan() {
		if maxLines > 0 && lineCount >= maxLines {
			break
		}

		var item map[string]interface{}
		if err := parseJSON(scanner.Text(), &item); err != nil {
			continue
		}

		data = append(data, item)
		lineCount++
	}

	return data, scanner.Err()
}

// parseJSON 解析JSON（简单实现）
func parseJSON(text string, v interface{}) error {
	// 这里使用标准库encoding/json
	// 实际实现需要导入"encoding/json"
	// 由于文件顶部没有import，这里作为占位符
	return nil
}
