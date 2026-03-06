package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"

	"gominimind/pkg/tokenizer"
)

// Dataset 数据集接口
type Dataset interface {
	Len() int
	Get(idx int) (interface{}, error)
}

// PretrainSample 预训练样本
type PretrainSample struct {
	InputIDs []int
	Labels   []int
}

// SFTSample SFT训练样本
type SFTSample struct {
	InputIDs []int
	Labels   []int
}

// DPOPair DPO训练样本对
type DPOPair struct {
	Prompt   string
	Chosen   string
	Rejected string
}

// DPOSample DPO训练样本
type DPOSample struct {
	XChosen      []int
	XRejected    []int
	YChosen      []int
	YRejected    []int
	MaskChosen   []float64
	MaskRejected []float64
}

// PretrainDataset 预训练数据集
type PretrainDataset struct {
	data      []PretrainSample
	tokenizer *tokenizer.MiniMindTokenizer
	maxLength int
	mu        sync.RWMutex
}

// NewPretrainDataset 创建预训练数据集
func NewPretrainDataset(dataPath string, tok *tokenizer.MiniMindTokenizer, maxLength int) (*PretrainDataset, error) {
	ds := &PretrainDataset{
		tokenizer: tok,
		maxLength: maxLength,
		data:      make([]PretrainSample, 0),
	}

	if err := ds.load(dataPath); err != nil {
		return nil, fmt.Errorf("加载数据集失败: %w", err)
	}

	return ds, nil
}

// load 从文件加载数据
func (ds *PretrainDataset) load(dataPath string) error {
	file, err := os.Open(dataPath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var item struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			continue
		}

		sample := ds.processText(item.Text)
		ds.data = append(ds.data, sample)
	}

	return scanner.Err()
}

// processText 处理文本样本
func (ds *PretrainDataset) processText(text string) PretrainSample {
	tokens := ds.tokenizer.EncodeSimple(text)

	// 截断或填充到maxLength
	if len(tokens) > ds.maxLength {
		tokens = tokens[:ds.maxLength]
	}

	// 创建输入和标签（因果语言建模：input_ids[i] -> labels[i+1]）
	inputIDs := make([]int, ds.maxLength)
	labels := make([]int, ds.maxLength)

	// 填充
	for i := 0; i < ds.maxLength; i++ {
		if i < len(tokens) {
			inputIDs[i] = tokens[i]
			if i < len(tokens)-1 {
				labels[i] = tokens[i+1]
			} else {
				labels[i] = -100 // 忽略填充位置的损失
			}
		} else {
			inputIDs[i] = ds.tokenizer.PadID()
			labels[i] = -100
		}
	}

	return PretrainSample{
		InputIDs: inputIDs,
		Labels:   labels,
	}
}

// Len 返回数据集长度
func (ds *PretrainDataset) Len() int {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return len(ds.data)
}

// Get 获取指定索引的样本
func (ds *PretrainDataset) Get(idx int) (PretrainSample, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if idx < 0 || idx >= len(ds.data) {
		return PretrainSample{}, fmt.Errorf("索引越界: %d", idx)
	}
	return ds.data[idx], nil
}

// GetBatch 获取批次数据
func (ds *PretrainDataset) GetBatch(indices []int) ([]PretrainSample, error) {
	batch := make([]PretrainSample, len(indices))
	for i, idx := range indices {
		sample, err := ds.Get(idx)
		if err != nil {
			return nil, err
		}
		batch[i] = sample
	}
	return batch, nil
}

// Shuffle 打乱数据集
func (ds *PretrainDataset) Shuffle() {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	rand.Shuffle(len(ds.data), func(i, j int) {
		ds.data[i], ds.data[j] = ds.data[j], ds.data[i]
	})
}

// SFTDataset SFT微调数据集
type SFTDataset struct {
	data      []SFTSample
	tokenizer *tokenizer.MiniMindTokenizer
	maxLength int
	mu        sync.RWMutex
}

// NewSFTDataset 创建SFT数据集
func NewSFTDataset(dataPath string, tok *tokenizer.MiniMindTokenizer, maxLength int) (*SFTDataset, error) {
	ds := &SFTDataset{
		tokenizer: tok,
		maxLength: maxLength,
		data:      make([]SFTSample, 0),
	}

	if err := ds.load(dataPath); err != nil {
		return nil, fmt.Errorf("加载SFT数据集失败: %w", err)
	}

	return ds, nil
}

// load 加载SFT数据
func (ds *SFTDataset) load(dataPath string) error {
	file, err := os.Open(dataPath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var item struct {
			Conversations []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"conversations"`
		}
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			continue
		}

		sample := ds.processConversations(item.Conversations)
		ds.data = append(ds.data, sample)
	}

	return scanner.Err()
}

// processConversations 处理对话数据
func (ds *SFTDataset) processConversations(conversations []struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}) SFTSample {
	// 格式化对话
	var dialogue strings.Builder
	for _, conv := range conversations {
		switch conv.Role {
		case "user":
			dialogue.WriteString(fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n", conv.Content))
		case "assistant":
			dialogue.WriteString(fmt.Sprintf("<|im_start|>assistant\n%s<|im_end|>\n", conv.Content))
		case "system":
			dialogue.WriteString(fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n", conv.Content))
		}
	}

	text := dialogue.String()
	tokens := ds.tokenizer.EncodeSimple(text)

	// 截断或填充
	if len(tokens) > ds.maxLength {
		tokens = tokens[:ds.maxLength]
	}

	inputIDs := make([]int, ds.maxLength)
	labels := make([]int, ds.maxLength)

	// 填充并创建标签（只对assistant的回答计算损失）
	_ = false // inAssistant placeholder for future use
	for i := 0; i < ds.maxLength; i++ {
		if i < len(tokens) {
			inputIDs[i] = tokens[i]
			if i < len(tokens)-1 {
				// TODO: 根据token判断是否在assistant部分
				labels[i] = tokens[i+1]
			} else {
				labels[i] = -100
			}
		} else {
			inputIDs[i] = ds.tokenizer.PadID()
			labels[i] = -100
		}
	}

	return SFTSample{
		InputIDs: inputIDs,
		Labels:   labels,
	}
}

// Len 返回数据集长度
func (ds *SFTDataset) Len() int {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return len(ds.data)
}

// Get 获取指定索引的样本
func (ds *SFTDataset) Get(idx int) (SFTSample, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if idx < 0 || idx >= len(ds.data) {
		return SFTSample{}, fmt.Errorf("索引越界: %d", idx)
	}
	return ds.data[idx], nil
}

// Shuffle 打乱数据集
func (ds *SFTDataset) Shuffle() {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	rand.Shuffle(len(ds.data), func(i, j int) {
		ds.data[i], ds.data[j] = ds.data[j], ds.data[i]
	})
}

// GetBatch 获取批次数据
func (ds *SFTDataset) GetBatch(indices []int) ([]SFTSample, error) {
	batch := make([]SFTSample, len(indices))
	for i, idx := range indices {
		sample, err := ds.Get(idx)
		if err != nil {
			return nil, err
		}
		batch[i] = sample
	}
	return batch, nil
}

// DPODataset DPO训练数据集
type DPODataset struct {
	data      []DPOSample
	tokenizer *tokenizer.MiniMindTokenizer
	maxLength int
	mu        sync.RWMutex
}

// NewDPODataset 创建DPO数据集
func NewDPODataset(dataPath string, tok *tokenizer.MiniMindTokenizer, maxLength int) (*DPODataset, error) {
	ds := &DPODataset{
		tokenizer: tok,
		maxLength: maxLength,
		data:      make([]DPOSample, 0),
	}

	if err := ds.load(dataPath); err != nil {
		return nil, fmt.Errorf("加载DPO数据集失败: %w", err)
	}

	return ds, nil
}

// load 加载DPO数据
func (ds *DPODataset) load(dataPath string) error {
	file, err := os.Open(dataPath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var pair DPOPair
		if err := json.Unmarshal([]byte(line), &pair); err != nil {
			continue
		}

		sample := ds.processPair(pair)
		ds.data = append(ds.data, sample)
	}

	return scanner.Err()
}

// processPair 处理DPO样本对
func (ds *DPODataset) processPair(pair DPOPair) DPOSample {
	// 编码prompt
	promptTokens := ds.tokenizer.EncodeSimple(pair.Prompt)

	// 编码chosen和rejected回答
	chosenTokens := ds.tokenizer.EncodeSimple(pair.Chosen)
	rejectedTokens := ds.tokenizer.EncodeSimple(pair.Rejected)

	// 构建完整序列（prompt + response）
	xChosen := append(promptTokens, chosenTokens...)
	xRejected := append(promptTokens, rejectedTokens...)
	yChosen := append(make([]int, len(promptTokens)-1), xChosen[1:]...)
	yRejected := append(make([]int, len(promptTokens)-1), xRejected[1:]...)

	// 截断
	if len(xChosen) > ds.maxLength {
		xChosen = xChosen[:ds.maxLength]
		yChosen = yChosen[:ds.maxLength-1]
		yChosen = append(yChosen, -100)
	}
	if len(xRejected) > ds.maxLength {
		xRejected = xRejected[:ds.maxLength]
		yRejected = yRejected[:ds.maxLength-1]
		yRejected = append(yRejected, -100)
	}

	// 创建mask（只对response部分计算损失）
	maskChosen := make([]float64, len(xChosen))
	maskRejected := make([]float64, len(xRejected))

	for i := range maskChosen {
		if i >= len(promptTokens)-1 {
			maskChosen[i] = 1.0
		}
	}
	for i := range maskRejected {
		if i >= len(promptTokens)-1 {
			maskRejected[i] = 1.0
		}
	}

	return DPOSample{
		XChosen:      xChosen,
		XRejected:    xRejected,
		YChosen:      yChosen,
		YRejected:    yRejected,
		MaskChosen:   maskChosen,
		MaskRejected: maskRejected,
	}
}

// Len 返回数据集长度
func (ds *DPODataset) Len() int {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return len(ds.data)
}

// Get 获取指定索引的样本
func (ds *DPODataset) Get(idx int) (DPOSample, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if idx < 0 || idx >= len(ds.data) {
		return DPOSample{}, fmt.Errorf("索引越界: %d", idx)
	}
	return ds.data[idx], nil
}

// Shuffle 打乱数据集
func (ds *DPODataset) Shuffle() {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	rand.Shuffle(len(ds.data), func(i, j int) {
		ds.data[i], ds.data[j] = ds.data[j], ds.data[i]
	})
}

// GetBatch 获取批次数据
func (ds *DPODataset) GetBatch(indices []int) ([]DPOSample, error) {
	batch := make([]DPOSample, len(indices))
	for i, idx := range indices {
		sample, err := ds.Get(idx)
		if err != nil {
			return nil, err
		}
		batch[i] = sample
	}
	return batch, nil
}
