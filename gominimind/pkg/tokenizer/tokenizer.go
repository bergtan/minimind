package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/sirupsen/logrus"
)

// Tokenizer 分词器接口
type Tokenizer interface {
	// 编码方法
	Encode(text string) ([]int, error)
	EncodeBatch(texts []string) ([][]int, error)

	// 解码方法
	Decode(tokens []int) (string, error)
	DecodeBatch(tokenLists [][]int) ([]string, error)

	// 词汇表管理
	GetVocabSize() int
	GetSpecialTokens() map[string]int
	AddSpecialToken(token string, id int) error

	// 工具方法
	CountTokens(text string) int
	Truncate(text string, maxTokens int) string
}

// MiniMindTokenizer MiniMind分词器实现
type MiniMindTokenizer struct {
	vocab         map[string]int
	idToToken     map[int]string
	specialTokens map[string]int
	vocabSize     int

	// 配置
	config *TokenizerConfig
	logger *logrus.Logger

	// 缓存
	cache map[string][]int
	mutex sync.RWMutex
}

// TokenizerConfig 分词器配置
type TokenizerConfig struct {
	VocabPath string `json:"vocab_path"`
	VocabSize int    `json:"vocab_size"`
	MaxLength int    `json:"max_length"`
	PadToken  string `json:"pad_token"`
	EOSToken  string `json:"eos_token"`
	BOSToken  string `json:"bos_token"`
	UNKToken  string `json:"unk_token"`

	// 分词策略
	UseByteFallback bool `json:"use_byte_fallback"`
	UseRegex        bool `json:"use_regex"`

	// 缓存配置
	EnableCache bool `json:"enable_cache"`
	CacheSize   int  `json:"cache_size"`
}

// NewMiniMindTokenizer 创建新的MiniMind分词器
func NewMiniMindTokenizer(config *TokenizerConfig) (*MiniMindTokenizer, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	tokenizer := &MiniMindTokenizer{
		vocab:         make(map[string]int),
		idToToken:     make(map[int]string),
		specialTokens: make(map[string]int),
		config:        config,
		logger:        logrus.New(),
		cache:         make(map[string][]int),
	}

	// 初始化词汇表
	if err := tokenizer.loadVocab(); err != nil {
		return nil, fmt.Errorf("failed to load vocab: %w", err)
	}

	// 初始化特殊token
	tokenizer.initializeSpecialTokens()

	tokenizer.logger.Info("MiniMind tokenizer initialized successfully")

	return tokenizer, nil
}

// loadVocab 加载词汇表
func (t *MiniMindTokenizer) loadVocab() error {
	if t.config.VocabPath == "" {
		return t.initializeDefaultVocab()
	}

	t.logger.Infof("Loading vocabulary from: %s", t.config.VocabPath)

	file, err := os.Open(t.config.VocabPath)
	if err != nil {
		return fmt.Errorf("failed to open vocab file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	tokenID := 0

	for scanner.Scan() {
		line := scanner.Text()
		token := strings.TrimSpace(line)

		if token == "" {
			continue
		}

		t.vocab[token] = tokenID
		t.idToToken[tokenID] = token
		tokenID++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading vocab file: %w", err)
	}

	t.vocabSize = tokenID
	t.logger.Infof("Loaded %d tokens from vocabulary", t.vocabSize)

	return nil
}

// initializeDefaultVocab 初始化默认词汇表
func (t *MiniMindTokenizer) initializeDefaultVocab() error {
	t.logger.Info("Initializing default vocabulary")

	// 添加ASCII字符
	for i := 0; i < 128; i++ {
		token := string(rune(i))
		t.vocab[token] = i
		t.idToToken[i] = token
	}

	// 添加常用中文汉字（简化实现）
	chineseChars := "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"

	startID := 128
	for i, char := range chineseChars {
		token := string(char)
		tokenID := startID + i
		t.vocab[token] = tokenID
		t.idToToken[tokenID] = token
	}

	// 添加特殊token
	specialTokens := map[string]int{
		"<|endoftext|>": 10000,
		"<|im_start|>":  10001,
		"<|im_end|>":    10002,
		"<|pad|>":       10003,
		"<|unk|>":       10004,
	}

	for token, id := range specialTokens {
		t.vocab[token] = id
		t.idToToken[id] = token
		t.specialTokens[token] = id
	}

	t.vocabSize = 10005
	t.logger.Infof("Initialized default vocabulary with %d tokens", t.vocabSize)

	return nil
}

// initializeSpecialTokens 初始化特殊token
func (t *MiniMindTokenizer) initializeSpecialTokens() {
	// 设置默认特殊token
	if t.config.PadToken != "" {
		t.specialTokens[t.config.PadToken] = t.vocab[t.config.PadToken]
	}
	if t.config.EOSToken != "" {
		t.specialTokens[t.config.EOSToken] = t.vocab[t.config.EOSToken]
	}
	if t.config.BOSToken != "" {
		t.specialTokens[t.config.BOSToken] = t.vocab[t.config.BOSToken]
	}
	if t.config.UNKToken != "" {
		t.specialTokens[t.config.UNKToken] = t.vocab[t.config.UNKToken]
	}
}

// Encode 编码文本为token序列
func (t *MiniMindTokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return []int{}, nil
	}

	// 检查缓存
	if t.config.EnableCache {
		t.mutex.RLock()
		if tokens, exists := t.cache[text]; exists {
			t.mutex.RUnlock()
			return tokens, nil
		}
		t.mutex.RUnlock()
	}

	tokens, err := t.encodeText(text)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	// 更新缓存
	if t.config.EnableCache {
		t.mutex.Lock()
		// 简单的LRU缓存管理
		if len(t.cache) >= t.config.CacheSize {
			// 移除最旧的条目（简化实现）
			for key := range t.cache {
				delete(t.cache, key)
				break
			}
		}
		t.cache[text] = tokens
		t.mutex.Unlock()
	}

	return tokens, nil
}

// encodeText 实际编码实现
func (t *MiniMindTokenizer) encodeText(text string) ([]int, error) {
	// 预处理文本
	text = t.preprocessText(text)

	if t.config.UseRegex {
		return t.encodeWithRegex(text)
	}

	return t.encodeWithBytePair(text)
}

// preprocessText 预处理文本
func (t *MiniMindTokenizer) preprocessText(text string) string {
	// 标准化空白字符
	re := regexp.MustCompile(`\s+`)
	text = re.ReplaceAllString(text, " ")

	// 去除首尾空白
	text = strings.TrimSpace(text)

	return text
}

// encodeWithRegex 使用正则表达式分词
func (t *MiniMindTokenizer) encodeWithRegex(text string) ([]int, error) {
	// 中文分词正则表达式（简化实现）
	pattern := `[\p{Han}]+|[A-Za-z]+|\d+|[^\s\p{Han}A-Za-z\d]`
	re := regexp.MustCompile(pattern)

	matches := re.FindAllString(text, -1)
	tokens := make([]int, 0, len(matches))

	for _, match := range matches {
		tokenID, exists := t.vocab[match]
		if !exists {
			if t.config.UseByteFallback {
				// 回退到字节编码
				byteTokens := t.encodeBytes(match)
				tokens = append(tokens, byteTokens...)
			} else {
				// 使用UNK token
				if unkID, exists := t.specialTokens[t.config.UNKToken]; exists {
					tokens = append(tokens, unkID)
				} else {
					return nil, fmt.Errorf("unknown token: %s", match)
				}
			}
		} else {
			tokens = append(tokens, tokenID)
		}
	}

	return tokens, nil
}

// encodeWithBytePair 使用BPE分词
func (t *MiniMindTokenizer) encodeWithBytePair(text string) ([]int, error) {
	// 简化BPE实现
	tokens := make([]int, 0)
	runes := []rune(text)

	for i := 0; i < len(runes); {
		// 尝试匹配最长token
		maxMatch := ""
		maxLength := 0

		for j := i + 1; j <= len(runes); j++ {
			substring := string(runes[i:j])
			if _, exists := t.vocab[substring]; exists {
				if len(substring) > len(maxMatch) {
					maxMatch = substring
					maxLength = j - i
				}
			}
		}

		if maxMatch != "" {
			tokenID := t.vocab[maxMatch]
			tokens = append(tokens, tokenID)
			i += maxLength
		} else {
			// 处理未知字符
			if t.config.UseByteFallback {
				byteTokens := t.encodeBytes(string(runes[i]))
				tokens = append(tokens, byteTokens...)
			} else {
				if unkID, exists := t.specialTokens[t.config.UNKToken]; exists {
					tokens = append(tokens, unkID)
				} else {
					return nil, fmt.Errorf("unknown character: %s", string(runes[i]))
				}
			}
			i++
		}
	}

	return tokens, nil
}

// encodeBytes 编码字节序列
func (t *MiniMindTokenizer) encodeBytes(text string) []int {
	bytes := []byte(text)
	tokens := make([]int, len(bytes))

	for i, b := range bytes {
		tokens[i] = int(b)
	}

	return tokens
}

// EncodeBatch 批量编码
func (t *MiniMindTokenizer) EncodeBatch(texts []string) ([][]int, error) {
	results := make([][]int, len(texts))
	errors := make([]error, len(texts))

	var wg sync.WaitGroup

	for i, text := range texts {
		wg.Add(1)
		go func(idx int, txt string) {
			defer wg.Done()

			tokens, err := t.Encode(txt)
			if err != nil {
				errors[idx] = err
				return
			}

			results[idx] = tokens
		}(i, text)
	}

	wg.Wait()

	// 检查错误
	for _, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("batch encoding failed: %w", err)
		}
	}

	return results, nil
}

// Decode 解码token序列为文本
func (t *MiniMindTokenizer) Decode(tokens []int) (string, error) {
	if len(tokens) == 0 {
		return "", nil
	}

	var builder strings.Builder

	for _, tokenID := range tokens {
		token, exists := t.idToToken[tokenID]
		if !exists {
			// 处理未知token
			if t.config.UseByteFallback {
				if tokenID >= 0 && tokenID < 256 {
					builder.WriteByte(byte(tokenID))
				} else {
					builder.WriteString("<?>")
				}
			} else {
				builder.WriteString("<?>")
			}
		} else {
			// 跳过特殊token（除非是文本内容）
			if _, isSpecial := t.specialTokens[token]; !isSpecial || token == t.config.PadToken {
				builder.WriteString(token)
			}
		}
	}

	return builder.String(), nil
}

// DecodeBatch 批量解码
func (t *MiniMindTokenizer) DecodeBatch(tokenLists [][]int) ([]string, error) {
	results := make([]string, len(tokenLists))
	errors := make([]error, len(tokenLists))

	var wg sync.WaitGroup

	for i, tokens := range tokenLists {
		wg.Add(1)
		go func(idx int, tks []int) {
			defer wg.Done()

			text, err := t.Decode(tks)
			if err != nil {
				errors[idx] = err
				return
			}

			results[idx] = text
		}(i, tokens)
	}

	wg.Wait()

	// 检查错误
	for _, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("batch decoding failed: %w", err)
		}
	}

	return results, nil
}

// GetVocabSize 获取词汇表大小
func (t *MiniMindTokenizer) GetVocabSize() int {
	return t.vocabSize
}

// GetSpecialTokens 获取特殊token映射
func (t *MiniMindTokenizer) GetSpecialTokens() map[string]int {
	return t.specialTokens
}

// AddSpecialToken 添加特殊token
func (t *MiniMindTokenizer) AddSpecialToken(token string, id int) error {
	if token == "" {
		return fmt.Errorf("token cannot be empty")
	}

	if _, exists := t.vocab[token]; exists {
		return fmt.Errorf("token already exists: %s", token)
	}

	t.vocab[token] = id
	t.idToToken[id] = token
	t.specialTokens[token] = id

	// 更新词汇表大小
	if id >= t.vocabSize {
		t.vocabSize = id + 1
	}

	t.logger.Infof("Added special token: %s -> %d", token, id)

	return nil
}

// CountTokens 计算token数量
func (t *MiniMindTokenizer) CountTokens(text string) int {
	tokens, err := t.Encode(text)
	if err != nil {
		// 如果编码失败，使用字符数作为估计
		return utf8.RuneCountInString(text)
	}

	return len(tokens)
}

// Truncate 截断文本到最大token数
func (t *MiniMindTokenizer) Truncate(text string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}

	tokens, err := t.Encode(text)
	if err != nil {
		// 如果编码失败，使用字符截断
		runes := []rune(text)
		if len(runes) > maxTokens {
			return string(runes[:maxTokens])
		}
		return text
	}

	if len(tokens) <= maxTokens {
		return text
	}

	// 解码前maxTokens个token
	truncatedTokens := tokens[:maxTokens]
	truncatedText, err := t.Decode(truncatedTokens)
	if err != nil {
		// 如果解码失败，返回原始文本的字符截断
		runes := []rune(text)
		if len(runes) > maxTokens {
			return string(runes[:maxTokens])
		}
		return text
	}

	return truncatedText
}

// SaveVocab 保存词汇表到文件
func (t *MiniMindTokenizer) SaveVocab(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create vocab file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// 按token排序
	tokens := make([]string, 0, len(t.vocab))
	for token := range t.vocab {
		tokens = append(tokens, token)
	}
	sort.Strings(tokens)

	for _, token := range tokens {
		if _, err := writer.WriteString(token + "\n"); err != nil {
			return fmt.Errorf("failed to write token: %w", err)
		}
	}

	if err := writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush writer: %w", err)
	}

	t.logger.Infof("Saved vocabulary to: %s", filePath)

	return nil
}

// SaveConfig 保存配置到文件
func (t *MiniMindTokenizer) SaveConfig(filePath string) error {
	configJSON, err := json.MarshalIndent(t.config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(filePath, configJSON, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	t.logger.Infof("Saved tokenizer config to: %s", filePath)

	return nil
}

// LoadConfig 从文件加载配置
func (t *MiniMindTokenizer) LoadConfig(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	var config TokenizerConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("failed to unmarshal config: %w", err)
	}

	t.config = &config
	t.logger.Infof("Loaded tokenizer config from: %s", filePath)

	return nil
}

// GetTokenInfo 获取token信息
func (t *MiniMindTokenizer) GetTokenInfo(tokenID int) (string, bool) {
	token, exists := t.idToToken[tokenID]
	return token, exists
}

// GetTokenID 获取token ID
func (t *MiniMindTokenizer) GetTokenID(token string) (int, bool) {
	tokenID, exists := t.vocab[token]
	return tokenID, exists
}

// GetConfig 获取配置
func (t *MiniMindTokenizer) GetConfig() *TokenizerConfig {
	return t.config
}

// ClearCache 清空缓存
func (t *MiniMindTokenizer) ClearCache() {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	t.cache = make(map[string][]int)
	t.logger.Info("Tokenizer cache cleared")
}

// GetCacheStats 获取缓存统计
func (t *MiniMindTokenizer) GetCacheStats() map[string]interface{} {
	t.mutex.RLock()
	defer t.mutex.RUnlock()

	return map[string]interface{}{
		"cache_size":     len(t.cache),
		"cache_enabled":  t.config.EnableCache,
		"max_cache_size": t.config.CacheSize,
	}
}

// ========== 工具函数 ==========

// DefaultTokenizerConfig 创建默认分词器配置
func DefaultTokenizerConfig() *TokenizerConfig {
	return &TokenizerConfig{
		VocabSize:       6400,
		MaxLength:       32768,
		PadToken:        "<|pad|>",
		EOSToken:        "<|endoftext|>",
		BOSToken:        "<|im_start|>",
		UNKToken:        "<|unk|>",
		UseByteFallback: true,
		UseRegex:        true,
		EnableCache:     true,
		CacheSize:       1000,
	}
}

// CreateMiniMindTokenizer 创建MiniMind分词器
func CreateMiniMindTokenizer(vocabPath string) (*MiniMindTokenizer, error) {
	config := DefaultTokenizerConfig()
	config.VocabPath = vocabPath

	return NewMiniMindTokenizer(config)
}

// CreateDefaultTokenizer 创建默认分词器
func CreateDefaultTokenizer() (*MiniMindTokenizer, error) {
	return NewMiniMindTokenizer(DefaultTokenizerConfig())
}

// Load 从文件加载分词器
func Load(path string) (*MiniMindTokenizer, error) {
	config := DefaultTokenizerConfig()
	config.VocabPath = path
	return NewMiniMindTokenizer(config)
}

// NewWithBPE 创建基于BPE的分词器
func NewWithBPE(vocab map[string]int, merges []string, vocabSize int, maxLength int) *MiniMindTokenizer {
	config := DefaultTokenizerConfig()
	if vocabSize > 0 {
		config.VocabSize = vocabSize
	}
	if maxLength > 0 {
		config.MaxLength = maxLength
	}
	tok, err := NewMiniMindTokenizer(config)
	if err != nil {
		// 回退到最简单的分词器
		return &MiniMindTokenizer{
			vocab:         make(map[string]int),
			idToToken:     make(map[int]string),
			specialTokens: make(map[string]int),
			config:        config,
			cache:         make(map[string][]int),
			vocabSize:     config.VocabSize,
		}
	}
	return tok
}

// VocabSize 返回词汇表大小
func (t *MiniMindTokenizer) VocabSize() int {
	return t.vocabSize
}

// PadID 返回填充token的ID
func (t *MiniMindTokenizer) PadID() int {
	if id, exists := t.specialTokens[t.config.PadToken]; exists {
		return id
	}
	return 0
}

// EOSID 返回EOS token的ID
func (t *MiniMindTokenizer) EOSID() int {
	if id, exists := t.specialTokens[t.config.EOSToken]; exists {
		return id
	}
	return 0
}

// BOSID 返回BOS token的ID
func (t *MiniMindTokenizer) BOSID() int {
	if id, exists := t.specialTokens[t.config.BOSToken]; exists {
		return id
	}
	return 0
}

// EncodeSimple 简化编码（不返回error，直接返回token列表）
func (t *MiniMindTokenizer) EncodeSimple(text string) []int {
	tokens, err := t.Encode(text)
	if err != nil {
		return []int{}
	}
	return tokens
}

// DecodeSimple 简化解码（不返回error）
func (t *MiniMindTokenizer) DecodeSimple(tokens []int) string {
	text, err := t.Decode(tokens)
	if err != nil {
		return ""
	}
	return text
}
