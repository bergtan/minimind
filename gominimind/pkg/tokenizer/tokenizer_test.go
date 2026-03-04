package tokenizer

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// createTestVocabulary - 创建测试词汇表
func createTestVocabulary() *Vocabulary {
	vocab := NewVocabulary()

	// Add special tokens
	vocab.AddSpecialToken("<pad>")
	vocab.AddSpecialToken("<bos>")
	vocab.AddSpecialToken("<eos>")
	vocab.AddSpecialToken("<unk>")

	// Add regular tokens
	tokens := []string{
		"hello", "world", "this", "is", "a", "test",
		"how", "are", "you", "today", "good", "morning",
		"the", "quick", "brown", "fox", "jumps", "over",
		"lazy", "dog", "cat", "dogs", "cats", "running",
		"run", "runs", "ran", "eat", "eating", "ate",
	}

	for _, token := range tokens {
		vocab.AddToken(token)
	}

	// Add some subword tokens for BPE
	vocab.AddToken("ing")
	vocab.AddToken("ed")
	vocab.AddToken("s")
	vocab.AddToken("run")
	vocab.AddToken("eat")

	return vocab
}

// TestNewVocabulary - 测试词汇表创建
func TestNewVocabulary(t *testing.T) {
	vocab := NewVocabulary()

	assert.NotNil(t, vocab)
	assert.NotNil(t, vocab.TokenToID)
	assert.NotNil(t, vocab.IDToToken)
	assert.NotNil(t, vocab.SpecialTokens)
	assert.Equal(t, 0, vocab.Size())
}

// TestAddToken - 测试添加token
func TestAddToken(t *testing.T) {
	vocab := NewVocabulary()

	t.Run("Add Regular Token", func(t *testing.T) {
		id := vocab.AddToken("hello")
		assert.Equal(t, 0, id)
		assert.Equal(t, 1, vocab.Size())

		// Should return same ID for existing token
		id2 := vocab.AddToken("hello")
		assert.Equal(t, id, id2)
		assert.Equal(t, 1, vocab.Size())
	})

	t.Run("Add Multiple Tokens", func(t *testing.T) {
		vocab.AddToken("world")
		vocab.AddToken("test")

		assert.Equal(t, 3, vocab.Size())
	})
}

// TestAddSpecialToken - 测试添加特殊token
func TestAddSpecialToken(t *testing.T) {
	vocab := NewVocabulary()

	t.Run("Add Special Tokens", func(t *testing.T) {
		vocab.AddSpecialToken("<pad>")
		vocab.AddSpecialToken("<bos>")
		vocab.AddSpecialToken("<eos>")

		assert.Equal(t, 3, vocab.Size())
		assert.True(t, vocab.IsSpecialToken("<pad>"))
		assert.True(t, vocab.IsSpecialToken("<bos>"))
		assert.True(t, vocab.IsSpecialToken("<eos>"))
		assert.False(t, vocab.IsSpecialToken("hello"))
	})

	t.Run("Reserved IDs", func(t *testing.T) {
		vocab := createTestVocabulary()

		padID := vocab.TokenToID["<pad>"]
		bosID := vocab.TokenToID["<bos>"]
		eosID := vocab.TokenToID["<eos>"]
		unkID := vocab.TokenToID["<unk>"]

		// Special tokens should have reserved low IDs
		assert.Less(t, padID, 10)
		assert.Less(t, bosID, 10)
		assert.Less(t, eosID, 10)
		assert.Less(t, unkID, 10)
	})
}

// TestGetTokenID - 测试获取token ID
func TestGetTokenID(t *testing.T) {
	vocab := createTestVocabulary()

	t.Run("Existing Token", func(t *testing.T) {
		id := vocab.GetTokenID("hello")
		assert.NotEqual(t, -1, id)

		// Verify reverse lookup
		token := vocab.GetTokenByID(id)
		assert.Equal(t, "hello", token)
	})

	t.Run("Unknown Token", func(t *testing.T) {
		id := vocab.GetTokenID("xyznonexistent")
		assert.Equal(t, vocab.GetTokenID("<unk>"), id)
	})
}

// TestEncodeSimple - 测试简单编码
func TestEncodeSimple(t *testing.T) {
	cfg := &Config{
		VocabSize:    1000,
		MaxSeqLength: 128,
		ModelDim:     512,
	}

	tokenizer, err := NewBPETokenizer(cfg)
	assert.NoError(t, err)

	// Create test vocabulary
	vocab := createTestVocabulary()
	tokenizer.Vocab = vocab

	t.Run("Encode Simple Text", func(t *testing.T) {
		text := "hello world"
		tokens := tokenizer.Encode(text, false)

		assert.Greater(t, len(tokens), 0)
		assert.LessOrEqual(t, len(tokens), cfg.MaxSeqLength)

		// Check that tokens are valid IDs
		for _, token := range tokens {
			assert.GreaterOrEqual(t, token.ID, 0)
			assert.Less(t, token.ID, vocab.Size())
		}
	})

	t.Run("Encode with Special Tokens", func(t *testing.T) {
		text := "test"
		tokens := tokenizer.Encode(text, true)

		// Should have BOS and EOS tokens
		bosID := vocab.GetTokenID("<bos>")
		eosID := vocab.GetTokenID("<eos>")

		assert.GreaterOrEqual(t, len(tokens), 2)
		assert.Equal(t, bosID, tokens[0].ID)
		assert.Equal(t, eosID, tokens[len(tokens)-1].ID)
	})

	t.Run("Encode Empty String", func(t *testing.T) {
		text := ""
		tokens := tokenizer.Encode(text, false)

		assert.Equal(t, 0, len(tokens))
	})
}

// TestDecodeSimple - 测试简单解码
func TestDecodeSimple(t *testing.T) {
	cfg := &Config{
		VocabSize:    1000,
		MaxSeqLength: 128,
		ModelDim:     512,
	}

	tokenizer, err := NewBPETokenizer(cfg)
	assert.NoError(t, err)

	vocab := createTestVocabulary()
	tokenizer.Vocab = vocab

	t.Run("Decode Tokens", func(t *testing.T) {
		// Encode and then decode
		original := "hello world"
		tokens := tokenizer.Encode(original, false)
		decoded := tokenizer.Decode(tokens)

		// Decoded text should contain original words
		assert.True(t, strings.Contains(strings.ToLower(decoded), "hello") ||
			strings.Contains(strings.ToLower(decoded), "world"))
	})

	t.Run("Decode with Special Tokens", func(t *testing.T) {
		text := "test"
		tokens := tokenizer.Encode(text, true)
		decoded := tokenizer.Decode(tokens)

		// Special tokens should be stripped in decode
		assert.False(t, strings.Contains(decoded, "<bos>"))
		assert.False(t, strings.Contains(decoded, "<eos>"))
	})
}

// TestSplitIntoWords - 测试文本分词
func TestSplitIntoWords(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{
			input:    "hello world",
			expected: []string{"hello", "world"},
		},
		{
			input:    "  hello   world  ",
			expected: []string{"hello", "world"},
		},
		{
			input:    "Hello, World!",
			expected: []string{"Hello", "World"},
		},
		{
			input:    "hello-world",
			expected: []string{"hello", "world"},
		},
		{
			input:    "123 test",
			expected: []string{"123", "test"},
		},
		{
			input:    "",
			expected: []string{},
		},
	}

	for _, test := range tests {
		result := splitIntoWords(test.input)
		assert.Equal(t, test.expected, result)
	}
}

// TestApplyBPE - 测试BPE分词
func TestApplyBPE(t *testing.T) {
	vocab := createTestVocabulary()

	t.Run("BPE on Simple Word", func(t *testing.T) {
		word := "running"
		tokens := applyBPE(word, vocab)

		assert.Greater(t, len(tokens), 0)

		// Check that tokens can be combined back
		reconstructed := ""
		for _, token := range tokens {
			reconstructed += token.Text
		}
		assert.Equal(t, word, reconstructed)
	})

	t.Run("BPE on Unknown Word", func(t *testing.T) {
		word := "xyznonexistent"
		tokens := applyBPE(word, vocab)

		// Should return character-level tokens
		assert.Greater(t, len(tokens), 0)
	})
}

// TestNewBPETokenizer - 测试BPE Tokenizer创建
func TestNewBPETokenizer(t *testing.T) {
	t.Run("Valid Config", func(t *testing.T) {
		cfg := &Config{
			VocabSize:    1000,
			MaxSeqLength: 128,
			ModelDim:     512,
		}

		tokenizer, err := NewBPETokenizer(cfg)
		assert.NoError(t, err)
		assert.NotNil(t, tokenizer)
		assert.NotNil(t, tokenizer.Vocab)
		assert.NotNil(t, tokenizer.Merges)
		assert.Equal(t, cfg, tokenizer.Config)
	})

	t.Run("Invalid Config - Zero VocabSize", func(t *testing.T) {
		cfg := &Config{
			VocabSize: 0,
		}

		tokenizer, err := NewBPETokenizer(cfg)
		assert.Error(t, err)
		assert.Nil(t, tokenizer)
	})

	t.Run("Invalid Config - Negative MaxSeqLength", func(t *testing.T) {
		cfg := &Config{
			VocabSize:    1000,
			MaxSeqLength: -1,
		}

		tokenizer, err := NewBPETokenizer(cfg)
		assert.Error(t, err)
		assert.Nil(t, tokenizer)
	})
}

// TestTokenizerLoad - 测试加载tokenizer
func TestTokenizerLoad(t *testing.T) {
	t.Run("Load Without Train", func(t *testing.T) {
		cfg := &Config{
			VocabSize:    100,
			MaxSeqLength: 32,
			ModelDim:     128,
		}

		tokenizer, err := LoadTokenizer(cfg)
		assert.NoError(t, err)
		assert.NotNil(t, tokenizer)

		// Should have basic special tokens
		padID := tokenizer.Vocab.GetTokenID("<pad>")
		assert.NotEqual(t, -1, padID)
	})
}

// TestBatchEncode - 测试批量编码
func TestBatchEncode(t *testing.T) {
	cfg := &Config{
		VocabSize:    1000,
		MaxSeqLength: 128,
		ModelDim:     512,
	}

	tokenizer, err := NewBPETokenizer(cfg)
	assert.NoError(t, err)

	vocab := createTestVocabulary()
	tokenizer.Vocab = vocab

	tests := []struct {
		texts            []string
		addSpecialTokens bool
	}{
		{
			texts:            []string{"hello", "world", "test"},
			addSpecialTokens: false,
		},
		{
			texts:            []string{"hello world", "good morning"},
			addSpecialTokens: true,
		},
		{
			texts:            []string{},
			addSpecialTokens: false,
		},
	}

	for _, test := range tests {
		results := tokenizer.BatchEncode(test.texts, test.addSpecialTokens)
		assert.Equal(t, len(test.texts), len(results))

		for _, tokens := range results {
			// Verify all tokens are valid
			for _, token := range tokens {
				assert.GreaterOrEqual(t, token.ID, 0)
				assert.Less(t, token.ID, vocab.Size())
				assert.LessOrEqual(t, len(tokens), cfg.MaxSeqLength)
			}
		}
	}
}

// TestBuildVocabularyFromCorpus - 测试从语料库构建词汇表
func TestBuildVocabularyFromCorpus(t *testing.T) {
	cfg := &Config{
		VocabSize:    100,
		MaxSeqLength: 32,
		ModelDim:     128,
	}

	tokenizer, err := NewBPETokenizer(cfg)
	assert.NoError(t, err)

	corpus := []string{
		"hello world this is a test",
		"the quick brown fox jumps",
		"over the lazy dog",
		"cats and dogs running fast",
	}

	vocab := tokenizer.BuildVocabularyFromCorpus(corpus)
	assert.NotNil(t, vocab)
	assert.Greater(t, vocab.Size(), 0)
	// Vocabulary should include special tokens
	assert.NotEqual(t, -1, vocab.GetTokenID("<pad>"))
	assert.NotEqual(t, -1, vocab.GetTokenID("<unk>"))
}

// TestSaveAndLoadVocabulary - 测试词汇表保存和加载
func TestSaveAndLoadVocabulary(t *testing.T) {
	original := createTestVocabulary()

	// Save to string
	data, err := original.Save()
	assert.NoError(t, err)
	assert.NotEmpty(t, data)

	// Load from string
	loaded := NewVocabulary()
	err = loaded.Load(data)
	assert.NoError(t, err)

	// Verify loaded vocabulary matches original
	assert.Equal(t, original.Size(), loaded.Size())

	// Check some tokens
	for token, id := range original.TokenToID {
		loadedID := loaded.GetTokenID(token)
		assert.Equal(t, id, loadedID)
	}
}

// TestCountPairs - 测试pair计数
func TestCountPairs(t *testing.T) {
	wordFreqs := map[string]int{
		"low":    5,
		"lower":  2,
		"lowest": 1,
	}

	pairs := countPairs(wordFreqs)

	// Check that pairs are counted correctly
	assert.Greater(t, len(pairs), 0)

	// 'l o' pair should appear frequently
	pair := Pair{First: "l", Second: "o"}
	assert.Greater(t, pairs[pair], 0)
}

// TestPerformMerge - 测试合并操作
func TestPerformMerge(t *testing.T) {
	wordFreqs := map[string]int{
		"low":    5,
		"lower":  2,
		"lowest": 1,
	}

	bestPair := Pair{First: "l", Second: "o"}
	newWordFreqs := performMerge(wordFreqs, bestPair)

	// Check that merge was performed
	assert.NotNil(t, newWordFreqs)

	// 'lo' should now appear in the vocabulary
	foundLo := false
	for word := range newWordFreqs {
		if strings.Contains(word, "lo") {
			foundLo = true
			break
		}
	}
	assert.True(t, foundLo)
}

// BenchmarkEncode - 编码性能测试
func BenchmarkEncode(b *testing.B) {
	cfg := &Config{
		VocabSize:    50000,
		MaxSeqLength: 512,
		ModelDim:     768,
	}

	tokenizer, _ := NewBPETokenizer(cfg)
	text := "The quick brown fox jumps over the lazy dog. This is a test sentence for benchmarking the tokenizer performance."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Encode(text, true)
	}
}

// BenchmarkDecode - 解码性能测试
func BenchmarkDecode(b *testing.B) {
	cfg := &Config{
		VocabSize:    50000,
		MaxSeqLength: 512,
		ModelDim:     768,
	}

	tokenizer, _ := NewBPETokenizer(cfg)
	text := "The quick brown fox jumps over the lazy dog."
	tokens := tokenizer.Encode(text, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Decode(tokens)
	}
}

// BenchmarkAddToken - 添加token性能测试
func BenchmarkAddToken(b *testing.B) {
	vocab := NewVocabulary()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vocab.AddToken(string(rune('a' + i%26)))
	}
}

// TestConcurrentAccess - 测试并发访问
func TestConcurrentAccess(t *testing.T) {
	vocab := createTestVocabulary()

	t.Run("Concurrent Read", func(t *testing.T) {
		done := make(chan bool, 100)

		for i := 0; i < 100; i++ {
			go func() {
				_ = vocab.GetTokenID("hello")
				done <- true
			}()
		}

		for i := 0; i < 100; i++ {
			<-done
		}
	})
}

// TestEdgeCases - 测试边界情况
func TestEdgeCases(t *testing.T) {
	cfg := &Config{
		VocabSize:    100,
		MaxSeqLength: 16,
		ModelDim:     64,
	}

	tokenizer, err := NewBPETokenizer(cfg)
	assert.NoError(t, err)

	vocab := createTestVocabulary()
	tokenizer.Vocab = vocab

	t.Run("Very Long Text", func(t *testing.T) {
		longText := strings.Repeat("hello ", 1000)
		tokens := tokenizer.Encode(longText, false)

		// Should be truncated to MaxSeqLength
		assert.LessOrEqual(t, len(tokens), cfg.MaxSeqLength)
	})

	t.Run("Text with Special Characters", func(t *testing.T) {
		text := "hello@world.com test!#%&*"
		tokens := tokenizer.Encode(text, false)
		assert.Greater(t, len(tokens), 0)
	})

	t.Run("Unicode Characters", func(t *testing.T) {
		text := "你好世界 こんにちは emoji👋"
		tokens := tokenizer.Encode(text, false)
		assert.Greater(t, len(tokens), 0)
	})

	t.Run("Numbers and Punctuation", func(t *testing.T) {
		text := "123,456.789 test-test_test+more"
		tokens := tokenizer.Encode(text, false)
		assert.Greater(t, len(tokens), 0)
	})
}
