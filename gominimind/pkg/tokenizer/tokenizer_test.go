package tokenizer

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// createTestTokenizer 创建测试用的分词器
func createTestTokenizer() *MiniMindTokenizer {
	tok, err := NewMiniMindTokenizer(DefaultTokenizerConfig())
	if err != nil {
		panic(err)
	}
	return tok
}

// TestNewMiniMindTokenizer 测试分词器创建
func TestNewMiniMindTokenizer(t *testing.T) {
	t.Run("Default Config", func(t *testing.T) {
		tok, err := NewMiniMindTokenizer(DefaultTokenizerConfig())
		assert.NoError(t, err)
		assert.NotNil(t, tok)
		assert.Greater(t, tok.GetVocabSize(), 0)
	})

	t.Run("Nil Config", func(t *testing.T) {
		tok, err := NewMiniMindTokenizer(nil)
		assert.Error(t, err)
		assert.Nil(t, tok)
	})
}

// TestEncode 测试编码
func TestEncode(t *testing.T) {
	tok := createTestTokenizer()

	t.Run("Empty String", func(t *testing.T) {
		tokens, err := tok.Encode("")
		assert.NoError(t, err)
		assert.Equal(t, 0, len(tokens))
	})

	t.Run("Simple Text", func(t *testing.T) {
		tokens, err := tok.Encode("hello world")
		assert.NoError(t, err)
		assert.Greater(t, len(tokens), 0)
	})

	t.Run("Chinese Text", func(t *testing.T) {
		tokens, err := tok.Encode("你好世界")
		assert.NoError(t, err)
		assert.Greater(t, len(tokens), 0)
	})
}

// TestDecode 测试解码
func TestDecode(t *testing.T) {
	tok := createTestTokenizer()

	t.Run("Empty Tokens", func(t *testing.T) {
		text, err := tok.Decode([]int{})
		assert.NoError(t, err)
		assert.Equal(t, "", text)
	})
}

// TestEncodeSimple 测试简化编码
func TestEncodeSimple(t *testing.T) {
	tok := createTestTokenizer()

	t.Run("Normal Text", func(t *testing.T) {
		tokens := tok.EncodeSimple("test text")
		assert.Greater(t, len(tokens), 0)
	})

	t.Run("Empty Text", func(t *testing.T) {
		tokens := tok.EncodeSimple("")
		assert.Equal(t, 0, len(tokens))
	})
}

// TestDecodeSimple 测试简化解码
func TestDecodeSimple(t *testing.T) {
	tok := createTestTokenizer()

	text := tok.DecodeSimple([]int{})
	assert.Equal(t, "", text)
}

// TestVocabSize 测试词汇表大小
func TestVocabSize(t *testing.T) {
	tok := createTestTokenizer()
	assert.Greater(t, tok.VocabSize(), 0)
	assert.Equal(t, tok.VocabSize(), tok.GetVocabSize())
}

// TestPadID 测试PadID
func TestPadID(t *testing.T) {
	tok := createTestTokenizer()
	padID := tok.PadID()
	assert.GreaterOrEqual(t, padID, 0)
}

// TestSpecialTokens 测试特殊token
func TestSpecialTokens(t *testing.T) {
	tok := createTestTokenizer()
	special := tok.GetSpecialTokens()
	assert.NotNil(t, special)
}

// TestAddSpecialToken 测试添加特殊token
func TestAddSpecialToken(t *testing.T) {
	tok := createTestTokenizer()

	t.Run("Add New Token", func(t *testing.T) {
		err := tok.AddSpecialToken("<test_special>", 99999)
		assert.NoError(t, err)
	})

	t.Run("Add Empty Token", func(t *testing.T) {
		err := tok.AddSpecialToken("", 99998)
		assert.Error(t, err)
	})
}

// TestCountTokens 测试token计数
func TestCountTokens(t *testing.T) {
	tok := createTestTokenizer()
	count := tok.CountTokens("hello world")
	assert.Greater(t, count, 0)
}

// TestTruncate 测试截断
func TestTruncate(t *testing.T) {
	tok := createTestTokenizer()

	t.Run("Short Text", func(t *testing.T) {
		text := tok.Truncate("hi", 100)
		assert.Equal(t, "hi", text)
	})

	t.Run("Zero Max", func(t *testing.T) {
		text := tok.Truncate("hello", 0)
		assert.Equal(t, "", text)
	})
}

// TestEncodeBatch 测试批量编码
func TestEncodeBatch(t *testing.T) {
	tok := createTestTokenizer()

	texts := []string{"hello", "world", "test"}
	results, err := tok.EncodeBatch(texts)
	assert.NoError(t, err)
	assert.Equal(t, 3, len(results))

	for _, tokens := range results {
		assert.Greater(t, len(tokens), 0)
	}
}

// TestLoad 测试Load函数
func TestLoad(t *testing.T) {
	t.Run("Non-existent Path", func(t *testing.T) {
		_, err := Load("/non/existent/path")
		assert.Error(t, err)
	})
}

// TestNewWithBPE 测试NewWithBPE函数
func TestNewWithBPE(t *testing.T) {
	tok := NewWithBPE(nil, nil, 0, 0)
	assert.NotNil(t, tok)
}

// TestCreateDefaultTokenizer 测试创建默认分词器
func TestCreateDefaultTokenizer(t *testing.T) {
	tok, err := CreateDefaultTokenizer()
	assert.NoError(t, err)
	assert.NotNil(t, tok)
}

// TestGetTokenInfo 测试获取token信息
func TestGetTokenInfo(t *testing.T) {
	tok := createTestTokenizer()
	_, exists := tok.GetTokenInfo(0)
	// 0号token可能存在也可能不存在，这取决于词汇表
	_ = exists
}

// TestCacheOperations 测试缓存操作
func TestCacheOperations(t *testing.T) {
	config := DefaultTokenizerConfig()
	config.EnableCache = true
	config.CacheSize = 10

	tok, err := NewMiniMindTokenizer(config)
	assert.NoError(t, err)

	// 编码相同文本两次，第二次应该命中缓存
	text := "cache test"
	tokens1, err := tok.Encode(text)
	assert.NoError(t, err)

	tokens2, err := tok.Encode(text)
	assert.NoError(t, err)

	assert.Equal(t, tokens1, tokens2)

	// 测试缓存统计
	stats := tok.GetCacheStats()
	assert.NotNil(t, stats)

	// 测试清除缓存
	tok.ClearCache()
	stats = tok.GetCacheStats()
	assert.Equal(t, 0, stats["cache_size"])
}

// BenchmarkEncode 编码性能测试
func BenchmarkEncode(b *testing.B) {
	tok := createTestTokenizer()
	text := "The quick brown fox jumps over the lazy dog. This is a test sentence."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(text)
	}
}

// BenchmarkDecode 解码性能测试
func BenchmarkDecode(b *testing.B) {
	tok := createTestTokenizer()
	tokens := tok.EncodeSimple("The quick brown fox jumps over the lazy dog.")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.Decode(tokens)
	}
}

// TestConcurrentAccess 测试并发访问
func TestConcurrentAccess(t *testing.T) {
	tok := createTestTokenizer()

	done := make(chan bool, 100)
	for i := 0; i < 100; i++ {
		go func() {
			_, _ = tok.Encode("concurrent test")
			done <- true
		}()
	}

	for i := 0; i < 100; i++ {
		<-done
	}
}

// TestEdgeCases 测试边界情况
func TestEdgeCases(t *testing.T) {
	tok := createTestTokenizer()

	t.Run("Very Long Text", func(t *testing.T) {
		longText := strings.Repeat("hello ", 1000)
		tokens, err := tok.Encode(longText)
		assert.NoError(t, err)
		assert.Greater(t, len(tokens), 0)
	})

	t.Run("Special Characters", func(t *testing.T) {
		text := "hello@world.com test!#%&*"
		tokens, err := tok.Encode(text)
		assert.NoError(t, err)
		assert.Greater(t, len(tokens), 0)
	})

	t.Run("Unicode Characters", func(t *testing.T) {
		text := "你好世界 emoji👋"
		tokens, err := tok.Encode(text)
		assert.NoError(t, err)
		assert.Greater(t, len(tokens), 0)
	})
}
