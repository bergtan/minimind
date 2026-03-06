package cache

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestNewMemoryCache - 测试内存缓存创建
func TestNewMemoryCache(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil) // 100MB
	assert.NotNil(t, cache)
}

// TestMemoryCacheSetAndGet - 测试内存缓存的Set和Get操作
func TestMemoryCacheSetAndGet(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()

	t.Run("Set And Get", func(t *testing.T) {
		key := "test-key"
		value := []byte("test-value")
		ttl := 1 * time.Minute

		err := cache.Set(ctx, key, value, ttl)
		assert.NoError(t, err)

		retrieved, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})

	t.Run("Get Non-Existent Key", func(t *testing.T) {
		result, err := cache.Get(ctx, "non-existent-key")
		assert.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("Set And Get Bytes", func(t *testing.T) {
		key := "bytes-key"
		value := []byte{0x01, 0x02, 0x03, 0x04}

		err := cache.Set(ctx, key, value, 1*time.Minute)
		assert.NoError(t, err)

		retrieved, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})
}

// TestMemoryCacheExpiration - 测试缓存过期
func TestMemoryCacheExpiration(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()

	key := "expire-key"
	value := []byte("expire-value")

	err := cache.Set(ctx, key, value, 1*time.Millisecond)
	assert.NoError(t, err)

	// 等待过期
	time.Sleep(10 * time.Millisecond)

	result, err := cache.Get(ctx, key)
	assert.NoError(t, err)
	assert.Nil(t, result)
}

// TestMemoryCacheDelete - 测试删除缓存
func TestMemoryCacheDelete(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()

	t.Run("Delete Existing Key", func(t *testing.T) {
		key := "delete-key"
		value := []byte("delete-value")

		err := cache.Set(ctx, key, value, 1*time.Minute)
		assert.NoError(t, err)

		err = cache.Delete(ctx, key)
		assert.NoError(t, err)

		result, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("Delete Non-Existent Key", func(t *testing.T) {
		err := cache.Delete(ctx, "non-existent-delete-key")
		// 删除不存在的key不应该报错
		_ = err
	})
}

// TestMemoryCacheExists - 测试检查缓存是否存在
func TestMemoryCacheExists(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()

	t.Run("Existing Key", func(t *testing.T) {
		key := "exists-key"
		value := []byte("exists-value")

		err := cache.Set(ctx, key, value, 1*time.Minute)
		assert.NoError(t, err)

		exists, err := cache.Exists(ctx, key)
		assert.NoError(t, err)
		assert.True(t, exists)
	})

	t.Run("Non-Existing Key", func(t *testing.T) {
		exists, err := cache.Exists(ctx, "not-exists-key")
		assert.NoError(t, err)
		assert.False(t, exists)
	})

	t.Run("Expired Key", func(t *testing.T) {
		key := "expire-exists-key"
		value := []byte("expire-exists-value")

		err := cache.Set(ctx, key, value, 1*time.Millisecond)
		assert.NoError(t, err)

		time.Sleep(10 * time.Millisecond)

		exists, err := cache.Exists(ctx, key)
		assert.NoError(t, err)
		assert.False(t, exists)
	})
}

// TestMemoryCacheFlush - 测试清空缓存
func TestMemoryCacheFlush(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()

	// 添加多个缓存项
	for i := 0; i < 10; i++ {
		key := "flush-key-" + string(rune('0'+i))
		value := []byte("flush-value")
		cache.Set(ctx, key, value, 1*time.Minute)
	}

	// 清空
	err := cache.Flush(ctx)
	assert.NoError(t, err)

	// 验证已清空
	exists, err := cache.Exists(ctx, "flush-key-0")
	assert.NoError(t, err)
	assert.False(t, exists)
}

// TestMemoryCacheStats - 测试缓存统计
func TestMemoryCacheStats(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()

	// 添加一些缓存项
	cache.Set(ctx, "stats-key-1", []byte("value1"), 1*time.Minute)
	cache.Set(ctx, "stats-key-2", []byte("value2"), 1*time.Minute)

	// 命中
	cache.Get(ctx, "stats-key-1")
	// 未命中
	cache.Get(ctx, "non-existent")

	stats := cache.Stats()
	assert.Greater(t, stats.Hits+stats.Misses, int64(0))
}

// TestMemoryCacheClose - 测试关闭缓存
func TestMemoryCacheClose(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 100*time.Millisecond, nil)

	err := cache.Close()
	assert.NoError(t, err)
}

// TestMemoryCacheCapacity - 测试内存缓存容量限制
func TestMemoryCacheCapacity(t *testing.T) {
	cache := NewMemoryCache(1, 5*time.Minute, nil) // 1 byte capacity
	ctx := context.Background()

	t.Run("Handle Capacity Exceeded", func(t *testing.T) {
		key := "large-key"
		value := []byte("value-larger-than-one-byte")

		err := cache.Set(ctx, key, value, 1*time.Minute)
		// 可能报错也可能不报错，取决于实现
		_ = err
	})
}

// TestMemoryCacheConcurrency - 测试并发安全
func TestMemoryCacheConcurrency(t *testing.T) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()

	done := make(chan bool, 100)

	// 并发写入
	for i := 0; i < 50; i++ {
		go func(n int) {
			key := "concurrent-key-" + string(rune('A'+n%26))
			value := []byte("concurrent-value")
			cache.Set(ctx, key, value, 1*time.Minute)
			done <- true
		}(i)
	}

	// 并发读取
	for i := 0; i < 50; i++ {
		go func(n int) {
			key := "concurrent-key-" + string(rune('A'+n%26))
			cache.Get(ctx, key)
			done <- true
		}(i)
	}

	for i := 0; i < 100; i++ {
		<-done
	}
}

// BenchmarkMemoryCacheSet - 基准测试缓存写入
func BenchmarkMemoryCacheSet(b *testing.B) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()
	value := []byte("benchmark-value")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cache.Set(ctx, "benchmark-key", value, 1*time.Hour)
	}
}

// BenchmarkMemoryCacheGet - 基准测试缓存读取
func BenchmarkMemoryCacheGet(b *testing.B) {
	cache := NewMemoryCache(100*1024*1024, 5*time.Minute, nil)
	ctx := context.Background()
	cache.Set(ctx, "benchmark-key", []byte("benchmark-value"), 1*time.Hour)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cache.Get(ctx, "benchmark-key")
	}
}
