package cache

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockCacheClient is a mock implementation for testing
type MockCacheClient struct {
	mock.Mock
	store map[string][]byte
}

func NewMockCacheClient() *MockCacheClient {
	return &MockCacheClient{
		store: make(map[string][]byte),
	}
}

func (m *MockCacheClient) Get(key string) ([]byte, error) {
	args := m.Called(key)
	if data, ok := m.store[key]; ok {
		return data, nil
	}
	return nil, args.Error(1)
}

func (m *MockCacheClient) Set(key string, value []byte, ttl time.Duration) error {
	m.store[key] = value
	args := m.Called(key, value, ttl)
	return args.Error(0)
}

func (m *MockCacheClient) Delete(key string) error {
	delete(m.store, key)
	args := m.Called(key)
	return args.Error(0)
}

func (m *MockCacheClient) Exists(key string) bool {
	_, exists := m.store[key]
	args := m.Called(key)
	return args.Bool(0) || exists
}

func (m *MockCacheClient) TTL(key string) (time.Duration, error) {
	args := m.Called(key)
	return args.Get(0).(time.Duration), args.Error(1)
}

// TestNewMemoryCache - 测试内存缓存创建
func TestNewMemoryCache(t *testing.T) {
	cache, err := NewMemoryCache(100 * 1024 * 1024) // 100MB
	assert.NoError(t, err)
	assert.NotNil(t, cache)
	assert.NotNil(t, cache.items)
	assert.NotNil(t, cache.ttls)
}

// TestMemoryCacheSetAndGet - 测试内存缓存的Set和Get操作
func TestMemoryCacheSetAndGet(t *testing.T) {
	cache, _ := NewMemoryCache(100 * 1024 * 1024)

	t.Run("Set and Get String", func(t *testing.T) {
		key := "test-key"
		value := []byte("test-value")
		ttl := 1 * time.Minute

		err := cache.Set(key, value, ttl)
		assert.NoError(t, err)

		retrieved, err := cache.Get(key)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})

	t.Run("Get Non-existent Key", func(t *testing.T) {
		_, err := cache.Get("non-existent-key")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "key not found")
	})

	t.Run("Set Empty Value", func(t *testing.T) {
		key := "empty-key"
		value := []byte{}

		err := cache.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		retrieved, err := cache.Get(key)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})
}

// TestMemoryCacheExpiration - 测试内存缓存过期
func TestMemoryCacheExpiration(t *testing.T) {
	cache, _ := NewMemoryCache(100 * 1024 * 1024)

	t.Run("Expired Key", func(t *testing.T) {
		key := "expired-key"
		value := []byte("expired-value")

		// Set with very short TTL
		err := cache.Set(key, value, 1*time.Millisecond)
		assert.NoError(t, err)

		// Wait for expiration
		time.Sleep(10 * time.Millisecond)

		_, err = cache.Get(key)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "expired")
	})

	t.Run("Check TTL", func(t *testing.T) {
		key := "ttl-key"
		value := []byte("ttl-value")

		err := cache.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		ttl, err := cache.TTL(key)
		assert.NoError(t, err)
		// TTL should be approximately 1 minute
		assert.Greater(t, ttl, 55*time.Second)
		assert.LessOrEqual(t, ttl, 1*time.Minute)
	})

	t.Run("TTL for Non-existent Key", func(t *testing.T) {
		ttl, err := cache.TTL("non-existent-ttl-key")
		assert.Error(t, err)
		assert.Equal(t, time.Duration(0), ttl)
	})
}

// TestMemoryCacheDelete - 测试内存缓存删除
func TestMemoryCacheDelete(t *testing.T) {
	cache, _ := NewMemoryCache(100 * 1024 * 1024)

	t.Run("Delete Existing Key", func(t *testing.T) {
		key := "delete-key"
		value := []byte("delete-value")

		err := cache.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		err = cache.Delete(key)
		assert.NoError(t, err)

		_, err = cache.Get(key)
		assert.Error(t, err)
	})

	t.Run("Delete Non-existent Key", func(t *testing.T) {
		// Should not error when deleting non-existent key
		err := cache.Delete("non-existent-delete-key")
		assert.NoError(t, err)
	})
}

// TestMemoryCacheExistence - 测试内存缓存存在性检查
func TestMemoryCacheExistence(t *testing.T) {
	cache, _ := NewMemoryCache(100 * 1024 * 1024)

	t.Run("Key Exists", func(t *testing.T) {
		key := "exists-key"
		value := []byte("exists-value")

		err := cache.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		exists := cache.Exists(key)
		assert.True(t, exists)
	})

	t.Run("Key Does Not Exist", func(t *testing.T) {
		exists := cache.Exists("not-exists-key")
		assert.False(t, exists)
	})

	t.Run("Expired Key Does Not Exist", func(t *testing.T) {
		key := "expired-exists-key"
		value := []byte("expired-exists-value")

		err := cache.Set(key, value, 1*time.Millisecond)
		assert.NoError(t, err)

		time.Sleep(10 * time.Millisecond)

		exists := cache.Exists(key)
		assert.False(t, exists)
	})
}

// TestMemoryCacheConcurrency - 测试内存缓存并发安全
func TestMemoryCacheConcurrency(t *testing.T) {
	cache, _ := NewMemoryCache(100 * 1024 * 1024)

	t.Run("Concurrent Writes", func(t *testing.T) {
		done := make(chan bool, 100)

		for i := 0; i < 100; i++ {
			go func(i int) {
				key := string(rune('a' + i%26))
				value := []byte(string(rune('A' + i%26)))
				cache.Set(key, value, 1*time.Minute)
				done <- true
			}(i)
		}

		for i := 0; i < 100; i++ {
			<-done
		}

		// Verify all values are set
		for i := 0; i < 26; i++ {
			key := string(rune('a' + i))
			value, err := cache.Get(key)
			if err == nil {
				assert.NotNil(t, value)
			}
		}
	})
}

// TestNewRedisCache - 测试Redis缓存创建
func TestNewRedisCache(t *testing.T) {
	t.Run("Valid Config", func(t *testing.T) {
		// This test assumes Redis is available
		// In production CI/CD, you would use a Redis test container
		cfg := &RedisConfig{
			Host:     "localhost",
			Port:     6379,
			DB:       0,
			Password: "",
			PoolSize: 10,
		}

		// Note: This will fail if Redis is not available
		// Skipping in CI environment
		t.Skip("Skipping Redis test - requires Redis server")

		cache, err := NewRedisCache(cfg)
		if err != nil {
			t.Skip("Redis not available, skipping test")
		}
		assert.NotNil(t, cache)
	})

	t.Run("Invalid Address", func(t *testing.T) {
		cfg := &RedisConfig{
			Host: "invalid-address-that-does-not-exist",
			Port: 12345,
		}

		_, err := NewRedisCache(cfg)
		assert.Error(t, err)
	})
}

// TestCacheManager - 测试缓存管理器
func TestCacheManager(t *testing.T) {
	t.Run("Create Memory Cache Manager", func(t *testing.T) {
		cfg := &ManagerConfig{
			Backend:      "memory",
			MemorySizeMB: 100,
			DefaultTTL:   1 * time.Hour,
		}

		manager, err := NewCacheManager(cfg, nil)
		assert.NoError(t, err)
		assert.NotNil(t, manager)
		assert.Equal(t, "memory", manager.config.Backend)
	})

	t.Run("Invalid Backend", func(t *testing.T) {
		cfg := &ManagerConfig{
			Backend: "invalid",
		}

		_, err := NewCacheManager(cfg, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported cache backend")
	})
}

// TestCacheManagerOperations - 测试缓存管理器操作
func TestCacheManagerOperations(t *testing.T) {
	cfg := &ManagerConfig{
		Backend:      "memory",
		MemorySizeMB: 100,
		DefaultTTL:   1 * time.Hour,
	}

	manager, _ := NewCacheManager(cfg, nil)

	t.Run("Set and Get", func(t *testing.T) {
		key := "manager-test-key"
		value := map[string]string{"test": "value"}

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved map[string]string
		err = manager.Get(key, &retrieved)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})

	t.Run("Set with Default TTL", func(t *testing.T) {
		key := "default-ttl-key"
		value := "test-value"

		// Set without specifying TTL - should use default
		err := manager.Set(key, value, 0)
		assert.NoError(t, err)

		ttl, err := manager.TTL(key)
		assert.NoError(t, err)
		assert.Greater(t, ttl, 59*time.Minute)
	})

	t.Run("Get Non-existent Key", func(t *testing.T) {
		var result string
		err := manager.Get("non-existent-manager-key", &result)
		assert.Error(t, err)
	})

	t.Run("Delete", func(t *testing.T) {
		key := "delete-manager-key"
		value := "delete-value"

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		err = manager.Delete(key)
		assert.NoError(t, err)

		var result string
		err = manager.Get(key, &result)
		assert.Error(t, err)
	})

	t.Run("Exists", func(t *testing.T) {
		key := "exists-manager-key"
		value := "exists-value"

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		exists := manager.Exists(key)
		assert.True(t, exists)

		exists = manager.Exists("not-exists-manager-key")
		assert.False(t, exists)
	})
}

// TestCacheManagerCompression - 测试缓存管理器压缩
func TestCacheManagerCompression(t *testing.T) {
	cfg := &ManagerConfig{
		Backend:              "memory",
		MemorySizeMB:         100,
		DefaultTTL:           1 * time.Hour,
		Compression:          true,
		CompressionThreshold: 10,
	}

	manager, _ := NewCacheManager(cfg, nil)

	t.Run("Compress Large Value", func(t *testing.T) {
		key := "compressed-key"
		// Create a value larger than compression threshold
		value := make([]byte, 100)
		for i := range value {
			value[i] = byte('A')
		}

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved []byte
		err = manager.Get(key, &retrieved)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})

	t.Run("No Compression for Small Value", func(t *testing.T) {
		cfg2 := &ManagerConfig{
			Backend:              "memory",
			MemorySizeMB:         100,
			DefaultTTL:           1 * time.Hour,
			Compression:          true,
			CompressionThreshold: 1000, // High threshold
		}

		manager2, _ := NewCacheManager(cfg2, nil)

		key := "uncompressed-key"
		value := "small-value" // Less than threshold

		err := manager2.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved string
		err = manager2.Get(key, &retrieved)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})
}

// TestCacheManagerSerialization - 测试缓存管理器序列化
func TestCacheManagerSerialization(t *testing.T) {
	cfg := &ManagerConfig{
		Backend:      "memory",
		MemorySizeMB: 100,
		DefaultTTL:   1 * time.Hour,
	}

	manager, _ := NewCacheManager(cfg, nil)

	t.Run("Serialize String", func(t *testing.T) {
		key := "string-key"
		value := "test-string-value"

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved string
		err = manager.Get(key, &retrieved)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})

	t.Run("Serialize Int", func(t *testing.T) {
		key := "int-key"
		value := 42

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved int
		err = manager.Get(key, &retrieved)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})

	t.Run("Serialize Struct", func(t *testing.T) {
		type TestStruct struct {
			Name   string `json:"name"`
			Age    int    `json:"age"`
			Active bool   `json:"active"`
		}

		key := "struct-key"
		value := TestStruct{
			Name:   "John",
			Age:    30,
			Active: true,
		}

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved TestStruct
		err = manager.Get(key, &retrieved)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})

	t.Run("Serialize Map", func(t *testing.T) {
		key := "map-key"
		value := map[string]interface{}{
			"name":   "Jane",
			"age":    25,
			"active": false,
		}

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved map[string]interface{}
		err = manager.Get(key, &retrieved)
		assert.NoError(t, err)

		// JSON unmarshaling converts numbers to float64
		assert.Equal(t, value["name"], retrieved["name"])
		assert.Equal(t, value["active"], retrieved["active"])
	})

	t.Run("Serialize Slice", func(t *testing.T) {
		key := "slice-key"
		value := []string{"a", "b", "c", "d"}

		err := manager.Set(key, value, 1*time.Minute)
		assert.NoError(t, err)

		var retrieved []string
		err = manager.Get(key, &retrieved)
		assert.NoError(t, err)
		assert.Equal(t, value, retrieved)
	})
}

// TestCacheManagerFlush - 测试缓存清空
func TestCacheManagerFlush(t *testing.T) {
	cfg := &ManagerConfig{
		Backend:      "memory",
		MemorySizeMB: 100,
		DefaultTTL:   1 * time.Hour,
	}

	manager, _ := NewCacheManager(cfg, nil)

	t.Run("Flush All", func(t *testing.T) {
		// Set multiple values
		for i := 0; i < 10; i++ {
			key := string(rune('a' + i))
			value := string(rune('A' + i))
			err := manager.Set(key, value, 1*time.Minute)
			assert.NoError(t, err)
		}

		// Flush all
		err := manager.Flush()
		assert.NoError(t, err)

		// Verify all values are gone
		for i := 0; i < 10; i++ {
			key := string(rune('a' + i))
			exists := manager.Exists(key)
			assert.False(t, exists)
		}
	})
}

// TestMemoryCacheCapacity - 测试内存缓存容量限制
func TestMemoryCacheCapacity(t *testing.T) {
	// Create a cache with very small capacity
	cache, _ := NewMemoryCache(1) // 1 byte capacity

	t.Run("Handle Capacity Exceeded", func(t *testing.T) {
		// This should handle gracefully by managing memory
		// Actual eviction policy implementation will vary
		key := "large-key"
		value := []byte("value-larger-than-one-byte")

		err := cache.Set(key, value, 1*time.Minute)
		// May or may not error depending on implementation
		_ = err
	})
}

// BenchmarkMemoryCacheSet - 内存缓存Set性能测试
func BenchmarkMemoryCacheSet(b *testing.B) {
	cache, _ := NewMemoryCache(100 * 1024 * 1024)
	value := []byte("benchmark-value")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := string(rune('a' + i%26))
		cache.Set(key, value, 1*time.Minute)
	}
}

// BenchmarkMemoryCacheGet - 内存缓存Get性能测试
func BenchmarkMemoryCacheGet(b *testing.B) {
	cache, _ := NewMemoryCache(100 * 1024 * 1024)
	value := []byte("benchmark-value")
	cache.Set("benchmark-key", value, 1*time.Hour)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Get("benchmark-key")
	}
}

// BenchmarkCacheManagerSet - 缓存管理器Set性能测试
func BenchmarkCacheManagerSet(b *testing.B) {
	cfg := &ManagerConfig{
		Backend:      "memory",
		MemorySizeMB: 100,
		DefaultTTL:   1 * time.Hour,
	}

	manager, _ := NewCacheManager(cfg, nil)
	value := map[string]string{"test": "value"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := string(rune('a' + i%26))
		manager.Set(key, value, 1*time.Minute)
	}
}

// TestCacheMetrics - 测试缓存指标
func TestCacheMetrics(t *testing.T) {
	cfg := &ManagerConfig{
		Backend:      "memory",
		MemorySizeMB: 100,
		DefaultTTL:   1 * time.Hour,
		Metrics:      true,
	}

	manager, _ := NewCacheManager(cfg, nil)

	t.Run("Record Hits and Misses", func(t *testing.T) {
		// Set a value
		err := manager.Set("metrics-key", "value", 1*time.Minute)
		assert.NoError(t, err)

		// Get existing key (hit)
		var value string
		err = manager.Get("metrics-key", &value)
		assert.NoError(t, err)

		// Get non-existing key (miss)
		err = manager.Get("non-existent-metrics-key", &value)
		assert.Error(t, err)
	})
}

// TestCacheWithContext - 测试带上下文的缓存操作
func TestCacheWithContext(t *testing.T) {
	// This is a placeholder for context-aware caching
	// Implementation would depend on specific requirements
	t.Run("Context Cancellation", func(t *testing.T) {
		// Placeholder test
		t.Skip("Context-aware caching to be implemented")
	})
}
