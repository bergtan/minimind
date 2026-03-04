package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/sirupsen/logrus"
)

// ========== 缓存接口定义 ==========

// Cache 缓存接口
type Cache interface {
	// Get 获取缓存值
	Get(ctx context.Context, key string) ([]byte, error)

	// Set 设置缓存值
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// Delete 删除缓存值
	Delete(ctx context.Context, key string) error

	// Exists 检查缓存是否存在
	Exists(ctx context.Context, key string) (bool, error)

	// Flush 清空缓存
	Flush(ctx context.Context) error

	// Stats 获取缓存统计信息
	Stats() CacheStats

	// Close 关闭缓存连接
	Close() error
}

// CacheStats 缓存统计信息
type CacheStats struct {
	Hits    int64 `json:"hits"`
	Misses  int64 `json:"misses"`
	Sets    int64 `json:"sets"`
	Deletes int64 `json:"deletes"`
	Errors  int64 `json:"errors"`
	Size    int64 `json:"size"`
	MaxSize int64 `json:"max_size"`
}

// ========== 内存缓存实现 ==========

// MemoryCache 内存缓存
type MemoryCache struct {
	data    map[string]*cacheItem
	mutex   sync.RWMutex
	stats   CacheStats
	maxSize int64
	cleanup time.Duration
	logger  *logrus.Logger
}

// cacheItem 缓存项
type cacheItem struct {
	Value     []byte    `json:"value"`
	ExpiresAt time.Time `json:"expires_at"`
	Size      int64     `json:"size"`
}

// NewMemoryCache 创建内存缓存
func NewMemoryCache(maxSize int64, cleanupInterval time.Duration, logger *logrus.Logger) *MemoryCache {
	cache := &MemoryCache{
		data:    make(map[string]*cacheItem),
		maxSize: maxSize,
		cleanup: cleanupInterval,
		logger:  logger,
		stats:   CacheStats{MaxSize: maxSize},
	}

	// 启动清理goroutine
	if cleanupInterval > 0 {
		go cache.startCleanup()
	}

	return cache
}

// Get 获取缓存值
func (mc *MemoryCache) Get(ctx context.Context, key string) ([]byte, error) {
	mc.mutex.RLock()
	item, exists := mc.data[key]
	mc.mutex.RUnlock()

	if !exists {
		mc.stats.Misses++
		return nil, nil
	}

	// 检查是否过期
	if time.Now().After(item.ExpiresAt) {
		mc.mutex.Lock()
		delete(mc.data, key)
		mc.stats.Size -= item.Size
		mc.mutex.Unlock()

		mc.stats.Misses++
		return nil, nil
	}

	mc.stats.Hits++
	return item.Value, nil
}

// Set 设置缓存值
func (mc *MemoryCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	if ttl <= 0 {
		return fmt.Errorf("TTL must be positive")
	}

	expiresAt := time.Now().Add(ttl)
	size := int64(len(value))

	// 检查缓存大小限制
	if mc.stats.Size+size > mc.maxSize {
		mc.logger.Warn("Cache size limit exceeded, performing cleanup")
		mc.performCleanup()

		// 再次检查
		if mc.stats.Size+size > mc.maxSize {
			return fmt.Errorf("cache size limit exceeded even after cleanup")
		}
	}

	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	// 如果键已存在，先删除旧值
	if oldItem, exists := mc.data[key]; exists {
		mc.stats.Size -= oldItem.Size
	}

	mc.data[key] = &cacheItem{
		Value:     value,
		ExpiresAt: expiresAt,
		Size:      size,
	}

	mc.stats.Size += size
	mc.stats.Sets++

	return nil
}

// Delete 删除缓存值
func (mc *MemoryCache) Delete(ctx context.Context, key string) error {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if item, exists := mc.data[key]; exists {
		mc.stats.Size -= item.Size
		delete(mc.data, key)
		mc.stats.Deletes++
	}

	return nil
}

// Exists 检查缓存是否存在
func (mc *MemoryCache) Exists(ctx context.Context, key string) (bool, error) {
	mc.mutex.RLock()
	item, exists := mc.data[key]
	mc.mutex.RUnlock()

	if !exists {
		return false, nil
	}

	// 检查是否过期
	if time.Now().After(item.ExpiresAt) {
		mc.mutex.Lock()
		delete(mc.data, key)
		mc.stats.Size -= item.Size
		mc.mutex.Unlock()
		return false, nil
	}

	return true, nil
}

// Flush 清空缓存
func (mc *MemoryCache) Flush(ctx context.Context) error {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	mc.data = make(map[string]*cacheItem)
	mc.stats.Size = 0

	mc.logger.Info("Memory cache flushed")

	return nil
}

// Stats 获取缓存统计信息
func (mc *MemoryCache) Stats() CacheStats {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	return mc.stats
}

// Close 关闭缓存连接
func (mc *MemoryCache) Close() error {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	mc.data = nil
	mc.logger.Info("Memory cache closed")

	return nil
}

// startCleanup 启动清理goroutine
func (mc *MemoryCache) startCleanup() {
	ticker := time.NewTicker(mc.cleanup)
	defer ticker.Stop()

	for range ticker.C {
		mc.performCleanup()
	}
}

// performCleanup 执行清理操作
func (mc *MemoryCache) performCleanup() {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	now := time.Now()
	cleaned := 0

	for key, item := range mc.data {
		if now.After(item.ExpiresAt) {
			mc.stats.Size -= item.Size
			delete(mc.data, key)
			cleaned++
		}
	}

	if cleaned > 0 {
		mc.logger.Infof("Cleaned up %d expired cache items", cleaned)
	}
}

// ========== Redis缓存实现 ==========

// RedisCache Redis缓存
type RedisCache struct {
	client *redis.Client
	prefix string
	stats  CacheStats
	logger *logrus.Logger
}

// NewRedisCache 创建Redis缓存
func NewRedisCache(addr, password string, db int, prefix string, logger *logrus.Logger) *RedisCache {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	return &RedisCache{
		client: client,
		prefix: prefix,
		logger: logger,
		stats:  CacheStats{MaxSize: -1}, // Redis没有固定大小限制
	}
}

// Get 获取缓存值
func (rc *RedisCache) Get(ctx context.Context, key string) ([]byte, error) {
	fullKey := rc.getFullKey(key)

	value, err := rc.client.Get(ctx, fullKey).Bytes()
	if err == redis.Nil {
		rc.stats.Misses++
		return nil, nil
	} else if err != nil {
		rc.stats.Errors++
		return nil, fmt.Errorf("failed to get cache value: %w", err)
	}

	rc.stats.Hits++
	return value, nil
}

// Set 设置缓存值
func (rc *RedisCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	fullKey := rc.getFullKey(key)

	err := rc.client.Set(ctx, fullKey, value, ttl).Err()
	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("failed to set cache value: %w", err)
	}

	rc.stats.Sets++
	return nil
}

// Delete 删除缓存值
func (rc *RedisCache) Delete(ctx context.Context, key string) error {
	fullKey := rc.getFullKey(key)

	err := rc.client.Del(ctx, fullKey).Err()
	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("failed to delete cache value: %w", err)
	}

	rc.stats.Deletes++
	return nil
}

// Exists 检查缓存是否存在
func (rc *RedisCache) Exists(ctx context.Context, key string) (bool, error) {
	fullKey := rc.getFullKey(key)

	exists, err := rc.client.Exists(ctx, fullKey).Result()
	if err != nil {
		rc.stats.Errors++
		return false, fmt.Errorf("failed to check cache existence: %w", err)
	}

	return exists > 0, nil
}

// Flush 清空缓存
func (rc *RedisCache) Flush(ctx context.Context) error {
	pattern := rc.prefix + "*"

	keys, err := rc.client.Keys(ctx, pattern).Result()
	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("failed to get cache keys: %w", err)
	}

	if len(keys) > 0 {
		err = rc.client.Del(ctx, keys...).Err()
		if err != nil {
			rc.stats.Errors++
			return fmt.Errorf("failed to delete cache keys: %w", err)
		}
	}

	rc.logger.Infof("Redis cache flushed, deleted %d keys", len(keys))

	return nil
}

// Stats 获取缓存统计信息
func (rc *RedisCache) Stats() CacheStats {
	return rc.stats
}

// Close 关闭缓存连接
func (rc *RedisCache) Close() error {
	if rc.client != nil {
		err := rc.client.Close()
		if err != nil {
			return fmt.Errorf("failed to close Redis client: %w", err)
		}

		rc.logger.Info("Redis cache connection closed")
	}

	return nil
}

// getFullKey 获取完整键名
func (rc *RedisCache) getFullKey(key string) string {
	return rc.prefix + ":" + key
}

// ========== 缓存管理器 ==========

// CacheManager 缓存管理器
type CacheManager struct {
	caches     map[string]Cache
	defaultTTL time.Duration
	logger     *logrus.Logger
	statsMutex sync.RWMutex
	stats      map[string]CacheStats
}

// NewCacheManager 创建缓存管理器
func NewCacheManager(defaultTTL time.Duration, logger *logrus.Logger) *CacheManager {
	return &CacheManager{
		caches:     make(map[string]Cache),
		defaultTTL: defaultTTL,
		logger:     logger,
		stats:      make(map[string]CacheStats),
	}
}

// AddCache 添加缓存实例
func (cm *CacheManager) AddCache(name string, cache Cache) error {
	if _, exists := cm.caches[name]; exists {
		return fmt.Errorf("cache '%s' already exists", name)
	}

	cm.caches[name] = cache
	cm.logger.Infof("Cache '%s' added", name)

	return nil
}

// GetCache 获取缓存实例
func (cm *CacheManager) GetCache(name string) (Cache, error) {
	cache, exists := cm.caches[name]
	if !exists {
		return nil, fmt.Errorf("cache '%s' not found", name)
	}

	return cache, nil
}

// RemoveCache 移除缓存实例
func (cm *CacheManager) RemoveCache(name string) error {
	cache, exists := cm.caches[name]
	if !exists {
		return fmt.Errorf("cache '%s' not found", name)
	}

	// 关闭缓存连接
	if err := cache.Close(); err != nil {
		cm.logger.Errorf("Failed to close cache '%s': %v", name, err)
	}

	delete(cm.caches, name)
	cm.logger.Infof("Cache '%s' removed", name)

	return nil
}

// Get 从默认缓存获取值
func (cm *CacheManager) Get(ctx context.Context, key string) ([]byte, error) {
	return cm.GetFrom(ctx, "default", key)
}

// Set 设置默认缓存值
func (cm *CacheManager) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	return cm.SetTo(ctx, "default", key, value, ttl)
}

// GetFrom 从指定缓存获取值
func (cm *CacheManager) GetFrom(ctx context.Context, cacheName, key string) ([]byte, error) {
	cache, err := cm.GetCache(cacheName)
	if err != nil {
		return nil, err
	}

	return cache.Get(ctx, key)
}

// SetTo 设置指定缓存值
func (cm *CacheManager) SetTo(ctx context.Context, cacheName, key string, value []byte, ttl time.Duration) error {
	if ttl <= 0 {
		ttl = cm.defaultTTL
	}

	cache, err := cm.GetCache(cacheName)
	if err != nil {
		return err
	}

	return cache.Set(ctx, key, value, ttl)
}

// Delete 删除缓存值
func (cm *CacheManager) Delete(ctx context.Context, key string) error {
	return cm.DeleteFrom(ctx, "default", key)
}

// DeleteFrom 从指定缓存删除值
func (cm *CacheManager) DeleteFrom(ctx context.Context, cacheName, key string) error {
	cache, err := cm.GetCache(cacheName)
	if err != nil {
		return err
	}

	return cache.Delete(ctx, key)
}

// Exists 检查缓存是否存在
func (cm *CacheManager) Exists(ctx context.Context, key string) (bool, error) {
	return cm.ExistsIn(ctx, "default", key)
}

// ExistsIn 在指定缓存中检查是否存在
func (cm *CacheManager) ExistsIn(ctx context.Context, cacheName, key string) (bool, error) {
	cache, err := cm.GetCache(cacheName)
	if err != nil {
		return false, err
	}

	return cache.Exists(ctx, key)
}

// Flush 清空所有缓存
func (cm *CacheManager) Flush(ctx context.Context) error {
	for name, cache := range cm.caches {
		if err := cache.Flush(ctx); err != nil {
			cm.logger.Errorf("Failed to flush cache '%s': %v", name, err)
			return err
		}
	}

	cm.logger.Info("All caches flushed")

	return nil
}

// GetStats 获取所有缓存统计信息
func (cm *CacheManager) GetStats() map[string]CacheStats {
	cm.statsMutex.RLock()
	defer cm.statsMutex.RUnlock()

	stats := make(map[string]CacheStats)

	for name, cache := range cm.caches {
		stats[name] = cache.Stats()
	}

	return stats
}

// UpdateStats 更新统计信息
func (cm *CacheManager) UpdateStats() {
	cm.statsMutex.Lock()
	defer cm.statsMutex.Unlock()

	for name, cache := range cm.caches {
		cm.stats[name] = cache.Stats()
	}
}

// Close 关闭所有缓存连接
func (cm *CacheManager) Close() error {
	for name, cache := range cm.caches {
		if err := cache.Close(); err != nil {
			cm.logger.Errorf("Failed to close cache '%s': %v", name, err)
			return err
		}
	}

	cm.logger.Info("All cache connections closed")

	return nil
}

// ========== 缓存工具函数 ==========

// GenerateCacheKey 生成缓存键
func GenerateCacheKey(prefix string, params ...interface{}) string {
	key := prefix

	for _, param := range params {
		key += fmt.Sprintf(":%v", param)
	}

	// 使用SHA256哈希确保键长度一致
	hash := sha256.Sum256([]byte(key))
	return fmt.Sprintf("%x", hash[:16]) // 取前16字节
}

// Serialize 序列化对象
func Serialize(obj interface{}) ([]byte, error) {
	return json.Marshal(obj)
}

// Deserialize 反序列化对象
func Deserialize(data []byte, obj interface{}) error {
	return json.Unmarshal(data, obj)
}

// CacheWithFallback 带回退的缓存获取
func CacheWithFallback(ctx context.Context, cache Cache, key string, ttl time.Duration, fallback func() ([]byte, error)) ([]byte, error) {
	// 尝试从缓存获取
	cached, err := cache.Get(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("failed to get from cache: %w", err)
	}

	if cached != nil {
		return cached, nil
	}

	// 缓存未命中，执行回退函数
	value, err := fallback()
	if err != nil {
		return nil, fmt.Errorf("fallback function failed: %w", err)
	}

	// 将结果存入缓存
	if err := cache.Set(ctx, key, value, ttl); err != nil {
		// 记录错误但不中断流程
		fmt.Printf("Failed to set cache: %v\n", err)
	}

	return value, nil
}

// ========== 缓存配置 ==========

// CacheConfig 缓存配置
type CacheConfig struct {
	Type       string        `json:"type"`        // memory, redis
	Enabled    bool          `json:"enabled"`     // 是否启用缓存
	DefaultTTL time.Duration `json:"default_ttl"` // 默认TTL

	// 内存缓存配置
	MemoryConfig struct {
		MaxSize         int64         `json:"max_size"`         // 最大缓存大小（字节）
		CleanupInterval time.Duration `json:"cleanup_interval"` // 清理间隔
	} `json:"memory_config"`

	// Redis缓存配置
	RedisConfig struct {
		Addr     string `json:"addr"`     // Redis地址
		Password string `json:"password"` // Redis密码
		DB       int    `json:"db"`       // Redis数据库
		Prefix   string `json:"prefix"`   // 键前缀
	} `json:"redis_config"`
}

// NewCacheFromConfig 根据配置创建缓存
func NewCacheFromConfig(config *CacheConfig, logger *logrus.Logger) (Cache, error) {
	if !config.Enabled {
		return nil, fmt.Errorf("cache is disabled")
	}

	switch config.Type {
	case "memory":
		return NewMemoryCache(
			config.MemoryConfig.MaxSize,
			config.MemoryConfig.CleanupInterval,
			logger,
		), nil

	case "redis":
		return NewRedisCache(
			config.RedisConfig.Addr,
			config.RedisConfig.Password,
			config.RedisConfig.DB,
			config.RedisConfig.Prefix,
			logger,
		), nil

	default:
		return nil, fmt.Errorf("unsupported cache type: %s", config.Type)
	}
}

// ========== 缓存中间件 ==========

// CacheMiddleware 缓存中间件
func CacheMiddleware(cache Cache, ttl time.Duration, keyGenerator func(*gin.Context) string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 只缓存GET请求
		if c.Request.Method != "GET" {
			c.Next()
			return
		}

		// 生成缓存键
		key := keyGenerator(c)

		// 尝试从缓存获取
		cached, err := cache.Get(c.Request.Context(), key)
		if err != nil {
			c.Next()
			return
		}

		if cached != nil {
			// 返回缓存响应
			c.Data(200, "application/json", cached)
			c.Abort()
			return
		}

		// 缓存未命中，继续处理
		c.Next()

		// 如果响应成功，缓存结果
		if c.Writer.Status() == 200 {
			// 获取响应体
			if c.Writer.Size() > 0 {
				// 这里需要拦截响应体，但Gin不支持直接获取
				// 实际应用中可能需要自定义ResponseWriter
			}
		}
	}
}

// DefaultCacheKeyGenerator 默认缓存键生成器
func DefaultCacheKeyGenerator(c *gin.Context) string {
	return GenerateCacheKey(
		"api",
		c.Request.Method,
		c.Request.URL.Path,
		c.Request.URL.Query().Encode(),
	)
}
