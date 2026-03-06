package logger

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// Config 日志配置
type Config struct {
	Level  string `json:"level" yaml:"level"`   // 日志级别: debug, info, warn, error
	Output string `json:"output" yaml:"output"` // 输出方式: console, file, both
	File   string `json:"file" yaml:"file"`     // 日志文件路径
	Format string `json:"format" yaml:"format"` // 日志格式: text, json
}

var (
	log    *logrus.Logger
	once   sync.Once
	initMu sync.Mutex
)

// Init 初始化日志系统
func Init(cfg Config) {
	initMu.Lock()
	defer initMu.Unlock()

	log = logrus.New()

	// 设置日志级别
	level, err := logrus.ParseLevel(cfg.Level)
	if err != nil {
		level = logrus.InfoLevel
	}
	log.SetLevel(level)

	// 设置日志格式
	if cfg.Format == "json" {
		log.SetFormatter(&logrus.JSONFormatter{
			TimestampFormat: "2006-01-02 15:04:05",
		})
	} else {
		log.SetFormatter(&logrus.TextFormatter{
			FullTimestamp:   true,
			TimestampFormat: "2006-01-02 15:04:05",
			ForceColors:     true,
		})
	}

	// 设置输出目标
	switch strings.ToLower(cfg.Output) {
	case "file":
		if cfg.File != "" {
			file, err := os.OpenFile(cfg.File, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
			if err == nil {
				log.SetOutput(file)
			}
		}
	case "both":
		if cfg.File != "" {
			file, err := os.OpenFile(cfg.File, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
			if err == nil {
				mw := io.MultiWriter(os.Stdout, file)
				log.SetOutput(mw)
			}
		}
	default:
		log.SetOutput(os.Stdout)
	}
}

// getLogger 获取日志实例
func getLogger() *logrus.Logger {
	if log == nil {
		once.Do(func() {
			Init(Config{
				Level:  "info",
				Output: "console",
			})
		})
	}
	return log
}

// Debug 输出debug级别日志
func Debug(format string, args ...interface{}) {
	getLogger().Debugf(format, args...)
}

// Info 输出info级别日志
func Info(format string, args ...interface{}) {
	getLogger().Infof(format, args...)
}

// Warn 输出warn级别日志
func Warn(format string, args ...interface{}) {
	getLogger().Warnf(format, args...)
}

// Error 输出error级别日志
func Error(format string, args ...interface{}) {
	getLogger().Errorf(format, args...)
}

// Fatal 输出fatal级别日志并退出
func Fatal(format string, args ...interface{}) {
	getLogger().Fatalf(format, args...)
}

// WithField 带字段的日志
func WithField(key string, value interface{}) *logrus.Entry {
	return getLogger().WithField(key, value)
}

// WithFields 带多字段的日志
func WithFields(fields map[string]interface{}) *logrus.Entry {
	return getLogger().WithFields(fields)
}

// SetLevel 设置日志级别
func SetLevel(level string) {
	l, err := logrus.ParseLevel(level)
	if err != nil {
		return
	}
	getLogger().SetLevel(l)
}

// GetLevel 获取当前日志级别
func GetLevel() string {
	return getLogger().GetLevel().String()
}

// TrainingLogger 训练专用日志器
type TrainingLogger struct {
	prefix string
	logger *logrus.Logger
}

// NewTrainingLogger 创建训练日志器
func NewTrainingLogger(prefix string) *TrainingLogger {
	return &TrainingLogger{
		prefix: prefix,
		logger: getLogger(),
	}
}

// Log 输出训练日志
func (tl *TrainingLogger) Log(format string, args ...interface{}) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	message := fmt.Sprintf(format, args...)
	fmt.Printf("[%s] [%s] %s\n", timestamp, tl.prefix, message)
}

// LogStep 输出训练步骤日志
func (tl *TrainingLogger) LogStep(epoch, step, totalSteps int, loss, lr float64) {
	tl.Log("Epoch:[%d](%d/%d), loss: %.4f, lr: %.8f", epoch+1, step, totalSteps, loss, lr)
}

// LogEpoch 输出epoch完成日志
func (tl *TrainingLogger) LogEpoch(epoch, totalEpochs int, avgLoss float64, duration time.Duration) {
	tl.Log("Epoch %d/%d 完成, avg_loss: %.4f, 用时: %s", epoch+1, totalEpochs, avgLoss, duration)
}
