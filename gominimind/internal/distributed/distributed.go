package distributed

import (
	"encoding/gob"
	"fmt"
	"net"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Backend 分布式后端类型
type Backend string

const (
	BackendGloo Backend = "gloo"
	BackendNCCL Backend = "nccl"
	BackendMPI  Backend = "mpi"
	BackendAuto Backend = "auto"
)

// DistConfig 分布式配置
type DistConfig struct {
	Backend      Backend
	WorldSize    int
	Rank         int
	LocalRank    int
	MasterAddr   string
	MasterPort   int
	Timeout      time.Duration
	UseAllReduce bool
	UseBroadcast bool
	UseReduce    bool
	UseGather    bool
	UseScatter   bool
}

// DefaultDistConfig 返回默认分布式配置
func DefaultDistConfig() *DistConfig {
	return &DistConfig{
		Backend:      BackendGloo,
		WorldSize:    1,
		Rank:         0,
		LocalRank:    0,
		MasterAddr:   "localhost",
		MasterPort:   29500,
		Timeout:      30 * time.Minute,
		UseAllReduce: true,
		UseBroadcast: true,
		UseReduce:    true,
		UseGather:    true,
		UseScatter:   true,
	}
}

// ProcessGroup 进程组
type ProcessGroup struct {
	config    *DistConfig
	members   []int
	rank      int
	worldSize int

	connections map[int]net.Conn
	listener    net.Listener
	mu          sync.RWMutex

	isInitialized bool
}

// NewProcessGroup 创建新的进程组
func NewProcessGroup(config *DistConfig, ranks []int) (*ProcessGroup, error) {
	if config == nil {
		config = DefaultDistConfig()
	}

	pg := &ProcessGroup{
		config:      config,
		members:     ranks,
		rank:        config.Rank,
		worldSize:   config.WorldSize,
		connections: make(map[int]net.Conn),
	}

	if len(ranks) == 0 {
		for i := 0; i < config.WorldSize; i++ {
			pg.members = append(pg.members, i)
		}
	}

	return pg, nil
}

// Initialize 初始化进程组
func (pg *ProcessGroup) Initialize() error {
	if pg.isInitialized {
		return nil
	}

	pg.mu.Lock()
	defer pg.mu.Unlock()

	// 启动监听
	if pg.rank == 0 {
		// Master节点
		addr := fmt.Sprintf("%s:%d", pg.config.MasterAddr, pg.config.MasterPort)
		listener, err := net.Listen("tcp", addr)
		if err != nil {
			return fmt.Errorf("failed to listen on %s: %w", addr, err)
		}
		pg.listener = listener

		// 等待所有worker连接
		go pg.acceptConnections()
	} else {
		// Worker节点，连接Master
		if err := pg.connectToMaster(); err != nil {
			return err
		}
	}

	pg.isInitialized = true
	return nil
}

// acceptConnections 接受连接
func (pg *ProcessGroup) acceptConnections() {
	for len(pg.connections) < pg.worldSize-1 {
		conn, err := pg.listener.Accept()
		if err != nil {
			continue
		}

		// 处理连接握手
		go pg.handleConnection(conn)
	}
}

// handleConnection 处理连接
func (pg *ProcessGroup) handleConnection(conn net.Conn) {
	// 读取对端rank
	decoder := gob.NewDecoder(conn)
	var rank int
	if err := decoder.Decode(&rank); err != nil {
		conn.Close()
		return
	}

	pg.mu.Lock()
	pg.connections[rank] = conn
	pg.mu.Unlock()
}

// connectToMaster 连接到Master
func (pg *ProcessGroup) connectToMaster() error {
	addr := net.JoinHostPort(pg.config.MasterAddr, fmt.Sprintf("%d", pg.config.MasterPort))

	var conn net.Conn
	var err error

	// 重试连接
	for i := 0; i < 30; i++ {
		conn, err = net.Dial("tcp", addr)
		if err == nil {
			break
		}
		time.Sleep(time.Second)
	}

	if err != nil {
		return fmt.Errorf("failed to connect to master: %w", err)
	}

	// 发送本机rank
	encoder := gob.NewEncoder(conn)
	if err := encoder.Encode(pg.rank); err != nil {
		conn.Close()
		return err
	}

	pg.mu.Lock()
	pg.connections[0] = conn
	pg.mu.Unlock()

	return nil
}

// Destroy 销毁进程组
func (pg *ProcessGroup) Destroy() error {
	pg.mu.Lock()
	defer pg.mu.Unlock()

	for _, conn := range pg.connections {
		conn.Close()
	}

	if pg.listener != nil {
		pg.listener.Close()
	}

	pg.isInitialized = false
	return nil
}

// ==================== 通信原语 ====================

// AllReduce 全归约操作
func (pg *ProcessGroup) AllReduce(sendBuf, recvBuf *mat.VecDense, op ReduceOp) error {
	if !pg.isInitialized {
		return fmt.Errorf("process group not initialized")
	}

	pg.mu.RLock()
	defer pg.mu.RUnlock()

	// 简化的AllReduce实现（Ring-AllReduce算法）
	data := sendBuf.RawVector().Data
	result := make([]float64, len(data))
	copy(result, data)

	// 环形传递和累加
	nextRank := (pg.rank + 1) % pg.worldSize
	prevRank := (pg.rank - 1 + pg.worldSize) % pg.worldSize

	for step := 0; step < pg.worldSize-1; step++ {
		// 发送给下一个rank
		go pg.sendTo(nextRank, result)

		// 接收来自上一个rank的数据
		received := pg.receiveFrom(prevRank)

		// 执行归约操作
		for i := range result {
			switch op {
			case Sum:
				result[i] += received[i]
			case Avg:
				result[i] = (result[i] + received[i]) / 2
			case Max:
				if received[i] > result[i] {
					result[i] = received[i]
				}
			case Min:
				if received[i] < result[i] {
					result[i] = received[i]
				}
			}
		}
	}

	// 广播结果
	if pg.rank == 0 {
		for i := 1; i < pg.worldSize; i++ {
			pg.sendTo(i, result)
		}
	} else {
		result = pg.receiveFrom(0)
	}

	recvBuf.SetRawVector(mat.VecDenseCopyOf(mat.NewVecDense(len(result), result)).RawVector())
	return nil
}

// Broadcast 广播操作
func (pg *ProcessGroup) Broadcast(tensor *mat.VecDense, srcRank int) error {
	if !pg.isInitialized {
		return fmt.Errorf("process group not initialized")
	}

	pg.mu.RLock()
	defer pg.mu.RUnlock()

	data := tensor.RawVector().Data

	if pg.rank == srcRank {
		// 发送给所有其他rank
		for rank := 0; rank < pg.worldSize; rank++ {
			if rank != srcRank {
				pg.sendTo(rank, data)
			}
		}
	} else {
		// 接收数据
		received := pg.receiveFrom(srcRank)
		copy(data, received)
	}

	return nil
}

// Reduce 归约操作
func (pg *ProcessGroup) Reduce(sendBuf, recvBuf *mat.VecDense, dstRank int, op ReduceOp) error {
	if !pg.isInitialized {
		return fmt.Errorf("process group not initialized")
	}

	// 先执行AllReduce，然后只保留dstRank的结果
	if err := pg.AllReduce(sendBuf, recvBuf, op); err != nil {
		return err
	}

	if pg.rank != dstRank {
		// 非目标rank清空结果
		data := recvBuf.RawVector().Data
		for i := range data {
			data[i] = 0
		}
	}

	return nil
}

// Gather 收集操作
func (pg *ProcessGroup) Gather(sendBuf *mat.VecDense, recvBuf *mat.VecDense, dstRank int) error {
	if !pg.isInitialized {
		return fmt.Errorf("process group not initialized")
	}

	pg.mu.RLock()
	defer pg.mu.RUnlock()

	sendData := sendBuf.RawVector().Data

	if pg.rank == dstRank {
		// 接收所有rank的数据
		recvData := recvBuf.RawVector().Data
		copy(recvData, sendData)

		for rank := 0; rank < pg.worldSize; rank++ {
			if rank != dstRank {
				received := pg.receiveFrom(rank)
				offset := len(sendData) * rank
				copy(recvData[offset:], received)
			}
		}
	} else {
		// 发送给dstRank
		pg.sendTo(dstRank, sendData)
	}

	return nil
}

// Scatter 分散操作
func (pg *ProcessGroup) Scatter(sendBuf *mat.VecDense, recvBuf *mat.VecDense, srcRank int) error {
	if !pg.isInitialized {
		return fmt.Errorf("process group not initialized")
	}

	pg.mu.RLock()
	defer pg.mu.RUnlock()

	chunkSize := recvBuf.Len()

	if pg.rank == srcRank {
		sendData := sendBuf.RawVector().Data

		// 发送给所有rank
		for rank := 0; rank < pg.worldSize; rank++ {
			offset := chunkSize * rank
			chunk := sendData[offset : offset+chunkSize]

			if rank == srcRank {
				// 直接复制
				recvBuf.SetRawVector(mat.VecDenseCopyOf(mat.NewVecDense(chunkSize, chunk)).RawVector())
			} else {
				// 发送给对应rank
				pg.sendTo(rank, chunk)
			}
		}
	} else {
		// 接收数据
		received := pg.receiveFrom(srcRank)
		recvBuf.SetRawVector(mat.VecDenseCopyOf(mat.NewVecDense(len(received), received)).RawVector())
	}

	return nil
}

// AllGather 全收集操作
func (pg *ProcessGroup) AllGather(sendBuf *mat.VecDense, recvBuf *mat.VecDense) error {
	if !pg.isInitialized {
		return fmt.Errorf("process group not initialized")
	}

	pg.mu.RLock()
	defer pg.mu.RUnlock()

	sendData := sendBuf.RawVector().Data
	recvData := recvBuf.RawVector().Data

	// 每个rank发送给所有其他rank
	for rank := 0; rank < pg.worldSize; rank++ {
		if rank == pg.rank {
			// 直接复制自己的数据
			offset := len(sendData) * rank
			copy(recvData[offset:], sendData)
		} else {
			go pg.sendTo(rank, sendData)
		}
	}

	// 接收其他rank的数据
	for rank := 0; rank < pg.worldSize; rank++ {
		if rank != pg.rank {
			received := pg.receiveFrom(rank)
			offset := len(sendData) * rank
			copy(recvData[offset:], received)
		}
	}

	return nil
}

// Barrier 屏障同步
func (pg *ProcessGroup) Barrier() error {
	if !pg.isInitialized {
		return fmt.Errorf("process group not initialized")
	}

	// 简化的屏障实现
	dummy := mat.NewVecDense(1, []float64{1.0})
	result := mat.NewVecDense(1, nil)

	return pg.AllReduce(dummy, result, Sum)
}

// ==================== 辅助函数 ====================

// sendTo 发送数据到指定rank
func (pg *ProcessGroup) sendTo(rank int, data []float64) error {
	conn, ok := pg.connections[rank]
	if !ok {
		return fmt.Errorf("no connection to rank %d", rank)
	}

	encoder := gob.NewEncoder(conn)
	return encoder.Encode(data)
}

// receiveFrom 从指定rank接收数据
func (pg *ProcessGroup) receiveFrom(rank int) []float64 {
	conn, ok := pg.connections[rank]
	if !ok {
		return nil
	}

	decoder := gob.NewDecoder(conn)
	var data []float64
	if err := decoder.Decode(&data); err != nil {
		return nil
	}

	return data
}

// ReduceOp 归约操作类型
type ReduceOp int

const (
	Sum ReduceOp = iota
	Avg
	Max
	Min
	Product
)

// ==================== DDP (DistributedDataParallel) ====================

// DDPOption DDP选项
type DDPOption struct {
	BucketCapMB          int
	BucketBytesCap       int64
	GradientAsBucketView bool
	StaticGraph          bool
	FindUnusedParameters bool
}

// DefaultDDPOption 返回默认DDP选项
func DefaultDDPOption() *DDPOption {
	return &DDPOption{
		BucketCapMB:          25,
		BucketBytesCap:       25 * 1024 * 1024,
		GradientAsBucketView: false,
		StaticGraph:          false,
		FindUnusedParameters: false,
	}
}

// DistributedDataParallel 分布式数据并行包装器
type DistributedDataParallel struct {
	module               interface{} // 实际的模型
	processGroup         *ProcessGroup
	deviceID             int
	outputDevice         int
	broadcastBuffers     bool
	findUnusedParameters bool
	gradBucket           map[string]*mat.VecDense
}

// NewDistributedDataParallel 创建DDP包装器
func NewDistributedDataParallel(module interface{}, deviceID int, outputDevice int, broadcastBuffers bool,
	findUnusedParameters bool, processGroup *ProcessGroup) (*DistributedDataParallel, error) {

	if processGroup == nil {
		return nil, fmt.Errorf("process group is required")
	}

	ddp := &DistributedDataParallel{
		module:               module,
		processGroup:         processGroup,
		deviceID:             deviceID,
		outputDevice:         outputDevice,
		broadcastBuffers:     broadcastBuffers,
		findUnusedParameters: findUnusedParameters,
		gradBucket:           make(map[string]*mat.VecDense),
	}

	// 广播初始参数
	if broadcastBuffers {
		if err := ddp.broadcastParameters(); err != nil {
			return nil, err
		}
	}

	return ddp, nil
}

// broadcastParameters 广播参数
func (ddp *DistributedDataParallel) broadcastParameters() error {
	// 从rank 0广播到所有其他rank
	return nil
}

// Forward 前向传播
func (ddp *DistributedDataParallel) Forward(input interface{}) interface{} {
	// 调用实际模型的前向传播
	return nil
}

// Backward 反向传播并同步梯度
func (ddp *DistributedDataParallel) Backward(loss interface{}) error {
	// 1. 执行反向传播
	// 2. 同步梯度
	return ddp.syncGradients()
}

// syncGradients 同步梯度
func (ddp *DistributedDataParallel) syncGradients() error {
	// 对所有梯度执行AllReduce
	for name, grad := range ddp.gradBucket {
		reduced := mat.NewVecDense(grad.Len(), nil)
		if err := ddp.processGroup.AllReduce(grad, reduced, Avg); err != nil {
			return fmt.Errorf("failed to reduce gradient %s: %w", name, err)
		}
		grad.CopyVec(reduced)
	}
	return nil
}

// NoSync 返回无同步上下文
func (ddp *DistributedDataParallel) NoSync() *NoSyncContext {
	return &NoSyncContext{ddp: ddp}
}

// NoSyncContext 无同步上下文
type NoSyncContext struct {
	ddp *DistributedDataParallel
}

// Enter 进入上下文
func (ctx *NoSyncContext) Enter() {
	// 禁用梯度同步
}

// Exit 退出上下文
func (ctx *NoSyncContext) Exit() {
	// 恢复梯度同步
}

// ==================== 辅助函数 ====================

// IsDistributed 检查是否处于分布式模式
func IsDistributed() bool {
	return GetWorldSize() > 1
}

// GetRank 获取当前rank
func GetRank() int {
	// 从环境变量或全局状态获取
	return 0
}

// GetWorldSize 获取世界大小
func GetWorldSize() int {
	// 从环境变量或全局状态获取
	return 1
}

// GetLocalRank 获取本地rank
func GetLocalRank() int {
	return 0
}

// InitProcessGroup 初始化进程组
func InitProcessGroup(backend Backend, timeout time.Duration) error {
	// 全局初始化
	return nil
}

// DestroyProcessGroup 销毁进程组
func DestroyProcessGroup() error {
	// 全局清理
	return nil
}
