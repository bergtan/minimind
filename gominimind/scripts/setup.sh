#!/bin/bash

# GoMiniMind 安装脚本
# 用于自动安装和配置项目

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印banner
print_banner() {
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║         GoMiniMind Installer           ║"
    echo "║         版本 1.0.0                     ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查Go版本
    if ! command -v go &> /dev/null; then
        log_error "Go 未安装，请先安装Go 1.20或更高版本"
        exit 1
    fi
    
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    log_info "检测到Go版本: $GO_VERSION"
    
    # 检查Docker (可选)
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        log_info "检测到Docker版本: $DOCKER_VERSION"
    else
        log_warning "Docker 未安装，将跳过Docker相关配置"
    fi
    
    # 检查Docker Compose (可选)
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        log_info "检测到Docker Compose"
    else
        log_warning "Docker Compose 未安装"
    fi
    
    log_success "系统要求检查完成"
}

# 安装Go依赖
install_go_deps() {
    log_info "安装Go依赖..."
    
    go mod download
    go mod verify
    
    log_success "Go依赖安装完成"
}

# 安装开发工具
install_dev_tools() {
    log_info "安装开发工具..."
    
    # golangci-lint
    if ! command -v golangci-lint &> /dev/null; then
        log_info "安装 golangci-lint..."
        curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.55.2
    fi
    
    # mockery (用于生成mock)
    if ! command -v mockery &> /dev/null; then
        log_info "安装 mockery..."
        go install github.com/vektra/mockery/v2@latest
    fi
    
    # air (热重载)
    if ! command -v air &> /dev/null; then
        log_info "安装 air (热重载工具)..."
        go install github.com/cosmtrek/air@latest
    fi
    
    log_success "开发工具安装完成"
}

# 创建目录结构
create_directories() {
    log_info "创建目录结构..."
    
    mkdir -p data
    mkdir -p logs
    mkdir -p models
    mkdir -p configs
    mkdir -p plugins
    mkdir -p middleware
    
    log_success "目录结构创建完成"
}

# 复制配置文件
copy_configs() {
    log_info "复制配置文件..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_warning "请编辑 .env 文件配置环境变量"
        fi
    fi
    
    log_success "配置文件准备完成"
}

# 构建项目
build_project() {
    log_info "构建项目..."
    
    # 清理旧构建
    rm -rf dist/
    
    # 构建API服务器
    log_info "构建API服务器..."
    go build -o dist/gominimind ./cmd/serve_openai_api
    
    # 构建训练工具
    log_info "构建训练工具..."
    go build -o dist/gominimind-train ./cmd/train
    
    # 构建模型转换工具
    log_info "构建模型转换工具..."
    go build -o dist/gominimind-convert ./cmd/convert_model
    
    # 构建评估工具
    log_info "构建评估工具..."
    go build -o dist/gominimind-eval ./cmd/evaluate
    
    log_success "项目构建完成，二进制文件在 dist/ 目录"
}

# 运行测试
run_tests() {
    log_info "运行测试..."
    
    # 运行单元测试
    go test -v -cover ./... -count=1
    
    log_success "测试完成"
}

# 配置systemd服务
setup_systemd() {
    if [ "$EUID" -eq 0 ]; then
        log_info "配置systemd服务..."
        
        cat > /etc/systemd/system/gominimind.service << EOF
[Unit]
Description=GoMiniMind API Server
After=network.target redis.service postgresql.service
Requires=network.target

[Service]
Type=simple
User=gominimind
Group=gominimind
WorkingDirectory=/opt/gominimind
ExecStart=/opt/gominimind/dist/gominimind --config /opt/gominimind/configs/config.yaml
Restart=always
RestartSec=5
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/opt/gominimind/.env

[Install]
WantedBy=multi-user.target
EOF
        
        # 创建用户
        if ! id "gominimind" &>/dev/null; then
            useradd -r -s /bin/false gominimind
        fi
        
        systemctl daemon-reload
        log_success "systemd服务配置完成"
        log_info "使用以下命令启动服务:"
        log_info "  sudo systemctl start gominimind"
        log_info "  sudo systemctl enable gominimind"
    fi
}

# 显示帮助信息
show_help() {
    echo "GoMiniMind 安装脚本"
    echo ""
    echo "用法: ./setup.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h       显示帮助信息"
    echo "  --dev            安装开发工具"
    echo "  --test           运行测试"
    echo "  --build          构建项目"
    echo "  --systemd        配置systemd服务 (需要root权限)"
    echo "  --all            执行完整安装"
    echo ""
}

# 主函数
main() {
    print_banner
    
    # 如果没有参数，显示帮助
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local DEV_MODE=false
    local RUN_TESTS=false
    local BUILD_PROJECT=false
    local SETUP_SYSTEMD=false
    local FULL_INSTALL=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --test)
                RUN_TESTS=true
                shift
                ;;
            --build)
                BUILD_PROJECT=true
                shift
                ;;
            --systemd)
                SETUP_SYSTEMD=true
                shift
                ;;
            --all)
                FULL_INSTALL=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 执行安装步骤
    if [ "$FULL_INSTALL" = true ]; then
        check_requirements
        install_go_deps
        install_dev_tools
        create_directories
        copy_configs
        build_project
        run_tests
        setup_systemd
    else
        check_requirements
        
        if [ "$DEV_MODE" = true ]; then
            install_dev_tools
        fi
        
        install_go_deps
        create_directories
        copy_configs
        
        if [ "$BUILD_PROJECT" = true ]; then
            build_project
        fi
        
        if [ "$RUN_TESTS" = true ]; then
            if [ ! "$BUILD_PROJECT" = true ]; then
                go build ./...
            fi
            run_tests
        fi
        
        if [ "$SETUP_SYSTEMD" = true ]; then
            setup_systemd
        fi
    fi
    
    echo ""
    log_success "安装完成！"
    echo ""
    echo "下一步:"
    echo "  1. 编辑 .env 文件配置环境变量"
    echo "  2. 编辑 configs/config.yaml 配置文件"
    echo "  3. 运行 'make serve' 启动服务器"
    echo ""
    echo "更多信息请参考 README.md"
    echo ""
}

# 运行主函数
main "$@"