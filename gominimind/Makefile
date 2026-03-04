# MiniMind Go API Server Makefile
# Build, test, and deployment automation

# =============================================================================
# Configuration
# =============================================================================

# Project information
PROJECT_NAME = minimind-api
VERSION = 1.0.0
BUILD_DATE = $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT = $(shell git rev-parse --short HEAD)
GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD)

# Go configuration
GO = go
GO_VERSION = $(shell go version | cut -d' ' -f3)
GO_PACKAGES = $(shell go list ./...)
GO_FILES = $(shell find . -name '*.go' -type f)

# Docker configuration
DOCKER = docker
DOCKER_COMPOSE = docker-compose
DOCKER_REGISTRY = registry.example.com
DOCKER_NAMESPACE = minimind
DOCKER_TAG = latest
DOCKER_IMAGE = $(DOCKER_REGISTRY)/$(DOCKER_NAMESPACE)/$(PROJECT_NAME):$(DOCKER_TAG)

# Build configuration
BUILD_DIR = ./build
BINARY_NAME = minimind-api
BINARY_PATH = $(BUILD_DIR)/$(BINARY_NAME)

# Test configuration
TEST_COVERAGE = 80
TEST_TIMEOUT = 30s

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help: ## Display this help message
	@echo "MiniMind Go API Server - Build and Deployment Automation"
	@echo ""
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Development
# =============================================================================

.PHONY: dev
dev: ## Start development environment
	@echo "Starting development environment..."
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml up --build

.PHONY: dev-down
dev-down: ## Stop development environment
	@echo "Stopping development environment..."
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml down

.PHONY: dev-logs
dev-logs: ## Show development environment logs
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml logs -f

.PHONY: run
run: ## Run the application locally
	@echo "Running application locally..."
	$(GO) run ./cmd/serve_openai_api

.PHONY: watch
watch: ## Run with file watching for development
	@echo "Starting file watcher..."
	air -c .air.toml

# =============================================================================
# Build
# =============================================================================

.PHONY: build
build: clean ## Build the application
	@echo "Building $(PROJECT_NAME)..."
	mkdir -p $(BUILD_DIR)
	$(GO) build -v -ldflags="\
		-X 'main.version=$(VERSION)' \
		-X 'main.buildDate=$(BUILD_DATE)' \
		-X 'main.gitCommit=$(GIT_COMMIT)' \
		-X 'main.gitBranch=$(GIT_BRANCH)'" \
		-o $(BINARY_PATH) ./cmd/serve_openai_api
	@echo "Build complete: $(BINARY_PATH)"

.PHONY: build-linux
build-linux: clean ## Build for Linux
	@echo "Building for Linux..."
	mkdir -p $(BUILD_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -v -ldflags="\
		-X 'main.version=$(VERSION)' \
		-X 'main.buildDate=$(BUILD_DATE)' \
		-X 'main.gitCommit=$(GIT_COMMIT)' \
		-X 'main.gitBranch=$(GIT_BRANCH)'" \
		-o $(BINARY_PATH)-linux-amd64 ./cmd/serve_openai_api
	@echo "Linux build complete: $(BINARY_PATH)-linux-amd64"

.PHONY: build-darwin
build-darwin: clean ## Build for macOS
	@echo "Building for macOS..."
	mkdir -p $(BUILD_DIR)
	GOOS=darwin GOARCH=amd64 $(GO) build -v -ldflags="\
		-X 'main.version=$(VERSION)' \
		-X 'main.buildDate=$(BUILD_DATE)' \
		-X 'main.gitCommit=$(GIT_COMMIT)' \
		-X 'main.gitBranch=$(GIT_BRANCH)'" \
		-o $(BINARY_PATH)-darwin-amd64 ./cmd/serve_openai_api
	@echo "macOS build complete: $(BINARY_PATH)-darwin-amd64"

.PHONY: build-windows
build-windows: clean ## Build for Windows
	@echo "Building for Windows..."
	mkdir -p $(BUILD_DIR)
	GOOS=windows GOARCH=amd64 $(GO) build -v -ldflags="\
		-X 'main.version=$(VERSION)' \
		-X 'main.buildDate=$(BUILD_DATE)' \
		-X 'main.gitCommit=$(GIT_COMMIT)' \
		-X 'main.gitBranch=$(GIT_BRANCH)'" \
		-o $(BINARY_PATH)-windows-amd64.exe ./cmd/serve_openai_api
	@echo "Windows build complete: $(BINARY_PATH)-windows-amd64.exe"

.PHONY: build-all
build-all: build-linux build-darwin build-windows ## Build for all platforms
	@echo "All platform builds complete"

# =============================================================================
# Docker
# =============================================================================

.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	$(DOCKER) build \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		-t $(DOCKER_IMAGE) .
	@echo "Docker image built: $(DOCKER_IMAGE)"

.PHONY: docker-run
docker-run: docker-build ## Run Docker container
	@echo "Running Docker container..."
	$(DOCKER) run -p 8000:8000 $(DOCKER_IMAGE)

.PHONY: docker-push
docker-push: docker-build ## Push Docker image to registry
	@echo "Pushing Docker image to registry..."
	$(DOCKER) push $(DOCKER_IMAGE)

.PHONY: docker-compose-up
docker-compose-up: ## Start services with Docker Compose
	@echo "Starting services with Docker Compose..."
	$(DOCKER_COMPOSE) up -d

.PHONY: docker-compose-down
docker-compose-down: ## Stop services with Docker Compose
	@echo "Stopping services with Docker Compose..."
	$(DOCKER_COMPOSE) down

.PHONY: docker-compose-logs
docker-compose-logs: ## Show Docker Compose logs
	$(DOCKER_COMPOSE) logs -f

# =============================================================================
# Testing
# =============================================================================

.PHONY: test
test: ## Run all tests
	@echo "Running tests..."
	$(GO) test -v -timeout=$(TEST_TIMEOUT) ./...

.PHONY: test-coverage
test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	$(GO) test -v -timeout=$(TEST_TIMEOUT) -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

.PHONY: test-benchmark
test-benchmark: ## Run benchmark tests
	@echo "Running benchmark tests..."
	$(GO) test -bench=. -benchmem ./...

.PHONY: test-race
test-race: ## Run tests with race detector
	@echo "Running tests with race detector..."
	$(GO) test -race -timeout=$(TEST_TIMEOUT) ./...

.PHONY: lint
lint: ## Run linters
	@echo "Running linters..."
	golangci-lint run

.PHONY: vet
vet: ## Run go vet
	@echo "Running go vet..."
	$(GO) vet ./...

.PHONY: fmt
fmt: ## Format code
	@echo "Formatting code..."
	$(GO) fmt ./...

.PHONY: tidy
tidy: ## Tidy go.mod
	@echo "Tidying go.mod..."
	$(GO) mod tidy

# =============================================================================
# Code Quality
# =============================================================================

.PHONY: quality
quality: lint vet test-coverage ## Run all quality checks
	@echo "All quality checks passed"

.PHONY: security
security: ## Run security checks
	@echo "Running security checks..."
	gosec ./...

.PHONY: audit
audit: ## Audit dependencies
	@echo "Auditing dependencies..."
	$(GO) list -json -m all | nancy sleuth

# =============================================================================
# Documentation
# =============================================================================

.PHONY: docs
docs: ## Generate documentation
	@echo "Generating documentation..."
	swag init -g cmd/serve_openai_api/main.go -o docs/api
	$(GO) doc ./... > docs/godoc.txt

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	cd docs && python -m http.server 8001

# =============================================================================
# Deployment
# =============================================================================

.PHONY: deploy-staging
deploy-staging: docker-build docker-push ## Deploy to staging
	@echo "Deploying to staging..."
	kubectl config use-context staging
	kubectl apply -f k8s/staging/

.PHONY: deploy-production
deploy-production: docker-build docker-push ## Deploy to production
	@echo "Deploying to production..."
	kubectl config use-context production
	kubectl apply -f k8s/production/

.PHONY: deploy-rollback
deploy-rollback: ## Rollback deployment
	@echo "Rolling back deployment..."
	kubectl rollout undo deployment/$(PROJECT_NAME)

# =============================================================================
# Monitoring
# =============================================================================

.PHONY: monitor
monitor: ## Start monitoring stack
	@echo "Starting monitoring stack..."
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml up -d

.PHONY: monitor-down
monitor-down: ## Stop monitoring stack
	@echo "Stopping monitoring stack..."
	$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml down

.PHONY: logs
logs: ## Show application logs
	$(DOCKER_COMPOSE) logs -f $(PROJECT_NAME)

.PHONY: status
status: ## Show service status
	$(DOCKER_COMPOSE) ps

# =============================================================================
# Database
# =============================================================================

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	$(GO) run ./cmd/migrate

.PHONY: db-seed
db-seed: ## Seed database with sample data
	@echo "Seeding database..."
	$(GO) run ./cmd/seed

.PHONY: db-backup
db-backup: ## Backup database
	@echo "Backing up database..."
	$(DOCKER_COMPOSE) exec postgres pg_dump -U minimind minimind > backup.sql

.PHONY: db-restore
db-restore: ## Restore database from backup
	@echo "Restoring database..."
	$(DOCKER_COMPOSE) exec -T postgres psql -U minimind minimind < backup.sql

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f coverage.out coverage.html
	rm -f $(BINARY_NAME)

.PHONY: clean-docker
clean-docker: ## Clean Docker artifacts
	@echo "Cleaning Docker artifacts..."
	$(DOCKER_COMPOSE) down -v --rmi all
	$(DOCKER) system prune -f

.PHONY: clean-all
clean-all: clean clean-docker ## Clean all artifacts
	@echo "All artifacts cleaned"

# =============================================================================
# Release
# =============================================================================

.PHONY: release
release: quality build-all docker-build ## Create a release
	@echo "Creating release $(VERSION)..."
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)
	@echo "Release v$(VERSION) created"

.PHONY: release-patch
release-patch: ## Create patch release
	@echo "Creating patch release..."
	$(eval NEW_VERSION = $(shell echo $(VERSION) | awk -F. '{print $$1"."$$2"."$$3+1}'))
	@sed -i.bak 's/VERSION = $(VERSION)/VERSION = $(NEW_VERSION)/' Makefile
	@rm Makefile.bak
	@make release

.PHONY: release-minor
release-minor: ## Create minor release
	@echo "Creating minor release..."
	$(eval NEW_VERSION = $(shell echo $(VERSION) | awk -F. '{print $$1"."$$2+1".0"}'))
	@sed -i.bak 's/VERSION = $(VERSION)/VERSION = $(NEW_VERSION)/' Makefile
	@rm Makefile.bak
	@make release

.PHONY: release-major
release-major: ## Create major release
	@echo "Creating major release..."
	$(eval NEW_VERSION = $(shell echo $(VERSION) | awk -F. '{print $$1+1".0.0"}'))
	@sed -i.bak 's/VERSION = $(VERSION)/VERSION = $(NEW_VERSION)/' Makefile
	@rm Makefile.bak
	@make release

# =============================================================================
# Utility
# =============================================================================

.PHONY: version
version: ## Show version information
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Build Date: $(BUILD_DATE)"
	@echo "Git Commit: $(GIT_COMMIT)"
	@echo "Git Branch: $(GIT_BRANCH)"
	@echo "Go Version: $(GO_VERSION)"

.PHONY: env
env: ## Show environment information
	@echo "Environment Variables:"
	@env | sort

.PHONY: deps
deps: ## Show dependencies
	@echo "Dependencies:"
	$(GO) list -m all

.PHONY: graph
graph: ## Generate dependency graph
	@echo "Generating dependency graph..."
	$(GO) mod graph | dot -Tpng -o deps.png

# =============================================================================
# CI/CD
# =============================================================================

.PHONY: ci-test
ci-test: lint vet test-coverage security audit ## Run CI test suite
	@echo "CI test suite completed successfully"

.PHONY: ci-build
ci-build: ci-test build docker-build ## Run CI build pipeline
	@echo "CI build pipeline completed successfully"

.PHONY: ci-deploy
ci-deploy: ci-build deploy-staging ## Run CI deployment pipeline
	@echo "CI deployment pipeline completed successfully"

# =============================================================================
# Default Target
# =============================================================================

.DEFAULT_GOAL := help