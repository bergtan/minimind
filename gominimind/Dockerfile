# Multi-stage Dockerfile for MiniMind Go API Server

# Stage 1: Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata \
    && update-ca-certificates

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download \
    && go mod verify

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags='-w -s -extldflags "-static"' -o /app/minimind-api ./cmd/serve_openai_api

# Stage 2: Runtime stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata \
    && update-ca-certificates \
    && addgroup -S minimind \
    && adduser -S minimind -G minimind

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data \
    && chown -R minimind:minimind /app

# Switch to non-root user
USER minimind

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder --chown=minimind:minimind /app/minimind-api .

# Copy configuration files
COPY --chown=minimind:minimind configs/ configs/

# Copy model files (if any)
COPY --chown=minimind:minimind models/ models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Set environment variables
ENV GIN_MODE=release
ENV PORT=8000
ENV HOST=0.0.0.0
ENV LOG_LEVEL=info
ENV CONFIG_PATH=/app/configs/default.yaml

# Start the application
CMD ["./minimind-api"]

# Labels
LABEL org.opencontainers.image.title="MiniMind API Server"
LABEL org.opencontainers.image.description="Go implementation of MiniMind API Server"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="MiniMind Team"
LABEL org.opencontainers.image.url="https://github.com/jingyaogong/minimind"
LABEL org.opencontainers.image.source="https://github.com/jingyaogong/minimind"
LABEL org.opencontainers.image.licenses="MIT"