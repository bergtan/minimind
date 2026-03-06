package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gominimind/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// setupMockServer creates a mock HTTP server for testing
func setupMockServer(handler http.HandlerFunc) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(handler))
}

// createTestConfig creates a test client configuration
func createTestConfig(baseURL string) *ClientConfig {
	return &ClientConfig{
		BaseURL:    baseURL,
		APIKey:     "test-api-key",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
		RetryDelay: time.Second,
	}
}

// TestNewClient - 娴嬭瘯瀹㈡埛绔垱寤?
func TestNewClient(t *testing.T) {
	t.Run("Valid Config", func(t *testing.T) {
		server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		defer server.Close()

		cfg := createTestConfig(server.URL)
		client := NewMiniMindClient(cfg)

		assert.NotNil(t, client)
		assert.NotNil(t, client.httpClient)
		assert.Equal(t, cfg, client.config)
	})

	t.Run("Empty BaseURL", func(t *testing.T) {
		cfg := &ClientConfig{
			BaseURL: "",
			APIKey:  "test-key",
		}

		client := NewMiniMindClient(cfg)
		// NewMiniMindClient涓嶈繑鍥瀍rror锛屼絾绌築aseURL浼氬鑷村悗缁姹傚け锟?
		assert.NotNil(t, client)
	})

	t.Run("Default Timeout", func(t *testing.T) {
		server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		defer server.Close()

		cfg := &ClientConfig{
			BaseURL: server.URL,
			APIKey:  "test-key",
			// Timeout not set, should use default
		}

		client := NewMiniMindClient(cfg)
		assert.NotNil(t, client)
		// Verify default timeout is set
		assert.Equal(t, 30*time.Second, client.config.Timeout)
	})

	t.Run("Default MaxRetries", func(t *testing.T) {
		server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		defer server.Close()

		cfg := &ClientConfig{
			BaseURL: server.URL,
			APIKey:  "test-key",
			// MaxRetries not set, should use default
		}

		client := NewMiniMindClient(cfg)
		assert.NotNil(t, client)
		assert.Equal(t, 3, client.config.MaxRetries)
	})
}

// TestChatCompletion - 娴嬭瘯鑱婂ぉ琛ュ叏
func TestChatCompletion(t *testing.T) {
	response := &types.ChatCompletionResponse{
		ID:      "chat-test-id",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "minimind",
		Choices: []types.ChatCompletionChoice{
			{
				Index: 0,
				Message: types.Message{
					Role:    "assistant",
					Content: "Hello! How can I help you today?",
				},
				FinishReason: "stop",
			},
		},
		Usage: types.ChatCompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 8,
			TotalTokens:      18,
		},
	}

	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req types.ChatCompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		assert.NoError(t, err)
		assert.Greater(t, len(req.Messages), 0)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	t.Run("Successful Chat Completion", func(t *testing.T) {
		req := &types.ChatCompletionRequest{
			Model: "minimind",
			Messages: []types.Message{
				{
					Role:    "user",
					Content: "Hello",
				},
			},
			MaxTokens:   100,
			Temperature: 0.7,
		}

		resp, err := client.ChatCompletion(req)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, response.ID, resp.ID)
		assert.Equal(t, response.Model, resp.Model)
		assert.Equal(t, len(response.Choices), len(resp.Choices))
		assert.Equal(t, response.Choices[0].Message.Content, resp.Choices[0].Message.Content)
	})

	t.Run("Chat Completion with Context", func(t *testing.T) {
		req := &types.ChatCompletionRequest{
			Model: "minimind",
			Messages: []types.Message{
				{
					Role:    "user",
					Content: "Hello",
				},
			},
		}

		ctx := context.Background()
		resp, err := client.ChatCompletionContext(ctx, req)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
	})

	t.Run("Network Error with Retry", func(t *testing.T) {
		failCount := 0
		unstableServer := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			failCount++
			if failCount < 2 {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		})
		defer unstableServer.Close()

		cfg := createTestConfig(unstableServer.URL)
		cfg.MaxRetries = 3
		cfg.RetryDelay = 100 * time.Millisecond

		unstableClient, _ := NewClient(cfg)

		req := &types.ChatCompletionRequest{
			Model: "minimind",
			Messages: []types.Message{
				{Role: "user", Content: "Hello"},
			},
		}

		resp, err := unstableClient.ChatCompletion(req)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.GreaterOrEqual(t, failCount, 1)
	})
}

// TestStreamChatCompletion - 娴嬭瘯娴佸紡鑱婂ぉ琛ュ叏
func TestStreamChatCompletion(t *testing.T) {
	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)

		var req types.ChatCompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		assert.NoError(t, err)
		assert.True(t, req.Stream)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		// Send streaming responses
		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}

		chunks := []string{
			"Hello",
			"! ",
			"How ",
			"can ",
			"I ",
			"help ",
			"you",
			"?",
		}

		for i, chunk := range chunks {
			resp := &types.ChatCompletionStreamResponse{
				ID:      fmt.Sprintf("chatcmpl-stream-%d", i),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   "minimind",
				Choices: []types.ChatCompletionStreamChoice{
					{
						Index: 0,
						Delta: types.ChatMessageDelta{
							Content: chunk,
						},
						FinishReason: nil,
					},
				},
			}

			data, _ := json.Marshal(resp)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}

		// Send done marker
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	t.Run("Streaming Chat Completion", func(t *testing.T) {
		req := &types.ChatCompletionRequest{
			Model: "minimind",
			Messages: []types.Message{
				{Role: "user", Content: "Hello"},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(req)
		assert.NoError(t, err)
		assert.NotNil(t, stream)

		chunks := make([]string, 0)
		for resp := range stream {
			if resp.Error != nil {
				t.Logf("Stream error: %v", resp.Error)
				break
			}

			for _, choice := range resp.Choices {
				chunks = append(chunks, choice.Delta.Content)
			}
		}

		// Verify we received chunks
		assert.Greater(t, len(chunks), 0)
		finalContent := strings.Join(chunks, "")
		assert.NotEmpty(t, finalContent)
	})
}

// TestTextCompletion - 娴嬭瘯鏂囨湰琛ュ叏
func TestTextCompletion(t *testing.T) {
	response := &types.CompletionResponse{
		ID:      "comp-test-id",
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   "minimind",
		Choices: []types.CompletionChoice{
			{
				Text:         " is a question. The answer is...",
				Index:        0,
				LogProbs:     nil,
				FinishReason: "length",
			},
		},
		Usage: types.CompletionUsage{
			PromptTokens:     5,
			CompletionTokens: 10,
			TotalTokens:      15,
		},
	}

	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/v1/completions", r.URL.Path)

		var req types.CompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		assert.NoError(t, err)
		assert.NotEmpty(t, req.Prompt)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	t.Run("Successful Text Completion", func(t *testing.T) {
		req := &types.CompletionRequest{
			Model:       "minimind",
			Prompt:      "What is AI?",
			MaxTokens:   50,
			Temperature: 0.8,
		}

		resp, err := client.TextCompletion(req)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, response.ID, resp.ID)
		assert.Greater(t, len(resp.Choices), 0)
	})
}

// TestEmbedding - 娴嬭瘯宓屽叆鐢熸垚
func TestEmbedding(t *testing.T) {
	response := &types.EmbeddingResponse{
		Object: "list",
		Data: []types.EmbeddingData{
			{
				Object:    "embedding",
				Embedding: make([]float64, 768),
				Index:     0,
			},
		},
		Model: "minimind",
		Usage: types.EmbeddingUsage{
			PromptTokens: 10,
			TotalTokens:  10,
		},
	}

	// Initialize embedding with some values
	for i := range response.Data[0].Embedding {
		response.Data[0].Embedding[i] = float64(i) * 0.001
	}

	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/v1/embeddings", r.URL.Path)

		var req types.EmbeddingRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		assert.NoError(t, err)
		assert.NotEmpty(t, req.Input)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	t.Run("Successful Embedding", func(t *testing.T) {
		req := &types.EmbeddingRequest{
			Model: "minimind",
			Input: []string{"This is a test sentence for embedding."},
		}

		resp, err := client.Embedding(req)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Greater(t, len(resp.Data), 0)
		assert.Greater(t, len(resp.Data[0].Embedding), 0)
	})

	t.Run("Embedding with Multiple Inputs", func(t *testing.T) {
		responseMulti := &types.EmbeddingResponse{
			Object: "list",
			Data: []types.EmbeddingData{
				{Object: "embedding", Embedding: make([]float64, 768), Index: 0},
				{Object: "embedding", Embedding: make([]float64, 768), Index: 1},
			},
			Model: "minimind",
			Usage: types.EmbeddingUsage{
				PromptTokens: 20,
				TotalTokens:  20,
			},
		}

		multiServer := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(responseMulti)
		})
		defer multiServer.Close()

		multiCfg := createTestConfig(multiServer.URL)
		multiClient, _ := NewClient(multiCfg)

		req := &types.EmbeddingRequest{
			Model: "minimind",
			Input: []string{
				"First sentence.",
				"Second sentence.",
			},
		}

		resp, err := multiClient.Embedding(req)
		assert.NoError(t, err)
		assert.Equal(t, 2, len(resp.Data))
	})
}

// TestBatchEmbedding - 娴嬭瘯鎵归噺宓屽叆鐢熸垚
func TestBatchEmbedding(t *testing.T) {
	response := &types.EmbeddingResponse{
		Object: "list",
		Data: []types.EmbeddingData{
			{Object: "embedding", Embedding: make([]float64, 768), Index: 0},
			{Object: "embedding", Embedding: make([]float64, 768), Index: 1},
		},
		Model: "minimind",
		Usage: types.EmbeddingUsage{
			PromptTokens: 20,
			TotalTokens:  20,
		},
	}

	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	t.Run("Batch Embedding Processing", func(t *testing.T) {
		inputs := []string{
			"First document for embedding.",
			"Second document for embedding.",
		}

		results, err := client.BatchEmbedding(inputs, "minimind")
		assert.NoError(t, err)
		assert.NotNil(t, results)
		assert.Greater(t, len(results.Data), 0)
	})
}

// TestCreateChatCompletion - 娴嬭瘯绠€鍖栫増鑱婂ぉ琛ュ叏
func TestCreateChatCompletion(t *testing.T) {
	response := &types.ChatCompletionResponse{
		ID:      "chat-simple-id",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "minimind",
		Choices: []types.ChatCompletionChoice{
			{
				Index: 0,
				Message: types.Message{
					Role:    "assistant",
					Content: "This is a response.",
				},
				FinishReason: "stop",
			},
		},
	}

	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	t.Run("Simplified Chat Completion", func(t *testing.T) {
		content, err := client.CreateChatCompletion("minimind", "Hello, how are you?")
		assert.NoError(t, err)
		assert.Equal(t, "This is a response.", content)
	})
}

// TestErrorHandling - 娴嬭瘯閿欒澶勭悊
func TestErrorHandling(t *testing.T) {
	t.Run("Server Error", func(t *testing.T) {
		server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			apiErr := &types.OpenAIError{
				Message: "Internal server error",
				Type:    "server_error",
				Code:    500,
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(gin.H{
				"error": apiErr,
			})
		})
		defer server.Close()

		cfg := createTestConfig(server.URL)
		client := NewMiniMindClient(cfg)

		req := &types.ChatCompletionRequest{
			Model:    "minimind",
			Messages: []types.Message{{Role: "user", Content: "Hello"}},
		}

		_, err := client.ChatCompletion(req)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "Internal server error")
	})

	t.Run("Rate Limit Error", func(t *testing.T) {
		server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Retry-After", "60")
			w.WriteHeader(http.StatusTooManyRequests)
			json.NewEncoder(w).Encode(gin.H{
				"error": types.OpenAIError{
					Message: "Rate limit exceeded",
					Type:    "rate_limit_error",
					Code:    429,
				},
			})
		})
		defer server.Close()

		cfg := createTestConfig(server.URL)
		client := NewMiniMindClient(cfg)

		req := &types.ChatCompletionRequest{
			Model:    "minimind",
			Messages: []types.Message{{Role: "user", Content: "Hello"}},
		}

		_, err := client.ChatCompletion(req)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "Rate limit")
	})

	t.Run("Authentication Error", func(t *testing.T) {
		server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(gin.H{
				"error": types.OpenAIError{
					Message: "Invalid API key",
					Type:    "authentication_error",
					Code:    401,
				},
			})
		})
		defer server.Close()

		cfg := createTestConfig(server.URL)
		cfg.APIKey = "wrong-key"
		client := NewMiniMindClient(cfg)

		req := &types.ChatCompletionRequest{
			Model:    "minimind",
			Messages: []types.Message{{Role: "user", Content: "Hello"}},
		}

		_, err := client.ChatCompletion(req)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "401")
	})
}

// TestClientClose - 娴嬭瘯瀹㈡埛绔叧锟?
func TestClientClose(t *testing.T) {
	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	err := client.Close()
	assert.NoError(t, err)
}

// BenchmarkChatCompletion - 鑱婂ぉ琛ュ叏鎬ц兘娴嬭瘯
func BenchmarkChatCompletion(b *testing.B) {
	response := &types.ChatCompletionResponse{
		ID:      "benchmark-id",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "minimind",
		Choices: []types.ChatCompletionChoice{
			{
				Index: 0,
				Message: types.Message{
					Role:    "assistant",
					Content: "Benchmark response",
				},
				FinishReason: "stop",
			},
		},
	}

	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	req := &types.ChatCompletionRequest{
		Model:    "minimind",
		Messages: []types.Message{{Role: "user", Content: "Hello"}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.ChatCompletion(req)
	}
}

// BenchmarkEmbedding - 宓屽叆鐢熸垚鎬ц兘娴嬭瘯
func BenchmarkEmbedding(b *testing.B) {
	response := &types.EmbeddingResponse{
		Object: "list",
		Data: []types.EmbeddingData{
			{
				Object:    "embedding",
				Embedding: make([]float64, 768),
				Index:     0,
			},
		},
		Model: "minimind",
	}

	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	req := &types.EmbeddingRequest{
		Model: "minimind",
		Input: []string{"Test input for embedding"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.Embedding(req)
	}
}

// TestClientConcurrency - 娴嬭瘯瀹㈡埛绔苟鍙戝畨锟?
func TestClientConcurrency(t *testing.T) {
	server := setupMockServer(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(&types.ChatCompletionResponse{
			ID:     "concurrent-id",
			Object: "chat.completion",
			Choices: []types.ChatCompletionChoice{
				{Index: 0, Message: types.Message{Role: "assistant", Content: "Response"}},
			},
		})
	})
	defer server.Close()

	cfg := createTestConfig(server.URL)
	client := NewMiniMindClient(cfg)

	t.Run("Concurrent Requests", func(t *testing.T) {
		done := make(chan bool, 100)

		for i := 0; i < 100; i++ {
			go func() {
				req := &types.ChatCompletionRequest{
					Model:    "minimind",
					Messages: []types.Message{{Role: "user", Content: "Hello"}},
				}
				_, err := client.ChatCompletion(req)
				if err != nil {
					t.Errorf("Request failed: %v", err)
				}
				done <- true
			}()
		}

		for i := 0; i < 100; i++ {
			<-done
		}
	})
}
