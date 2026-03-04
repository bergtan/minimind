/**
 * MiniMind JavaScript客户端SDK
 * 提供与MiniMind API的完整交互功能
 */

class MiniMindClient {
    /**
     * 创建MiniMind客户端
     * @param {Object} config - 客户端配置
     * @param {string} config.baseUrl - API基础URL
     * @param {string} config.apiKey - API密钥
     * @param {number} config.timeout - 请求超时时间（毫秒）
     * @param {number} config.maxRetries - 最大重试次数
     * @param {number} config.retryDelay - 重试延迟时间（毫秒）
     */
    constructor(config = {}) {
        this.config = {
            baseUrl: config.baseUrl || 'http://localhost:8000',
            apiKey: config.apiKey || '',
            timeout: config.timeout || 30000,
            maxRetries: config.maxRetries || 3,
            retryDelay: config.retryDelay || 1000,
            ...config
        };

        // 初始化服务客户端
        this.chat = new ChatClient(this);
        this.completions = new CompletionsClient(this);
        this.embeddings = new EmbeddingsClient(this);
        this.models = new ModelsClient(this);
        this.files = new FilesClient(this);
    }

    /**
     * 执行HTTP请求
     * @param {string} method - HTTP方法
     * @param {string} path - API路径
     * @param {Object} data - 请求数据
     * @param {Object} options - 请求选项
     * @returns {Promise<Response>}
     */
    async request(method, path, data = null, options = {}) {
        const url = new URL(path, this.config.baseUrl);
        
        // 添加查询参数
        if (options.query) {
            Object.entries(options.query).forEach(([key, value]) => {
                url.searchParams.append(key, value);
            });
        }

        const headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'minimind-js-client/1.0.0',
            ...options.headers
        };

        // 添加认证头
        if (this.config.apiKey) {
            headers['Authorization'] = `Bearer ${this.config.apiKey}`;
        }

        const requestOptions = {
            method: method.toUpperCase(),
            headers: headers,
            timeout: options.timeout || this.config.timeout,
            ...options
        };

        // 添加请求体
        if (data && ['POST', 'PUT', 'PATCH'].includes(method.toUpperCase())) {
            requestOptions.body = JSON.stringify(data);
        }

        let lastError;
        
        for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
            try {
                const response = await this._fetchWithTimeout(url.toString(), requestOptions);
                
                if (response.ok) {
                    return response;
                }

                // 检查是否应该重试
                if (this._shouldRetry(response.status, attempt)) {
                    lastError = new Error(`HTTP ${response.status}: ${response.statusText}`);
                    
                    if (attempt < this.config.maxRetries) {
                        await this._sleep(this.config.retryDelay * Math.pow(2, attempt));
                        continue;
                    }
                }

                // 处理错误响应
                throw await this._handleErrorResponse(response);
            } catch (error) {
                lastError = error;
                
                if (attempt < this.config.maxRetries && this._shouldRetry(null, attempt)) {
                    await this._sleep(this.config.retryDelay * Math.pow(2, attempt));
                    continue;
                }
                
                throw lastError;
            }
        }
    }

    /**
     * 带超时的fetch请求
     */
    async _fetchWithTimeout(url, options) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), options.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    /**
     * 判断是否应该重试
     */
    _shouldRetry(statusCode, attempt) {
        // 网络错误或5xx状态码可以重试
        if (!statusCode) return true;
        if (statusCode >= 500 && statusCode < 600) return true;
        if (statusCode === 429) return true; // 限流
        return false;
    }

    /**
     * 睡眠函数
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * 处理错误响应
     */
    async _handleErrorResponse(response) {
        let errorData;
        
        try {
            errorData = await response.json();
        } catch (e) {
            errorData = { error: { message: response.statusText } };
        }

        const error = new Error(errorData.error?.message || 'API request failed');
        error.status = response.status;
        error.code = errorData.error?.code;
        error.type = errorData.error?.type;
        error.param = errorData.error?.param;
        
        return error;
    }

    /**
     * 健康检查
     */
    async health() {
        const response = await this.request('GET', '/health');
        return await response.json();
    }
}

/**
 * 聊天客户端
 */
class ChatClient {
    constructor(client) {
        this.client = client;
    }

    /**
     * 创建聊天补全
     */
    async createCompletion(request) {
        const response = await this.client.request('POST', '/v1/chat/completions', request);
        return await response.json();
    }

    /**
     * 创建流式聊天补全
     */
    async createCompletionStream(request, onChunk) {
        const streamRequest = { ...request, stream: true };
        const response = await this.client.request('POST', '/v1/chat/completions', streamRequest);
        
        return this._handleStreamResponse(response, onChunk);
    }

    /**
     * 批量聊天
     */
    async batchChat(request) {
        const response = await this.client.request('POST', '/api/v1/batch/chat', request);
        return await response.json();
    }

    /**
     * 处理流式响应
     */
    async _handleStreamResponse(response, onChunk) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim());
                
                for (const line of lines) {
                    if (line === 'data: [DONE]') {
                        return;
                    }
                    
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        
                        try {
                            const parsed = JSON.parse(data);
                            if (onChunk) {
                                onChunk(parsed);
                            }
                        } catch (error) {
                            console.warn('Failed to parse stream chunk:', error);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }
}

/**
 * 补全客户端
 */
class CompletionsClient {
    constructor(client) {
        this.client = client;
    }

    /**
     * 创建文本补全
     */
    async createCompletion(request) {
        const response = await this.client.request('POST', '/v1/completions', request);
        return await response.json();
    }

    /**
     * 创建流式文本补全
     */
    async createCompletionStream(request, onChunk) {
        const streamRequest = { ...request, stream: true };
        const response = await this.client.request('POST', '/v1/completions', streamRequest);
        
        return this._handleStreamResponse(response, onChunk);
    }

    /**
     * 处理流式响应
     */
    async _handleStreamResponse(response, onChunk) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim());
                
                for (const line of lines) {
                    if (line === 'data: [DONE]') {
                        return;
                    }
                    
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        
                        try {
                            const parsed = JSON.parse(data);
                            if (onChunk) {
                                onChunk(parsed);
                            }
                        } catch (error) {
                            console.warn('Failed to parse stream chunk:', error);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }
}

/**
 * 嵌入客户端
 */
class EmbeddingsClient {
    constructor(client) {
        this.client = client;
    }

    /**
     * 创建嵌入
     */
    async createEmbedding(request) {
        const response = await this.client.request('POST', '/v1/embeddings', request);
        return await response.json();
    }

    /**
     * 批量嵌入
     */
    async batchEmbedding(request) {
        const response = await this.client.request('POST', '/api/v1/batch/embeddings', request);
        return await response.json();
    }
}

/**
 * 模型客户端
 */
class ModelsClient {
    constructor(client) {
        this.client = client;
    }

    /**
     * 列出模型
     */
    async list() {
        const response = await this.client.request('GET', '/v1/models');
        return await response.json();
    }

    /**
     * 获取模型详情
     */
    async retrieve(modelId) {
        const response = await this.client.request('GET', `/v1/models/${modelId}`);
        return await response.json();
    }
}

/**
 * 文件客户端
 */
class FilesClient {
    constructor(client) {
        this.client = client;
    }

    /**
     * 列出文件
     */
    async list() {
        const response = await this.client.request('GET', '/v1/files');
        return await response.json();
    }

    /**
     * 上传文件
     */
    async upload(file, purpose) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('purpose', purpose);

        const response = await this.client.request('POST', '/v1/files', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        
        return await response.json();
    }

    /**
     * 删除文件
     */
    async delete(fileId) {
        const response = await this.client.request('DELETE', `/v1/files/${fileId}`);
        return await response.json();
    }

    /**
     * 获取文件内容
     */
    async content(fileId) {
        const response = await this.client.request('GET', `/v1/files/${fileId}/content`);
        return await response.text();
    }
}

/**
 * 类型定义
 */

// 聊天补全请求
class ChatCompletionRequest {
    constructor(options = {}) {
        this.model = options.model || 'minimind';
        this.messages = options.messages || [];
        this.max_tokens = options.max_tokens;
        this.temperature = options.temperature;
        this.top_p = options.top_p;
        this.stream = options.stream || false;
        this.stop = options.stop;
        this.presence_penalty = options.presence_penalty;
        this.frequency_penalty = options.frequency_penalty;
        this.logit_bias = options.logit_bias;
        this.user = options.user;
    }
}

// 聊天消息
class ChatMessage {
    constructor(role, content, name) {
        this.role = role;
        this.content = content;
        this.name = name;
    }
}

// 补全请求
class CompletionRequest {
    constructor(options = {}) {
        this.model = options.model || 'minimind';
        this.prompt = options.prompt || '';
        this.max_tokens = options.max_tokens;
        this.temperature = options.temperature;
        this.top_p = options.top_p;
        this.stream = options.stream || false;
        this.stop = options.stop;
        this.presence_penalty = options.presence_penalty;
        this.frequency_penalty = options.frequency_penalty;
        this.logit_bias = options.logit_bias;
        this.user = options.user;
    }
}

// 嵌入请求
class EmbeddingRequest {
    constructor(options = {}) {
        this.model = options.model || 'minimind';
        this.input = options.input || [];
        this.user = options.user;
    }
}

// 批量请求
class BatchRequest {
    constructor(options = {}) {
        this.requests = options.requests || [];
        this.parallel = options.parallel;
    }
}

// 批量请求项
class BatchRequestItem {
    constructor(options = {}) {
        this.messages = options.messages || [];
        this.max_tokens = options.max_tokens;
    }
}

// 批量嵌入请求
class BatchEmbeddingRequest {
    constructor(options = {}) {
        this.texts = options.texts || [];
        this.batch_size = options.batch_size;
    }
}

/**
 * 导出模块
 */

// Node.js环境导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MiniMindClient,
        ChatClient,
        CompletionsClient,
        EmbeddingsClient,
        ModelsClient,
        FilesClient,
        ChatCompletionRequest,
        ChatMessage,
        CompletionRequest,
        EmbeddingRequest,
        BatchRequest,
        BatchRequestItem,
        BatchEmbeddingRequest
    };
}

// 浏览器环境导出
if (typeof window !== 'undefined') {
    window.MiniMindClient = MiniMindClient;
    window.ChatClient = ChatClient;
    window.CompletionsClient = CompletionsClient;
    window.EmbeddingsClient = EmbeddingsClient;
    window.ModelsClient = ModelsClient;
    window.FilesClient = FilesClient;
    window.ChatCompletionRequest = ChatCompletionRequest;
    window.ChatMessage = ChatMessage;
    window.CompletionRequest = CompletionRequest;
    window.EmbeddingRequest = EmbeddingRequest;
    window.BatchRequest = BatchRequest;
    window.BatchRequestItem = BatchRequestItem;
    window.BatchEmbeddingRequest = BatchEmbeddingRequest;
}

/**
 * 使用示例
 */

// 浏览器使用示例
/*
const client = new MiniMindClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your_api_key'
});

// 聊天示例
const response = await client.chat.createCompletion({
    model: 'minimind',
    messages: [
        { role: 'user', content: '你好' }
    ],
    max_tokens: 100
});

console.log(response.choices[0].message.content);

// 流式聊天示例
await client.chat.createCompletionStream({
    model: 'minimind',
    messages: [
        { role: 'user', content: '你好' }
    ],
    max_tokens: 100
}, (chunk) => {
    console.log(chunk.choices[0].delta.content);
});

// 嵌入示例
const embedding = await client.embeddings.createEmbedding({
    model: 'minimind',
    input: ['今天天气很好']
});

console.log(embedding.data[0].embedding);
*/

// Node.js使用示例
/*
const { MiniMindClient } = require('./minimind-client.js');

const client = new MiniMindClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your_api_key'
});

async function main() {
    const response = await client.chat.createCompletion({
        model: 'minimind',
        messages: [
            { role: 'user', content: '你好' }
        ],
        max_tokens: 100
    });

    console.log(response.choices[0].message.content);
}

main().catch(console.error);
*/