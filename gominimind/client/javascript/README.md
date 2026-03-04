# MiniMind JavaScript Client

A lightweight JavaScript client for interacting with the MiniMind API - a complete language model training framework.

## Features

- **OpenAI API Compatible**: Full compatibility with OpenAI Chat, Completions, and Embeddings APIs
- **Streaming Support**: Real-time streaming responses for chat completions
- **Batch Processing**: Efficient batch processing for multiple requests
- **Error Handling**: Comprehensive error handling with retry logic
- **TypeScript Ready**: Full TypeScript support with type definitions
- **Cross-Platform**: Works in browsers, Node.js, and Deno
- **Authentication**: API key authentication and security features

## Installation

### npm
```bash
npm install minimind-client
```

### yarn
```bash
yarn add minimind-client
```

### CDN (Browser)
```html
<script src="https://unpkg.com/minimind-client@1.0.0/dist/minimind-client.min.js"></script>
```

## Quick Start

### Browser Usage
```html
<!DOCTYPE html>
<html>
<head>
    <title>MiniMind Demo</title>
</head>
<body>
    <script type="module">
        import { MiniMindClient } from 'https://unpkg.com/minimind-client@1.0.0/minimind-client.js';
        
        const client = new MiniMindClient({
            baseUrl: 'http://localhost:8000',
            apiKey: 'your_api_key'
        });
        
        async function chat() {
            const response = await client.chat.createCompletion({
                model: 'minimind',
                messages: [
                    { role: 'user', content: 'Hello, how are you?' }
                ],
                max_tokens: 100
            });
            
            console.log(response.choices[0].message.content);
        }
        
        chat();
    </script>
</body>
</html>
```

### Node.js Usage
```javascript
import { MiniMindClient } from 'minimind-client';

const client = new MiniMindClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your_api_key'
});

async function main() {
    const response = await client.chat.createCompletion({
        model: 'minimind',
        messages: [
            { role: 'user', content: 'Hello, how are you?' }
        ],
        max_tokens: 100
    });
    
    console.log(response.choices[0].message.content);
}

main().catch(console.error);
```

## API Reference

### Client Configuration

```javascript
const client = new MiniMindClient({
    baseUrl: 'http://localhost:8000', // API base URL
    apiKey: 'your_api_key',           // API key (optional)
    timeout: 30000,                    // Request timeout in ms
    maxRetries: 3,                     // Maximum retry attempts
    retryDelay: 1000                   // Retry delay in ms
});
```

### Chat Completions

#### Basic Chat
```javascript
const response = await client.chat.createCompletion({
    model: 'minimind',
    messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Tell me about MiniMind project.' }
    ],
    max_tokens: 200,
    temperature: 0.7
});

console.log(response.choices[0].message.content);
```

#### Streaming Chat
```javascript
await client.chat.createCompletionStream({
    model: 'minimind',
    messages: [
        { role: 'user', content: 'Write a short story about AI.' }
    ],
    max_tokens: 500
}, (chunk) => {
    // Handle each stream chunk
    if (chunk.choices[0].delta.content) {
        process.stdout.write(chunk.choices[0].delta.content);
    }
});
```

#### Batch Chat
```javascript
const batchResponse = await client.chat.batchChat({
    requests: [
        {
            messages: [{ role: 'user', content: 'Question 1' }],
            max_tokens: 100
        },
        {
            messages: [{ role: 'user', content: 'Question 2' }],
            max_tokens: 150
        }
    ],
    parallel: 2
});
```

### Text Completions

```javascript
const response = await client.completions.createCompletion({
    model: 'minimind',
    prompt: 'The future of artificial intelligence',
    max_tokens: 100,
    temperature: 0.8
});

console.log(response.choices[0].text);
```

### Embeddings

#### Single Text
```javascript
const embedding = await client.embeddings.createEmbedding({
    model: 'minimind',
    input: 'This is a sample text for embedding.'
});

console.log(embedding.data[0].embedding);
```

#### Multiple Texts
```javascript
const embeddings = await client.embeddings.createEmbedding({
    model: 'minimind',
    input: [
        'First text for embedding',
        'Second text for embedding'
    ]
});

console.log(embeddings.data.length); // 2
```

#### Batch Embeddings
```javascript
const batchEmbeddings = await client.embeddings.batchEmbedding({
    texts: [
        'Text 1', 'Text 2', 'Text 3', 'Text 4'
    ],
    batch_size: 2
});
```

### Model Management

#### List Models
```javascript
const models = await client.models.list();
console.log(models.data);
```

#### Get Model Details
```javascript
const model = await client.models.retrieve('minimind');
console.log(model);
```

### File Management

#### Upload File
```javascript
const fileInput = document.getElementById('fileInput');
const file = fileInput.files[0];

const uploadResult = await client.files.upload(file, 'fine-tune');
console.log(uploadResult.id);
```

#### List Files
```javascript
const files = await client.files.list();
console.log(files.data);
```

### Health Check

```javascript
const health = await client.health();
console.log(health.status); // 'healthy' or 'unhealthy'
```

## Advanced Usage

### Custom Headers

```javascript
const response = await client.request('POST', '/v1/chat/completions', data, {
    headers: {
        'X-Custom-Header': 'custom-value'
    }
});
```

### Error Handling

```javascript
try {
    const response = await client.chat.createCompletion({
        model: 'minimind',
        messages: [{ role: 'user', content: 'Hello' }]
    });
} catch (error) {
    console.error('Error:', error.message);
    console.error('Status:', error.status);
    console.error('Code:', error.code);
    
    if (error.status === 429) {
        console.log('Rate limit exceeded, please try again later');
    }
}
```

### Request Options

```javascript
const response = await client.request('GET', '/v1/models', null, {
    query: { limit: 10, offset: 0 },
    timeout: 60000,
    headers: { 'X-Request-ID': '12345' }
});
```

## Type Definitions

### Chat Completion Request

```typescript
interface ChatCompletionRequest {
    model: string;
    messages: Array<{
        role: 'system' | 'user' | 'assistant';
        content: string;
        name?: string;
    }>;
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stream?: boolean;
    stop?: string[];
    presence_penalty?: number;
    frequency_penalty?: number;
    logit_bias?: Record<string, number>;
    user?: string;
}
```

### Chat Completion Response

```typescript
interface ChatCompletionResponse {
    id: string;
    object: 'chat.completion';
    created: number;
    model: string;
    choices: Array<{
        index: number;
        message: {
            role: 'assistant';
            content: string;
        };
        finish_reason: 'stop' | 'length' | 'content_filter';
    }>;
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}
```

### Embedding Request

```typescript
interface EmbeddingRequest {
    model: string;
    input: string | string[];
    user?: string;
}
```

### Embedding Response

```typescript
interface EmbeddingResponse {
    object: 'list';
    data: Array<{
        object: 'embedding';
        embedding: number[];
        index: number;
    }>;
    model: string;
    usage: {
        prompt_tokens: number;
        total_tokens: number;
    };
}
```

## Examples

### React Component

```jsx
import React, { useState } from 'react';
import { MiniMindClient } from 'minimind-client';

function ChatComponent() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    
    const client = new MiniMindClient({
        baseUrl: 'http://localhost:8000'
    });
    
    const sendMessage = async () => {
        if (!input.trim()) return;
        
        setLoading(true);
        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        
        try {
            const response = await client.chat.createCompletion({
                model: 'minimind',
                messages: [...messages, userMessage],
                max_tokens: 200
            });
            
            const assistantMessage = response.choices[0].message;
            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error('Chat error:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.role}`}>
                        {msg.content}
                    </div>
                ))}
            </div>
            <div className="input-area">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    disabled={loading}
                />
                <button onClick={sendMessage} disabled={loading}>
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
}
```

### Node.js CLI Tool

```javascript
import { MiniMindClient } from 'minimind-client';
import readline from 'readline';

const client = new MiniMindClient({
    baseUrl: 'http://localhost:8000'
});

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function chatLoop() {
    const messages = [];
    
    while (true) {
        const userInput = await new Promise(resolve => {
            rl.question('You: ', resolve);
        });
        
        if (userInput.toLowerCase() === 'quit') {
            break;
        }
        
        messages.push({ role: 'user', content: userInput });
        
        const response = await client.chat.createCompletion({
            model: 'minimind',
            messages,
            max_tokens: 200
        });
        
        const assistantMessage = response.choices[0].message;
        messages.push(assistantMessage);
        
        console.log('Assistant:', assistantMessage.content);
    }
    
    rl.close();
}

chatLoop().catch(console.error);
```

## Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Verify API key |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Verify endpoint URL |
| 429 | Too Many Requests | Reduce request frequency |
| 500 | Internal Server Error | Check server logs |
| 503 | Service Unavailable | Service is down or overloaded |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: [MiniMind Docs](https://github.com/jingyaogong/minimind)
- Issues: [GitHub Issues](https://github.com/jingyaogong/minimind/issues)
- Discussions: [GitHub Discussions](https://github.com/jingyaogong/minimind/discussions)