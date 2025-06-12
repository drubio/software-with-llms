/**
 * LLM Tester - LlamaIndex JavaScript Framework Implementation
 * ONLY LlamaIndex-specific logic, inherits all generic functionality
 */

import { 
    Anthropic, 
    OpenAI, 
    Gemini,
    ChatMessage 
} from 'llamaindex';

import { 
    getApiKey, 
    getDefaultModel, 
    BaseLLMManager, 
    interactiveCli 
} from '../utils.js';

class LlamaIndexLLMManager extends BaseLLMManager {
    constructor() {
        super('LlamaIndex');
    }

    async _testProvider(provider) {
        // Test LlamaIndex provider initialization
        await this._createClient(provider, 0.7, 1000);
    }

    _createClient(provider, temperature, maxTokens) {
        // Create LlamaIndex client - the only LlamaIndex-specific logic
        
        if (provider === 'anthropic') {
            return new Anthropic({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature: temperature,
                maxTokens: maxTokens
            });
        }
        
        else if (provider === 'openai') {
            return new OpenAI({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature: temperature,
                maxTokens: maxTokens
            });
        }
        
        else if (provider === 'google') {
            return new Gemini({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature: temperature,
                maxOutputTokens: maxTokens  // Gemini's parameter name
            });
        }
        
        else if (provider === 'xai') {
            return new OpenAI({
                apiKey: getApiKey(provider),
                apiBase: 'https://api.x.ai/v1',
                model: getDefaultModel(provider),
                temperature: temperature,
                maxTokens: maxTokens
            });
        }
        
        else {
            throw new Error(`Unsupported provider: ${provider}`);
        }
    }

    async askQuestion(topic, provider = null, template = '{topic}', maxTokens = 1000, temperature = 0.7) {
        // LlamaIndex-specific question asking
        
        const prompt = template.replace('{topic}', topic);
        
        // Use first available provider if none specified
        const availableProviders = this.getAvailableProviders();
        if (!provider || !availableProviders.includes(provider)) {
            if (availableProviders.length === 0) {
                return {
                    success: false,
                    error: 'No providers available',
                    provider: 'none',
                    model: 'none',
                    prompt: prompt,
                    response: null
                };
            }
            provider = availableProviders[0];
        }

        const model = getDefaultModel(provider);

        try {
            // Check if we're in web mode to avoid print statements
            const webMode = typeof process.stdout.write !== 'function' || process.stdout.isTTY === false;
            
            if (!webMode) {
                console.log(`Creating LlamaIndex client for ${provider} (temp=${temperature}, max_tokens=${maxTokens})`);
            }
            
            // LlamaIndex-specific: Create client
            const client = this._createClient(provider, temperature, maxTokens);
            
            if (!webMode) {
                console.log(`Making LlamaIndex call to ${provider}...`);
            }
            
            // LlamaIndex-specific: Make the call
            // LlamaIndex supports both complete() and chat() methods
            // Use chat() for better conversation handling
            let result;
            if (typeof client.chat === 'function') {
                const messages = [new ChatMessage({ role: 'user', content: prompt })];
                const response = await client.chat({ messages });
                result = response.message.content;
            } else {
                // Fallback to complete() for models that don't support chat
                const response = await client.complete(prompt);
                result = String(response);
            }
            
            if (!webMode) {
                console.log(`LlamaIndex call completed for ${provider}`);
            }
            
            return {
                success: true,
                provider: provider,
                model: model,
                prompt: prompt,
                response: result,
                temperature: temperature,
                maxTokens: maxTokens
            };
            
        } catch (error) {
            return {
                success: false,
                provider: provider,
                model: model,
                prompt: prompt,
                error: error.message,
                response: null,
                temperature: temperature,
                maxTokens: maxTokens
            };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    
    if (args.length > 0 && args[0] === 'web') {
        try {
            const { runWebServer } = await import('../web.js');
            await runWebServer(LlamaIndexLLMManager);
        } catch (error) {
            console.error('Error: web.js not found or Express not installed.');
            console.error('Install Express: npm install express cors');
            process.exit(1);
        }
    } else {
        // CLI mode - all generic logic is in utils.interactiveCli()
        const manager = new LlamaIndexLLMManager();
        await manager._checkProviders(); // Wait for provider initialization
        await interactiveCli(manager);
    }
}

// Export for web.js
export { LlamaIndexLLMManager };

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
