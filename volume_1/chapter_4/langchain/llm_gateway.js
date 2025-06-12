/**
 * LLM Tester - LangChain JavaScript Framework Implementation
 * ONLY LangChain-specific logic, inherits all generic functionality
 */

import { ChatAnthropic } from '@langchain/anthropic';
import { ChatOpenAI } from '@langchain/openai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

import { 
    getApiKey, 
    getDefaultModel, 
    BaseLLMManager, 
    interactiveCli 
} from '../utils.js';

class LangChainLLMManager extends BaseLLMManager {
    constructor() {
        super('LangChain JS');
    }

    async _testProvider(provider) {
        // Test LangChain provider initialization
        await this._createClient(provider, 0.7, 1000);
    }

    _createClient(provider, temperature, maxTokens) {
        // Create LangChain client - the only LangChain-specific logic
        
        if (provider === 'anthropic') {
            return new ChatAnthropic({
                anthropicApiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature: temperature,
                maxTokens: maxTokens
            });
        }
        
        else if (provider === 'openai') {
            return new ChatOpenAI({
                openAIApiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature: temperature,
                maxTokens: maxTokens
            });
        }
        
        else if (provider === 'google') {
            return new ChatGoogleGenerativeAI({
                apiKey: getApiKey(provider),
                model: getDefaultModel(provider),
                temperature: temperature,
                maxOutputTokens: maxTokens  // Google's parameter name
            });
        }
        
        else if (provider === 'xai') {
            return new ChatOpenAI({
                openAIApiKey: getApiKey(provider),
                configuration: {
                    baseURL: 'https://api.x.ai/v1'
                },
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
        // LangChain-specific question asking
        
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
                console.log(`Creating LangChain client for ${provider} (temp=${temperature}, max_tokens=${maxTokens})`);
            }
            
            // LangChain-specific: Create client
            const client = this._createClient(provider, temperature, maxTokens);
            
            // LangChain-specific: Create messages
            const messages = [
                new SystemMessage('You are a helpful AI assistant.'),
                new HumanMessage(prompt)
            ];
            
            if (!webMode) {
                console.log(`Making LangChain invoke() call to ${provider}...`);
            }
            
            // LangChain-specific: Make the call
            const result = await client.invoke(messages);
            
            if (!webMode) {
                console.log(`LangChain call completed for ${provider}`);
            }
            
            return {
                success: true,
                provider: provider,
                model: model,
                prompt: prompt,
                response: result.content,
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
            await runWebServer(LangChainLLMManager);
        } catch (error) {
            console.error('Error: web.js not found or Express not installed.');
            console.error('Install Express: npm install express cors');
            process.exit(1);
        }
    } else {
        // CLI mode - all generic logic is in utils.interactiveCli()
        const manager = new LangChainLLMManager();
        await manager._checkProviders(); // Wait for provider initialization
        await interactiveCli(manager);
    }
}

// Export for web.js
export { LangChainLLMManager };

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
