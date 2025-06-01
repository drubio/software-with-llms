/**
 * utils.js - Common utilities and configurations shared across all JavaScript frameworks
 */

import 'dotenv/config';
import fs from 'fs';
import readline from 'readline';
import { config } from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Get the directory of this utils.js file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env from the same directory as utils.js
config({ path: join(__dirname, '.env') });

// Provider configurations
export const PROVIDERS = {
    anthropic: {
        apiKeyEnv: 'ANTHROPIC_API_KEY',
        defaultModel: 'claude-3-5-sonnet-20241022',
        displayName: 'Anthropic Claude'
    },
    openai: {
        apiKeyEnv: 'OPENAI_API_KEY',
        defaultModel: 'gpt-4o',
        displayName: 'OpenAI GPT'
    },
    google: {
        apiKeyEnv: 'GOOGLE_API_KEY',
        defaultModel: 'gemini-2.0-flash',
        displayName: 'Google Gemini'
    },
    xai: {
        apiKeyEnv: 'XAI_API_KEY',
        defaultModel: 'grok-3',
        displayName: 'xAI Grok'
    }
};

export function getApiKey(provider) {
    if (provider in PROVIDERS) {
        return process.env[PROVIDERS[provider].apiKeyEnv];
    }
    return null;
}

export function getDefaultModel(provider) {
    return PROVIDERS[provider]?.defaultModel || '';
}

export function getDisplayName(provider) {
    return PROVIDERS[provider]?.displayName || provider.charAt(0).toUpperCase() + provider.slice(1);
}

export function getAllProviders() {
    return Object.keys(PROVIDERS);
}

export function getAvailableProviders() {
    const available = [];
    for (const providerName of Object.keys(PROVIDERS)) {
        if (getApiKey(providerName)) {
            available.push(providerName);
        }
    }
    return available;
}

export async function getUserParameters() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const question = (prompt) => new Promise((resolve) => {
        rl.question(prompt, resolve);
    });

    try {
        // Ask for temperature
        const tempInput = await question('Temperature (0.0-2.0, default 0.7): ');
        let temperature = 0.7;
        if (tempInput.trim()) {
            const parsed = parseFloat(tempInput);
            if (!isNaN(parsed)) {
                temperature = Math.max(0.0, Math.min(2.0, parsed));
            } else {
                console.log('Invalid temperature, using default: 0.7');
            }
        }

        // Ask for max tokens
        const tokensInput = await question('Max tokens (default 1000): ');
        let maxTokens = 1000;
        if (tokensInput.trim()) {
            const parsed = parseInt(tokensInput);
            if (!isNaN(parsed)) {
                maxTokens = Math.max(1, Math.min(4000, parsed));
            } else {
                console.log('Invalid max tokens, using default: 1000');
            }
        }

        rl.close();
        return { temperature, maxTokens };
    } catch (error) {
        rl.close();
        throw error;
    }
}

export function saveResponseToFile(response, filename) {
    fs.writeFileSync(filename, JSON.stringify(response, null, 2));
    console.log(`Response saved to ${filename}`);
}

export function displayProviderResponse(provider, response, framework = '') {
    const frameworkSuffix = framework ? ` (${framework})` : '';
    const providerDisplay = `${getDisplayName(provider)}${frameworkSuffix} answered:`;
    
    console.log(`\n=== ${providerDisplay} ===`);
    
    // Show configuration if available
    const configParts = [];
    if (response.temperature !== undefined) {
        configParts.push(`temp: ${response.temperature}`);
    }
    if (response.maxTokens !== undefined) {
        configParts.push(`max_tokens: ${response.maxTokens}`);
    }
    if (response.model) {
        configParts.push(`model: ${response.model}`);
    }
    
    if (configParts.length > 0) {
        console.log(`[${configParts.join(', ')}]`);
    }
    
    if (response.success) {
        console.log(response.response || 'No response');
    } else {
        console.log(`Error: ${response.error || 'Unknown error'}`);
    }
    console.log('='.repeat(60));
}

export function printInitializationStatus(framework, messages) {
    console.log(`\n=== ${framework} Framework - Provider Status ===`);
    for (const [provider, message] of Object.entries(messages)) {
        console.log(`${getDisplayName(provider)}: ${message}`);
    }
    console.log('='.repeat(50) + '\n');
}

export async function getUserChoice(options, prompt) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    console.log(`\n${prompt}`);
    options.forEach((option, i) => {
        console.log(`${i + 1}. ${option}`);
    });

    while (true) {
        try {
            const answer = await new Promise((resolve) => {
                rl.question(`Select an option (1-${options.length}): `, resolve);
            });
            
            const choice = parseInt(answer) - 1;
            if (choice >= 0 && choice < options.length) {
                rl.close();
                return choice;
            } else {
                console.log('Invalid selection. Please try again.');
            }
        } catch (error) {
            console.log('Invalid input. Please enter a number.');
        }
    }
}

export function formatFilename(question, framework) {
    const safeQuestion = question.slice(0, 20).replace(/\s+/g, '_').replace(/[?!]/g, '');
    return `llm_responses_${framework}_${safeQuestion}.json`;
}

/**
 * Base class for LLM framework managers - handles all generic logic
 */
export class BaseLLMManager {
    constructor(frameworkName) {
        this.framework = frameworkName;
        this.initializationMessages = {};
        this._checkProviders();
    }

    async _checkProviders() {
        for (const provider of getAllProviders()) {
            if (getApiKey(provider)) {
                try {
                    await this._testProvider(provider);
                    this.initializationMessages[provider] = '✓ Initialized successfully';
                } catch (error) {
                    this.initializationMessages[provider] = `✗ Failed: ${error.message}`;
                }
            } else {
                this.initializationMessages[provider] = '✗ API key not found';
            }
        }
    }

    async _testProvider(provider) {
        throw new Error('Subclasses must implement _testProvider');
    }

    getAvailableProviders() {
        const available = [];
        for (const [provider, status] of Object.entries(this.initializationMessages)) {
            if (status.startsWith('✓')) {
                available.push(provider);
            }
        }
        return available;
    }

    displayInitializationStatus() {
        printInitializationStatus(this.framework, this.initializationMessages);
    }

    async askQuestion(topic, provider = null, template = '{topic}', maxTokens = 1000, temperature = 0.7) {
        throw new Error('Subclasses must implement askQuestion');
    }

    async queryAllProviders(topic, template = '{topic}', maxTokens = 1000, temperature = 0.7) {
        const availableProviders = this.getAvailableProviders();
        
        if (availableProviders.length === 0) {
            return {
                success: false,
                error: 'No providers available',
                prompt: template.replace('{topic}', topic),
                responses: {}
            };
        }

        const responses = {};
        for (const provider of availableProviders) {
            console.log(`Querying ${getDisplayName(provider)} via ${this.framework}...`);
            const response = await this.askQuestion(topic, provider, template, maxTokens, temperature);
            responses[provider] = response;
        }

        return {
            success: true,
            prompt: template.replace('{topic}', topic),
            responses
        };
    }
}

/**
 * Generic interactive CLI that works with any LLM manager
 */
export async function interactiveCli(manager) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const question = (prompt) => new Promise((resolve) => {
        rl.question(prompt, resolve);
    });

    try {
        console.log('='.repeat(60));
        console.log(`LLM Tester - ${manager.framework} Framework`);
        console.log('='.repeat(60));
        
        manager.displayInitializationStatus();
        
        const availableProviders = manager.getAvailableProviders();
        
        if (availableProviders.length === 0) {
            console.log('No providers available. Check your .env file.');
            return;
        }
        
        const topic = await question('What topic do you want to ask about? ');
        
        // Get temperature
        const tempInput = await question('Temperature (0.0-2.0, default 0.7): ');
        let temperature = 0.7;
        if (tempInput.trim()) {
            const parsed = parseFloat(tempInput);
            if (!isNaN(parsed)) {
                temperature = Math.max(0.0, Math.min(2.0, parsed));
            } else {
                console.log('Invalid temperature, using default: 0.7');
            }
        }

        // Get max tokens
        const tokensInput = await question('Max tokens (default 1000): ');
        let maxTokens = 1000;
        if (tokensInput.trim()) {
            const parsed = parseInt(tokensInput);
            if (!isNaN(parsed)) {
                maxTokens = Math.max(1, Math.min(4000, parsed));
            } else {
                console.log('Invalid max tokens, using default: 1000');
            }
        }
        
        console.log(`\nUsing temperature: ${temperature}, max tokens: ${maxTokens}`);
        
        if (availableProviders.length > 1) {
            console.log(`\nAvailable providers: ${availableProviders.map(p => getDisplayName(p)).join(', ')}`);
            const queryAll = await question('Query ALL providers or select one? (all/one): ');
            
            if (['all', 'a', ''].includes(queryAll.toLowerCase())) {
                console.log('\n' + '='.repeat(50));
                console.log(`${manager.framework.toUpperCase()} API CALLS - QUERYING ALL PROVIDERS`);
                console.log('='.repeat(50));
                
                const results = await manager.queryAllProviders(topic, '{topic}', maxTokens, temperature);
                
                if (results.success) {
                    for (const [provider, response] of Object.entries(results.responses)) {
                        displayProviderResponse(provider, response, manager.framework);
                    }
                } else {
                    console.log(`Error: ${results.error}`);
                }
                
                const saveOption = await question('\nSave results? (y/n): ');
                if (['y', 'yes'].includes(saveOption.toLowerCase())) {
                    const filename = formatFilename(topic, manager.framework.toLowerCase());
                    saveResponseToFile(results, filename);
                }
            } else {
                console.log('\nSelect a provider:');
                const providerNames = availableProviders.map(p => getDisplayName(p));
                providerNames.forEach((name, i) => {
                    console.log(`${i + 1}. ${name}`);
                });
                
                const choiceInput = await question(`Select an option (1-${providerNames.length}): `);
                const choiceIdx = parseInt(choiceInput) - 1;
                
                if (choiceIdx >= 0 && choiceIdx < availableProviders.length) {
                    const provider = availableProviders[choiceIdx];
                    
                    console.log(`\n${'='.repeat(50)}`);
                    console.log(`${manager.framework.toUpperCase()} API CALL - ${getDisplayName(provider).toUpperCase()}`);
                    console.log('='.repeat(50));
                    
                    const result = await manager.askQuestion(topic, provider, '{topic}', maxTokens, temperature);
                    displayProviderResponse(provider, result, manager.framework);
                } else {
                    console.log('Invalid selection.');
                }
            }
        } else {
            const provider = availableProviders[0];
            console.log(`\nUsing only available provider: ${getDisplayName(provider)}`);
            
            console.log(`\n${'='.repeat(50)}`);
            console.log(`${manager.framework.toUpperCase()} API CALL - ${getDisplayName(provider).toUpperCase()}`);
            console.log('='.repeat(50));
            
            const result = await manager.askQuestion(topic, provider, '{topic}', maxTokens, temperature);
            displayProviderResponse(provider, result, manager.framework);
        }
        
        console.log(`\nThank you for using the ${manager.framework} LLM Tester!`);
    } catch (error) {
        console.error('Error:', error.message);
    } finally {
        rl.close();
    }
}
