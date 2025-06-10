/**
 * web.js - Clean web interface for JavaScript LLM testers
 * Provides clean JSON responses without CLI artifacts
 */

import express from 'express';
import cors from 'cors';
import { getDisplayName, getDefaultModel } from './utils.js';

// Capture console output for web responses
function captureConsoleOutput(fn) {
    const originalLog = console.log;
    const logs = [];
    
    console.log = (...args) => {
        logs.push(args.join(' '));
    };
    
    try {
        const result = fn();
        return { result, logs: logs.join('\n') };
    } finally {
        console.log = originalLog;
    }
}

async function captureConsoleOutputAsync(fn) {
    const originalLog = console.log;
    const logs = [];
    
    console.log = (...args) => {
        logs.push(args.join(' '));
    };
    
    try {
        const result = await fn();
        return { result, logs: logs.join('\n') };
    } finally {
        console.log = originalLog;
    }
}

function createWebApi(ManagerClass) {
    const app = express();
    
    // Middleware
    app.use(cors());
    app.use(express.json());
    
    // Initialize the manager
    let manager;
    
    // Initialize manager asynchronously
    const initPromise = (async () => {
        manager = new ManagerClass();
        await manager._checkProviders();
        return manager;
    })();
    
    // Middleware to ensure manager is initialized
    app.use(async (req, res, next) => {
        if (!manager) {
            await initPromise;
        }
        next();
    });
    
    // Get service status
    app.get('/', async (req, res) => {
        try {
            const availableProviders = manager.getAvailableProviders();
            
            res.json({
                framework: manager.framework,
                available_providers: availableProviders,
                total_available: availableProviders.length,
                initialization_status: manager.initializationMessages,
                status: availableProviders.length > 0 ? 'healthy' : 'no_providers'
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
    
    // Get available providers with details
    app.get('/providers', async (req, res) => {
        try {
            const availableProviders = manager.getAvailableProviders();
            
            const providersDetail = availableProviders.map(provider => ({
                name: provider,
                display_name: getDisplayName(provider),
                model: getDefaultModel(provider),
                status: manager.initializationMessages[provider] || 'Unknown'
            }));
            
            res.json({
                framework: manager.framework,
                providers: providersDetail,
                count: availableProviders.length
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
    
    // Query single provider
    app.post('/query', async (req, res) => {
        try {
            const {
                topic,
                provider = null,
                template = '{topic}',
                max_tokens = 1000,
                temperature = 0.7
            } = req.body;
            
            if (!topic) {
                return res.status(400).json({ error: 'Topic is required' });
            }
            
            // Capture any console output
            const { result, logs } = await captureConsoleOutputAsync(async () => {
                return await manager.askQuestion(topic, provider, template, max_tokens, temperature);
            });
            
            if (!result.success) {
                return res.status(400).json({
                    error: result.error || 'Query failed',
                    provider: result.provider,
                    debug: logs || null
                });
            }
            
            // Clean response for web
            const cleanResult = {
                success: true,
                framework: manager.framework,
                provider: result.provider,
                model: result.model,
                response: result.response,
                parameters: {
                    temperature: result.temperature,
                    max_tokens: result.maxTokens,
                    template: template
                },
                prompt: result.prompt
            };
            
            // Optionally include debug info
            if (logs) {
                cleanResult.debug = logs;
            }
            
            res.json(cleanResult);
            
        } catch (error) {
            res.status(500).json({
                error: error.message,
                framework: manager.framework
            });
        }
    });
    
    // Query all available providers
    app.post('/query-all', async (req, res) => {
        try {
            const {
                topic,
                template = '{topic}',
                max_tokens = 1000,
                temperature = 0.7
            } = req.body;
            
            if (!topic) {
                return res.status(400).json({ error: 'Topic is required' });
            }
            
            // Capture any console output
            const { result, logs } = await captureConsoleOutputAsync(async () => {
                return await manager.queryAllProviders(topic, template, max_tokens, temperature);
            });
            
            if (!result.success) {
                return res.status(400).json({
                    error: result.error || 'Query failed',
                    framework: manager.framework,
                    debug: logs || null
                });
            }
            
            // Clean up the responses for web
            const cleanResponses = {};
            let successful = 0;
            let failed = 0;
            
            for (const [provider, response] of Object.entries(result.responses)) {
                if (response.success) {
                    cleanResponses[provider] = {
                        success: true,
                        response: response.response,
                        model: response.model,
                        parameters: {
                            temperature: response.temperature,
                            max_tokens: response.maxTokens
                        }
                    };
                    successful++;
                } else {
                    cleanResponses[provider] = {
                        success: false,
                        error: response.error || 'Unknown error',
                        model: response.model || 'unknown'
                    };
                    failed++;
                }
            }
            
            const cleanResult = {
                success: true,
                framework: manager.framework,
                prompt: result.prompt,
                responses: cleanResponses,
                summary: {
                    total_providers: Object.keys(result.responses).length,
                    successful: successful,
                    failed: failed
                },
                parameters: {
                    temperature: temperature,
                    max_tokens: max_tokens,
                    template: template
                }
            };
            
            // Optionally include debug info
            if (logs) {
                cleanResult.debug = logs;
            }
            
            res.json(cleanResult);
            
        } catch (error) {
            res.status(500).json({
                error: error.message,
                framework: manager.framework
            });
        }
    });
    
    // Simple health check
    app.get('/health', async (req, res) => {
        try {
            const availableProviders = manager.getAvailableProviders();
            res.json({
                status: availableProviders.length > 0 ? 'healthy' : 'unhealthy',
                framework: manager.framework,
                providers_available: availableProviders.length
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
    
    return app;
}

export async function runWebServer(ManagerClass, host = '0.0.0.0', port = 8000) {
    const app = createWebApi(ManagerClass);
    
    // Get framework name for display
    let frameworkName = 'Unknown';
    try {
        const tempManager = new ManagerClass();
        frameworkName = tempManager.framework;
    } catch (error) {
        // Ignore
    }
    
    app.listen(port, host, () => {
        console.log(`Starting web server for ${frameworkName} framework...`);
        console.log(`API documentation available at: http://${host}:${port}/`);
        console.log(`Health check: http://${host}:${port}/health`);
        console.log(`Status: http://${host}:${port}/`);
        console.log('Press Ctrl+C to stop the server');
    });
}

// Main function for standalone web server
function main() {
    console.log('Universal LLM Web API (JavaScript)');
    console.log('This should be called from an LLM tester script like:');
    console.log('node llm_tester.js web');
}

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
