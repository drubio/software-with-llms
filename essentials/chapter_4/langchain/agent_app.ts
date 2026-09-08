/**
 * LLM application to chat with multiple LLMs - LangChain TypeScript framework implementation.
 */

import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import {
    BaseLLMManager,
    createLangChainModel,
    interactiveCli,
    normalizeResponseText,
    printCliHelp,
} from '../../../shared/utils.ts';

type LangChainResponse = {
    content?: unknown;
    text?: unknown;
};

class LangChainLLMManager extends BaseLLMManager {
    constructor() {
        super('LangChain TypeScript');
    }

    async _testProvider(provider: string): Promise<void> {
        await this._createModel(this.providerModelIdentifier(provider), 0.7, 1000);
    }

    _createModel(selectedModel: string, temperature: number, maxTokens: number) {
        return createLangChainModel(selectedModel, {
            temperature,
            maxTokens,
        });
    }

    _buildMessages(prompt: string) {
        return [
            new SystemMessage('You are a helpful AI assistant.'),
            new HumanMessage(prompt),
        ];
    }

    _extractText(provider: string, result: LangChainResponse): string {
        if (provider === 'google' && typeof result?.text !== 'undefined') {
            return normalizeResponseText(result.text);
        }
        return normalizeResponseText(result?.content ?? result);
    }

    async askQuestion(
        topic: string,
        provider: string | null = null,
        template = '{topic}',
        maxTokens = 1000,
        temperature = 0.7,
    ) {
        const prompt = template.replace('{topic}', topic);
        const modelConfig = this.resolveModelConfig(provider);

        if (!modelConfig) {
            return {
                success: false,
                error: 'No providers available',
                provider: 'none',
                model: 'none',
                prompt,
                response: null,
            };
        }

        try {
            const model = this._createModel(modelConfig.name, temperature, maxTokens);
            const messages = this._buildMessages(prompt);
            const response = this._extractText(modelConfig.provider, await model.invoke(messages));
            return {
                success: true,
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
                prompt,
                response,
                temperature,
                maxTokens,
            };
        } catch (error: unknown) {
            return {
                success: false,
                provider: modelConfig.provider,
                model: modelConfig.model,
                modelIdentifier: modelConfig.name,
                prompt,
                error: error instanceof Error ? error.message : String(error),
                response: null,
                temperature,
                maxTokens,
            };
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    if (args.includes('-h') || args.includes('--help')) {
        printCliHelp(process.argv[1]);
        return;
    }

    if (args.includes('web')) {
        try {
            const { runWebServer } = await import('../../../shared/essentials/web.ts');
            await runWebServer(() => new LangChainLLMManager());
        } catch (error) {
            console.error('Error: shared web API not found or Express not installed.');
            console.error('Install Express: npm install express cors');
            process.exit(1);
        }
    } else {
        const manager = new LangChainLLMManager();
        await manager._checkProviders();
        await interactiveCli(manager);
    }
}

export { LangChainLLMManager };

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
