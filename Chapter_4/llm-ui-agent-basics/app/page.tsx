'use client';

import React, { useState, useEffect } from 'react';
import { Settings } from 'lucide-react';

// Framework imports
import { useStream } from '@langchain/langgraph-sdk/react';
import { ChatSection, ChatMessages, ChatInput } from '@llamaindex/chat-ui';
import { useChat } from 'ai/react';
import { 
  ThreadPrimitive, 
  MessagePrimitive, 
  ComposerPrimitive,
  AssistantRuntimeProvider, 
  useLocalRuntime,
  type ChatModelAdapter
} from '@assistant-ui/react';

// ============================================================================
// SHARED HOOKS AND UTILITIES
// ============================================================================

// Custom hook for API settings and providers
const useAPISettings = () => {
  const [providers, setProviders] = useState([]);
  const [settings, setSettings] = useState({
    queryMode: 'single',
    selectedProvider: '',
    temperature: 0.7,
    maxTokens: 1000
  });

  useEffect(() => {
    fetch('http://localhost:8000/providers')
      .then(res => res.json())
      .then(data => {
        setProviders(data.providers || []);
        if (data.providers?.length > 0) {
          setSettings(prev => ({ ...prev, selectedProvider: data.providers[0].name }));
        }
      })
      .catch(() => setProviders([]));
  }, []);

  return { providers, settings, setSettings };
};

// Shared API call logic
const callAPI = async (message, settings) => {
  const endpoint = settings.queryMode === 'single' ? '/query' : '/query-all';
  const payload = {
    topic: message,
    temperature: settings.temperature,
    max_tokens: settings.maxTokens,
    template: '{topic}',
    ...(settings.queryMode === 'single' && { provider: settings.selectedProvider })
  };

  const response = await fetch(`http://localhost:8000${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  
  if (settings.queryMode === 'single') {
    return data.success ? data.response : `Error: ${data.error}`;
  } else {
    if (data.success) {
      let content = `Results from ${data.summary.total_providers} providers:\n\n`;
      for (const [provider, response] of Object.entries(data.responses)) {
        content += `**${provider}**: ${response.success ? response.response : response.error}\n\n`;
      }
      return content;
    } else {
      return `Error: ${data.error}`;
    }
  }
};

// Shared Settings Sidebar Component
const SettingsSidebar = ({ isOpen, onClose, settings, onSettingsChange, providers }) => (
  <div className={`fixed right-0 top-0 h-full w-80 bg-white border-l shadow-xl transform transition-transform z-50 ${
    isOpen ? 'translate-x-0' : 'translate-x-full'
  }`}>
    <div className="p-4 border-b bg-gray-50">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold">API Settings</h2>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-700">✕</button>
      </div>
    </div>
    <div className="p-4 space-y-4 overflow-y-auto">
      <div>
        <label className="block text-sm font-medium mb-2">Query Mode</label>
        <div className="space-y-2">
          <label className="flex items-center">
            <input
              type="radio"
              value="single"
              checked={settings.queryMode === 'single'}
              onChange={(e) => onSettingsChange({ ...settings, queryMode: e.target.value })}
              className="text-blue-600"
            />
            <span className="ml-2 text-sm">Single Provider</span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              value="all"
              checked={settings.queryMode === 'all'}
              onChange={(e) => onSettingsChange({ ...settings, queryMode: e.target.value })}
              className="text-blue-600"
            />
            <span className="ml-2 text-sm">All Providers</span>
          </label>
        </div>
      </div>

      {settings.queryMode === 'single' && (
        <div>
          <label className="block text-sm font-medium mb-2">Provider</label>
          <select
            value={settings.selectedProvider}
            onChange={(e) => onSettingsChange({ ...settings, selectedProvider: e.target.value })}
            className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
          >
            {providers.map(p => (
              <option key={p.name} value={p.name}>{p.display_name}</option>
            ))}
          </select>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium mb-2">Temperature: {settings.temperature}</label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={settings.temperature}
          onChange={(e) => onSettingsChange({ ...settings, temperature: parseFloat(e.target.value) })}
          className="w-full"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Max Tokens</label>
        <input
          type="number"
          min="1"
          max="4000"
          value={settings.maxTokens || ''}
          onChange={(e) => {
            const value = e.target.value;
            if (value === '' || (!isNaN(Number(value)) && Number(value) > 0)) {
              onSettingsChange({ 
                ...settings, 
                maxTokens: value === '' ? 1000 : parseInt(value, 10) 
              });
            }
          }}
          placeholder="1000"
          className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  </div>
);

// Shared Header Component
const FrameworkHeader = ({ title, color, settings, onSettingsClick }) => (
  <div className={`bg-${color}-600 text-white p-3 flex justify-between items-center`}>
    <div>
      <h1 className="text-lg font-semibold">{title}</h1>
      <div className="text-xs opacity-75">
        Mode: {settings.queryMode} | Provider: {settings.selectedProvider} | Temp: {settings.temperature} | Max Tokens: {settings.maxTokens}
      </div>
    </div>
    <button 
      onClick={onSettingsClick}
      className={`p-2 hover:bg-${color}-700 rounded`}
    >
      <Settings className="w-4 h-4" />
    </button>
  </div>
);

// ============================================================================
// FRAMEWORK COMPONENTS
// ============================================================================

// LangChain Component
const LangChainPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings } = useAPISettings();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m your LangChain assistant connected to your API.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input?.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setInput('');

    try {
      const content = await callAPI(input, settings);
      setMessages(prev => [...prev, { role: 'assistant', content }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Connection Error: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="LangChain Agent UI (Real Components)"
        color="blue"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
      />
      
      <div className="flex-1 overflow-hidden p-4">
        <div className="h-full bg-white rounded-lg border overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto p-4">
            {messages?.map((message, i) => (
              <div key={i} className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                <div className={`inline-block max-w-md px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-900 whitespace-pre-wrap'
                }`}>
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="text-left mb-4">
                <div className="inline-block bg-gray-100 px-4 py-2 rounded-lg">
                  <div className="animate-pulse">LangChain agent processing...</div>
                </div>
              </div>
            )}
          </div>

          <div className="border-t p-4">
            <form onSubmit={handleSubmit}>
              <div className="flex space-x-2">
                <input
                  value={input || ''}
                  onChange={(e) => setInput(e.target.value)}
                  type="text"
                  placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}		  
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input?.trim()}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-md"
                >
                  Send
                </button>
              </div>
            </form>
            <div className="text-xs text-gray-500 mt-2">
              Powered by @langchain/langgraph-sdk • Connected to localhost:8000
            </div>
          </div>
        </div>
      </div>

      <SettingsSidebar
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSettingsChange={setSettings}
        providers={providers}
      />

      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40" onClick={() => setShowSettings(false)} />
      )}
    </div>
  );
};

// LlamaIndex Component
const LlamaIndexPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings } = useAPISettings();

  const chat = useChat({
    api: '/api/llamaindex-chat',
    body: {
      queryMode: settings.queryMode,
      selectedProvider: settings.selectedProvider,
      temperature: settings.temperature,
      maxTokens: settings.maxTokens,
      template: '{topic}'
    },
    onError: (error) => console.error('LlamaIndex chat error:', error)
  });

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="LlamaIndex Chat UI (Real Components)"
        color="purple"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
      />
      
      <div className="flex-1 overflow-hidden">
        <ChatSection handler={chat} className="h-full">
          <ChatMessages className="flex-1 overflow-y-auto" />
          <ChatInput className="border-t">
            <ChatInput.Form className="p-4">
              <ChatInput.Field 
                type="textarea" 
                placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <div className="flex justify-between items-center mt-2">
                <ChatInput.Submit className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg ml-2">
                  Send
                </ChatInput.Submit>
              </div>
            </ChatInput.Form>
	      <div className="text-xs text-gray-500">
                  Powered by @llamaindex/chat-ui • {settings.queryMode === 'single' ? `Using ${settings.selectedProvider}` : 'Using all providers'}
              </div>	    
          </ChatInput>
        </ChatSection>
      </div>

      <SettingsSidebar
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSettingsChange={setSettings}
        providers={providers}
      />

      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40" onClick={() => setShowSettings(false)} />
      )}
    </div>
  );
};

// Assistant UI Component
const AssistantUIPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings } = useAPISettings();

  // Assistant UI adapter for your API
  const modelAdapter: ChatModelAdapter = {
    async run({ messages, abortSignal }) {
      const lastMessage = messages[messages.length - 1];
      
      try {
        // Extract text content properly from the message
        let messageText = '';
        if (lastMessage.content) {
          if (Array.isArray(lastMessage.content)) {
            // Handle content array format
            const textContent = lastMessage.content.find(c => c.type === 'text');
            messageText = textContent ? textContent.text : '';
          } else if (typeof lastMessage.content === 'string') {
            // Handle string format
            messageText = lastMessage.content;
          }
        }

        if (!messageText) {
          throw new Error('No text content found in message');
        }

        const endpoint = settings.queryMode === 'single' ? '/query' : '/query-all';
        const payload = {
          topic: messageText, // Use extracted text instead of full content
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          template: '{topic}',
          ...(settings.queryMode === 'single' && { provider: settings.selectedProvider })
        };

        console.log('Assistant UI API call:', { endpoint, payload }); // Debug log

        const response = await fetch(`http://localhost:8000${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: abortSignal
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        
        let content = '';
        if (settings.queryMode === 'single') {
          content = data.success ? data.response : `Error: ${data.error}`;
        } else {
          if (data.success) {
            content = `🤖 Assistant UI Results from ${data.summary.total_providers} providers:\n\n`;
            for (const [provider, response] of Object.entries(data.responses)) {
              content += `**${provider}**: ${response.success ? response.response : response.error}\n\n`;
            }
          } else {
            content = `Error: ${data.error}`;
          }
        }

        console.log('Assistant UI API response:', content); // Debug log

        return {
          content: [
            {
              type: "text",
              text: content
            }
          ]
        };
      } catch (error) {
        console.error('Assistant UI API error:', error); // Debug log
        return {
          content: [
            {
              type: "text", 
              text: `Connection Error: ${error.message}`
            }
          ]
        };
      }
    }
  };

  // Assistant UI runtime using useLocalRuntime
  const runtime = useLocalRuntime(modelAdapter);

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="Assistant UI (Real Components)"
        color="green"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
      />
      
      <div className="flex-1 overflow-hidden">
        {/* Assistant UI Components */}
        <AssistantRuntimeProvider runtime={runtime}>
          <ThreadPrimitive.Root className="h-full bg-gradient-to-b from-green-50 to-white flex flex-col">
            <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto p-4">
              <ThreadPrimitive.Empty>
                <div className="text-center text-gray-500 mt-8">
                  Hello! I'm your Assistant UI powered by real @assistant-ui/react components.
                </div>
              </ThreadPrimitive.Empty>
              <ThreadPrimitive.Messages 
                components={{
                  UserMessage: ({ children }) => (
                    <div className="mb-4 text-right">
                      <div className="bg-green-600 text-white px-4 py-2 rounded-lg inline-block max-w-md">
                        <MessagePrimitive.Content />
                      </div>
                    </div>
                  ),
                  AssistantMessage: ({ children }) => (
                    <div className="mb-4 text-left">
                      <div className="bg-gray-100 text-gray-900 px-4 py-2 rounded-lg inline-block max-w-md whitespace-pre-wrap">
                        <MessagePrimitive.Content />
                      </div>
                    </div>
                  )
                }}
              />
            </ThreadPrimitive.Viewport>
            <div className="border-t p-4">
              <ComposerPrimitive.Root>
                <div className="flex space-x-2">
                  <ComposerPrimitive.Input 
                    placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                    className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-500"
                  />
                  <ComposerPrimitive.Send className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg">
                    Send
                  </ComposerPrimitive.Send>
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  Powered by @assistant-ui/react • Real streaming conversations
                </div>
              </ComposerPrimitive.Root>
            </div>
          </ThreadPrimitive.Root>
        </AssistantRuntimeProvider>
      </div>

      <SettingsSidebar
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSettingsChange={setSettings}
        providers={providers}
      />

      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40" onClick={() => setShowSettings(false)} />
      )}
    </div>
  );
};

// Custom Chat Component (vanilla React implementation)
const CustomChatPage = () => {
  const [showSettings, setShowSettings] = useState(false);
  const { providers, settings, setSettings } = useAPISettings();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', content: '💬 Hello! This is a custom chat interface built with vanilla React components.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input?.trim() || isLoading) return;

    const userMessage = { id: Date.now(), role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setInput('');

    try {
      const content = await callAPI(input, settings);
      const formattedContent = settings.queryMode === 'all' 
        ? content.replace('Results from', '💬 Custom Chat Results from')
        : content;
      
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', content: formattedContent }]);
    } catch (error) {
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', content: `Connection Error: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <FrameworkHeader 
        title="Custom Chat UI (Vanilla React)"
        color="orange"
        settings={settings}
        onSettingsClick={() => setShowSettings(!showSettings)}
      />
      
      <div className="flex-1 overflow-hidden p-4">
        <div className="h-full bg-gradient-to-b from-orange-50 to-white rounded-lg border overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto p-4">
            {messages?.map((message) => (
              <div key={message.id} className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                <div className={`inline-block max-w-md px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-orange-600 text-white'
                    : 'bg-orange-100 text-gray-900 whitespace-pre-wrap'
                }`}>
                  {message.role === 'assistant' && (
                    <div className="text-xs font-semibold mb-1 text-orange-700">Custom Assistant</div>
                  )}
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="text-left mb-4">
                <div className="inline-block bg-orange-100 px-4 py-2 rounded-lg">
                  <div className="text-xs font-semibold mb-1 text-orange-700">Custom Assistant</div>
                  <div className="animate-pulse">💬 Processing your request...</div>
                </div>
              </div>
            )}
          </div>
          
          <div className="border-t bg-white p-4">
            <form onSubmit={handleSubmit}>
              <div className="flex space-x-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={`Ask questions... (${settings.queryMode === 'single' ? settings.selectedProvider : 'all providers'})`}
                  disabled={isLoading}
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-orange-500"
                />
                <button
                  type="submit"
                  disabled={isLoading || !input?.trim()}
                  className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg"
                >
                  Send
                </button>
              </div>
            </form>
            <div className="text-xs text-gray-500 mt-2">
              Custom React chat interface • No external UI framework
            </div>
          </div>
        </div>
      </div>

      <SettingsSidebar
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSettingsChange={setSettings}
        providers={providers}
      />

      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40" onClick={() => setShowSettings(false)} />
      )}
    </div>
  );
};

// ============================================================================
// MAIN APP
// ============================================================================

const App = () => {
  const [activeFramework, setActiveFramework] = useState('langchain');

  const frameworks = [
    { id: 'langchain', label: 'LangChain Agent UI', color: 'blue' },
    { id: 'llamaindex', label: 'LlamaIndex Chat UI', color: 'purple' },
    { id: 'assistant', label: 'Assistant UI', color: 'green' },
    { id: 'custom', label: 'Custom Chat UI', color: 'orange' }    
  ];

  return (
    <div className="h-screen">
      <div className="bg-gray-900 text-white p-4">
        <div className="flex space-x-4 flex-wrap">
          {frameworks.map(framework => (
            <button
              key={framework.id}
              onClick={() => setActiveFramework(framework.id)}
              className={`px-4 py-2 rounded transition-colors ${
                activeFramework === framework.id 
                  ? `bg-${framework.color}-600` 
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              {framework.label}
            </button>
          ))}
        </div>
      </div>

      {activeFramework === 'langchain' && <LangChainPage />}
      {activeFramework === 'llamaindex' && <LlamaIndexPage />}
      {activeFramework === 'assistant' && <AssistantUIPage />}
      {activeFramework === 'custom' && <CustomChatPage />}      
    </div>
  );
};

export default App;
