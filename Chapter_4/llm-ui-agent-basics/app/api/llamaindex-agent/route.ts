import { NextRequest } from 'next/server';
export async function POST(req: NextRequest) {
     try {
       const { messages, queryMode, selectedProvider, temperature, maxTokens } = await req.json();
       const lastMessage = messages[messages.length - 1];
       
       const endpoint = queryMode === 'single' ? '/query' : '/query-all';
       const payload = {
         topic: lastMessage.content,
         temperature: temperature || 0.7,
         max_tokens: maxTokens || 1000,
         template: '{topic}',
         ...(queryMode === 'single' && { provider: selectedProvider })
       };
       
       const response = await fetch(`http://localhost:8000${endpoint}`, {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify(payload)
       });
       
       const data = await response.json();
       const content = data.success ? data.response : `Error: ${data.error}`;
       return new Response(content);
     } catch (error) {
       return new Response(`Error: ${error.message}`, { status: 500 });
     }
}
