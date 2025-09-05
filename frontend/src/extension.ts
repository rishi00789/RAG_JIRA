import * as vscode from 'vscode';
import axios from 'axios';

// Configuration
const RAG_BACKEND_URL = 'http://localhost:8000';

interface RAGResponse {
  answer: string;
  context: string[];
  sources: string[];
}

export function activate(context: vscode.ExtensionContext) {
  console.log('RAG Assistant extension is now active!');

  // Register chat participant
  const chatParticipant = vscode.chat.createChatParticipant('rag-assistant.chat-participant', async (request: vscode.ChatRequest, context: vscode.ChatContext, stream: vscode.ChatResponseStream, token: vscode.CancellationToken) => {
      try {
        const userMessage = request.prompt;
        
        // Show typing indicator
        stream.progress('🤔 Thinking...');
        
        // Call RAG backend
        const response = await callRAGBackend(userMessage);
        
        // Format the response
        const formattedResponse = formatRAGResponse(response);
        
        // Stream the response
        stream.markdown(formattedResponse);
        
      } catch (error) {
        console.error('Error in chat participant:', error);
        stream.markdown(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`);
      }
    });

  // Register commands
  const askCommand = vscode.commands.registerCommand('rag-assistant.ask', async () => {
    const question = await vscode.window.showInputBox({
      prompt: 'Ask RAG Assistant about your codebase:',
      placeHolder: 'e.g., How do I implement authentication in Spring Boot?'
    });

    if (question) {
      try {
        const response = await callRAGBackend(question);
        const formattedResponse = formatRAGResponse(response);
        
        // Show response in a new document
        const document = await vscode.workspace.openTextDocument({
          content: formattedResponse,
          language: 'markdown'
        });
        
        await vscode.window.showTextDocument(document);
      } catch (error) {
        vscode.window.showErrorMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
  });

  const searchCommand = vscode.commands.registerCommand('rag-assistant.search', async () => {
    const query = await vscode.window.showInputBox({
      prompt: 'Search for code patterns or functionality:',
      placeHolder: 'e.g., JWT authentication, REST endpoints, database queries'
    });

    if (query) {
      try {
        const response = await callRAGBackend(query);
        const formattedResponse = formatRAGResponse(response);
        
        // Show response in a new document
        const document = await vscode.workspace.openTextDocument({
          content: formattedResponse,
          language: 'markdown'
        });
        
        await vscode.window.showTextDocument(document);
      } catch (error) {
        vscode.window.showErrorMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
  });

  // Add commands to context
  context.subscriptions.push(chatParticipant, askCommand, searchCommand);
}

async function callRAGBackend(question: string): Promise<RAGResponse> {
  try {
    const response = await axios.post(`${RAG_BACKEND_URL}/query`, {
      question: question,
      max_results: 5
    }, {
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'application/json'
      }
    });

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error('RAG backend is not running. Please start the backend server first.');
      } else if (error.response) {
        throw new Error(`Backend error: ${error.response.status} - ${error.response.data?.detail || error.message}`);
      } else {
        throw new Error(`Network error: ${error.message}`);
      }
    }
    throw error;
  }
}

function formatRAGResponse(response: RAGResponse): string {
  let formatted = `# RAG Assistant Response\n\n`;
  
  // Add the main answer
  formatted += `## Answer\n\n${response.answer}\n\n`;
  
  // Add context if available
  if (response.context && response.context.length > 0) {
    formatted += `## Context\n\n`;
    response.context.forEach((ctx, index) => {
      formatted += `**Context ${index + 1}:**\n\`\`\`\n${ctx}\n\`\`\`\n\n`;
    });
  }
  
  // Add sources if available
  if (response.sources && response.sources.length > 0) {
    formatted += `## Sources\n\n`;
    response.sources.forEach((source, index) => {
      formatted += `${index + 1}. \`${source}\`\n`;
    });
  }
  
  return formatted;
}

export function deactivate() {
  console.log('RAG Assistant extension is now deactivated!');
} 