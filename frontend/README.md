# RAG Assistant VS Code Extension

This VS Code extension provides a chat participant that integrates with your RAG (Retrieval-Augmented Generation) backend to answer questions about your codebase.

## Features

- **Chat Participant**: Ask questions about your codebase directly in VS Code's chat interface
- **RAG Integration**: Connects to your local RAG backend running on `http://localhost:8000`
- **Command Palette**: Quick access via `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- **Rich Responses**: Formatted markdown responses with context and sources

## Prerequisites

1. **RAG Backend**: Your RAG backend must be running on `http://localhost:8000`
2. **Node.js**: Version 18 or higher
3. **VS Code**: Version 1.85.0 or higher

## Installation

1. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Compile the Extension**:
   ```bash
   npm run compile
   ```

3. **Package the Extension**:
   ```bash
   npm install -g vsce
   vsce package
   ```

4. **Install in VS Code**:
   - Open VS Code
   - Go to Extensions (`Ctrl+Shift+X`)
   - Click "..." and select "Install from VSIX..."
   - Choose the generated `.vsix` file

## Usage

### Chat Participant

1. Open VS Code's chat interface (`Ctrl+Shift+L` or `Cmd+Shift+L`)
2. Select "RAG Assistant" from the participant list
3. Ask questions about your codebase
4. Get AI-powered answers with context from your code

### Command Palette

- **Ask RAG Assistant**: `Ctrl+Shift+P` → "Ask RAG Assistant"
- **Search Code**: `Ctrl+Shift+P` → "Search Code"

### Example Questions

- "How do I implement authentication in Spring Boot?"
- "Show me examples of REST endpoints"
- "How do I handle database connections?"
- "What's the project structure?"

## Configuration

The extension is configured to connect to `http://localhost:8000`. To change this:

1. Edit `frontend/src/extension.ts`
2. Modify the `RAG_BACKEND_URL` constant
3. Recompile and reinstall the extension

## Development

### Project Structure

```
frontend/
├── src/
│   └── extension.ts          # Main extension logic
├── package.json              # Extension manifest
├── tsconfig.json            # TypeScript configuration
└── README.md               # This file
```

### Building

```bash
# Watch mode (auto-recompile on changes)
npm run watch

# One-time build
npm run compile

# Lint code
npm run lint
```

### Testing

```bash
# Run tests
npm test
```

## Troubleshooting

### "RAG backend is not running"

- Ensure your RAG backend is started: `uvicorn backend.server:app --reload --port 8000`
- Check if port 8000 is available
- Verify the backend responds to `http://localhost:8000/`

### Extension not loading

- Check VS Code's Developer Console (`Help` → `Toggle Developer Tools`)
- Ensure all dependencies are installed: `npm install`
- Recompile the extension: `npm run compile`

### Chat participant not appearing

- Restart VS Code after installation
- Check if the extension is enabled in Extensions view
- Verify the `package.json` has correct `chatParticipants` configuration

## API Integration

The extension makes HTTP POST requests to your RAG backend:

```typescript
POST http://localhost:8000/query
Content-Type: application/json

{
  "question": "Your question here",
  "max_results": 5
}
```

Response format:
```typescript
{
  "answer": "AI-generated answer",
  "context": ["context1", "context2"],
  "sources": ["file1.py", "file2.js"]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This extension is part of the RAG Assistant project. 