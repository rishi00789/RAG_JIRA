# LangChain JIRA RAG Assistant

A comprehensive RAG (Retrieval-Augmented Generation) assistant for JIRA data, powered by **LangChain** and integrated with **FastMCP**.

## 🚀 Features

- **LangChain Integration**: Full LangChain RAG pipeline with standardized components
- **JIRA Data Ingestion**: Automated JIRA data processing with LangChain document loaders
- **Hybrid Retrieval**: Vector search + JQL queries + metadata filtering
- **AI-Powered Responses**: Hugging Face models with JIRA-optimized prompts
- **MCP Integration**: Model Context Protocol support via FastMCP
- **Real-time Sync**: Automatic data synchronization
- **Sprint Management**: Specialized Agile sprint operations

## 🏗️ Architecture

```
User Query → LangChain RAG Chain → Hybrid Retrieval → LangChain LLM → Response
                ↓
        [Vector Search + JQL + Metadata Filtering]
                ↓
        [JIRA Data + Sprint Context + JQL Info]
                ↓
        [Enhanced Prompt + Context → AI Response]
```

## 📁 Project Structure

```
rag-assistant/
├── backend/                    # LangChain RAG implementation
│   ├── langchain_*.py         # Core LangChain components
│   ├── jira_operations.py     # JIRA API operations
│   └── README.md              # Backend documentation
├── frontend/                   # VS Code extension
├── requirements.txt            # Python dependencies
├── start_langchain_mcp_server.sh  # Startup script
└── LANGCHAIN_RAG_IMPLEMENTATION.md # Detailed documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Milvus)
- JIRA API access

### Installation

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd rag-assistant
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   ```bash
   export JIRA_BASE_URL="https://your-domain.atlassian.net"
   export JIRA_USERNAME="your-email@domain.com"
   export JIRA_API_TOKEN="your-api-token"
   export HF_TOKEN="your-huggingface-token"  # Optional
   ```

3. **Start Milvus**:
   ```bash
   docker run -p 19530:19530 -d milvusdb/milvus:latest
   ```

4. **Ingest JIRA Data**:
   ```bash
   cd backend
   python3 langchain_ingest.py --project-key ALL --clear-existing
   ```

5. **Start LangChain MCP Server**:
   ```bash
   ./start_langchain_mcp_server.sh
   ```

## 🔧 Usage

### MCP Server
The server runs on `http://127.0.0.1:8003/mcp` and provides:
- `rag_query` - Comprehensive RAG queries
- `ingest_jira_data` - Data ingestion
- `get_collection_stats` - Collection statistics

### Example Queries
- "What are the high priority stories in progress?"
- "Show me all bugs assigned to John"
- "What is the current sprint status?"
- "Create a new story for user authentication"

## 🧪 Testing

```bash
cd backend
python3 test_langchain_rag.py
```

## 📚 Documentation

- **[LangChain Implementation Guide](LANGCHAIN_RAG_IMPLEMENTATION.md)** - Detailed technical documentation
- **[Backend README](backend/README.md)** - Backend-specific documentation
- **[MCP Implementation](MCP_IMPLEMENTATION.md)** - MCP integration details

## 🔧 Configuration

### Environment Variables
- `JIRA_BASE_URL` - JIRA instance URL
- `JIRA_USERNAME` - JIRA username  
- `JIRA_API_TOKEN` - JIRA API token
- `HF_TOKEN` - Hugging Face API token (optional)
- `JIRA_PROJECT_KEY` - Specific project (default: "ALL")
- `SYNC_CACHE_DURATION` - Sync cache duration in seconds

### LangChain Components
All components are configurable through constructor parameters and environment variables. See individual module documentation for details.

## 🆚 Migration from Original

This implementation replaces the original custom RAG system with LangChain components while maintaining the MCP interface:

| Component | Before | After |
|-----------|--------|-------|
| Document Loading | Custom JIRA API | LangChain `BaseLoader` |
| Vector Storage | Direct Milvus | LangChain `VectorStore` |
| Retrieval | Simple similarity | Hybrid retrieval |
| LLM Integration | Direct OpenAI | LangChain `BaseChatModel` |
| RAG Pipeline | Custom implementation | LangChain `RAGChain` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License