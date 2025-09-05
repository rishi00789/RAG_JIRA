# LangChain RAG Implementation for JIRA Assistant

## 🚀 Overview

This implementation refactors the JIRA RAG Assistant to use **LangChain** for all RAG functionalities while maintaining the **FastMCP** interface. This provides a more robust, standardized, and extensible RAG pipeline.

## 🏗️ Architecture

### Before (Original Implementation)
```
User Query → Custom Embedding → Milvus Search → Custom LLM → Response
```

### After (LangChain Implementation)
```
User Query → LangChain RAG Chain → Hybrid Retrieval → LangChain LLM → Response
```

## 📁 New File Structure

```
rag-assistant/backend/
├── langchain_jira_loader.py      # LangChain document loader for JIRA
├── langchain_milvus_store.py     # LangChain vector store for Milvus
├── langchain_retriever.py        # Hybrid retriever with JQL integration
├── langchain_llm.py              # LangChain LLM wrapper for Hugging Face
├── langchain_rag_chain.py        # Complete RAG chain implementation
├── langchain_mcp_server.py       # LangChain-powered MCP server
├── langchain_ingest.py           # LangChain-based ingestion script
├── test_langchain_rag.py         # Comprehensive test suite
└── start_langchain_mcp_server.sh # Startup script
```

## 🔧 Key Components

### 1. **LangChain Document Loader** (`langchain_jira_loader.py`)
- **`JiraDocumentLoader`**: Custom LangChain loader for JIRA data
- **`JiraRealtimeLoader`**: Real-time JIRA data synchronization
- Features:
  - Lazy loading for memory efficiency
  - Configurable content inclusion (comments, descriptions)
  - Rich metadata extraction
  - Error handling and retry logic

### 2. **LangChain Vector Store** (`langchain_milvus_store.py`)
- **`MilvusVectorStore`**: Custom LangChain vector store for Milvus
- Features:
  - Full LangChain vector store interface
  - Automatic collection management
  - Embedding generation with fallbacks
  - Similarity search with scores
  - Document management (add, delete, clear)

### 3. **Hybrid Retriever** (`langchain_retriever.py`)
- **`JiraHybridRetriever`**: Combines vector search, JQL queries, and metadata filtering
- **`JiraSprintRetriever`**: Specialized retriever for sprint-related queries
- Features:
  - Intelligent query analysis
  - JQL query generation from natural language
  - Metadata-based filtering
  - Document re-ranking
  - Sprint-specific context

### 4. **LangChain LLM Integration** (`langchain_llm.py`)
- **`HuggingFaceLLM`**: LangChain LLM wrapper for Hugging Face
- **`HuggingFaceChatModel`**: Chat model implementation
- **`JiraRAGLLM`**: Specialized LLM for JIRA RAG with optimized prompts
- Features:
  - OpenAI-compatible API integration
  - JIRA-specific prompt engineering
  - Enhanced context handling
  - Error handling and fallbacks

### 5. **Complete RAG Chain** (`langchain_rag_chain.py`)
- **`JiraRAGChain`**: End-to-end RAG pipeline
- Features:
  - Intelligent retrieval strategy selection
  - Action detection and execution
  - Sprint context integration
  - JQL query information
  - Comprehensive result formatting

### 6. **LangChain MCP Server** (`langchain_mcp_server.py`)
- **`LangChainJIRARAGMCPServer`**: MCP server using LangChain components
- Features:
  - FastMCP integration maintained
  - Real-time data synchronization
  - LangChain-powered ingestion
  - Enhanced error handling
  - Collection statistics

## 🚀 Getting Started

### 1. Install Dependencies
```bash
cd rag-assistant
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
# Required
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your-email@domain.com
JIRA_API_TOKEN=your-api-token

# Optional
HF_TOKEN=your-huggingface-token
JIRA_PROJECT_KEY=ALL
SYNC_CACHE_DURATION=5
```

### 3. Start Milvus
```bash
docker run -p 19530:19530 -d milvusdb/milvus:latest
```

### 4. Ingest JIRA Data (LangChain)
```bash
cd backend
python3 langchain_ingest.py --project-key ALL --max-issues 1000 --clear-existing
```

### 5. Start LangChain MCP Server
```bash
./start_langchain_mcp_server.sh
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
cd backend
python3 test_langchain_rag.py
```

### Test Individual Components
```bash
# Test document loader
python3 langchain_jira_loader.py

# Test vector store
python3 langchain_milvus_store.py

# Test retriever
python3 langchain_retriever.py

# Test LLM
python3 langchain_llm.py

# Test RAG chain
python3 langchain_rag_chain.py
```

## 🔄 Migration from Original Implementation

### Key Differences

| Feature | Original | LangChain |
|---------|----------|-----------|
| Document Loading | Custom JIRA API calls | LangChain `BaseLoader` |
| Vector Storage | Direct Milvus operations | LangChain `VectorStore` |
| Retrieval | Simple similarity search | Hybrid retrieval with JQL |
| LLM Integration | Direct OpenAI client | LangChain `BaseChatModel` |
| RAG Pipeline | Custom implementation | LangChain `RAGChain` |
| Error Handling | Basic try/catch | LangChain callbacks |
| Extensibility | Limited | High (LangChain ecosystem) |

### Benefits of LangChain Implementation

1. **Standardization**: Uses LangChain's standardized interfaces
2. **Extensibility**: Easy to add new components and features
3. **Robustness**: Better error handling and retry logic
4. **Ecosystem**: Access to LangChain's rich ecosystem
5. **Maintainability**: Cleaner, more modular code
6. **Testing**: Better testability with LangChain's testing utilities
7. **Documentation**: Well-documented LangChain patterns

## 🛠️ Usage Examples

### Basic RAG Query
```python
from langchain_rag_chain import create_jira_rag_chain

# Create RAG chain
rag_chain = create_jira_rag_chain(collection_name="jira_data")

# Process query
result = rag_chain.process_query("What are the high priority stories in progress?")
print(result["answer"])
```

### Custom Document Loading
```python
from langchain_jira_loader import JiraDocumentLoader

# Create loader
loader = JiraDocumentLoader(
    project_key="MYPROJECT",
    max_results=500,
    include_comments=True
)

# Load documents
documents = loader.load()
```

### Vector Store Operations
```python
from langchain_milvus_store import MilvusVectorStore

# Create vector store
vector_store = MilvusVectorStore(collection_name="jira_data")

# Add documents
vector_store.add_documents(documents)

# Search
results = vector_store.similarity_search("authentication", k=5)
```

### Hybrid Retrieval
```python
from langchain_retriever import JiraHybridRetriever

# Create retriever
retriever = JiraHybridRetriever(
    vector_store=vector_store,
    jira_ops=jira_ops,
    k=10,
    jql_k=20
)

# Retrieve documents
docs = retriever._get_relevant_documents("Show me all bugs assigned to John")
```

## 🔧 Configuration Options

### Document Loader Configuration
```python
loader = JiraDocumentLoader(
    project_key="ALL",           # JIRA project key
    max_results=1000,           # Maximum issues to fetch
    days_back=365,              # Days back to fetch
    include_comments=True,      # Include issue comments
    include_descriptions=True   # Include issue descriptions
)
```

### Vector Store Configuration
```python
vector_store = MilvusVectorStore(
    collection_name="jira_data",  # Milvus collection name
    embedding_function=embedding, # Custom embedding function
    connection_args={             # Milvus connection args
        "host": "localhost",
        "port": "19530"
    }
)
```

### RAG Chain Configuration
```python
rag_chain = JiraRAGChain(
    vector_store=vector_store,
    llm=llm,
    jira_ops=jira_ops,
    use_hybrid_retrieval=True,    # Enable hybrid retrieval
    use_sprint_retrieval=True,    # Enable sprint retrieval
    max_retrieval_results=50      # Maximum retrieval results
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install langchain langchain-community langchain-openai
   ```

2. **Milvus Connection Issues**
   ```bash
   # Check if Milvus is running
   curl http://localhost:19530/health
   
   # Start Milvus
   docker run -p 19530:19530 -d milvusdb/milvus:latest
   ```

3. **JIRA API Issues**
   - Verify environment variables are set correctly
   - Check JIRA API token permissions
   - Ensure JIRA base URL is correct

4. **Hugging Face API Issues**
   - Set `HF_TOKEN` environment variable
   - Check API token validity
   - Verify model availability

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Considerations

### Optimization Tips

1. **Batch Processing**: Use batch operations for large datasets
2. **Caching**: Enable caching for frequently accessed data
3. **Indexing**: Optimize Milvus indexes for your use case
4. **Memory Management**: Use lazy loading for large datasets
5. **Connection Pooling**: Reuse connections where possible

### Monitoring

- Collection statistics via `get_collection_stats()`
- Query performance metrics
- Memory usage monitoring
- API rate limiting

## 🔮 Future Enhancements

1. **Advanced Retrieval**: Implement more sophisticated retrieval strategies
2. **Multi-modal Support**: Add support for images and attachments
3. **Streaming**: Implement streaming responses
4. **Caching**: Add Redis-based caching
5. **Analytics**: Add query analytics and insights
6. **Custom Embeddings**: Support for custom embedding models
7. **Multi-language**: Support for multiple languages

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Milvus Documentation](https://milvus.io/docs)
- [JIRA REST API](https://developer.atlassian.com/cloud/jira/platform/rest/v3/)

---

**🎯 The LangChain implementation provides a more robust, extensible, and maintainable RAG system while preserving the MCP interface for seamless integration!**
