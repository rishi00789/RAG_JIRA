# LangChain RAG Backend

This directory contains the LangChain-powered RAG implementation for the JIRA Assistant.

## 📁 File Structure

### Core LangChain Components
- **`langchain_jira_loader.py`** - LangChain document loader for JIRA data
- **`langchain_milvus_store.py`** - LangChain vector store for Milvus
- **`langchain_retriever.py`** - Hybrid retriever with JQL integration
- **`langchain_llm.py`** - LangChain LLM wrapper for Hugging Face
- **`langchain_rag_chain.py`** - Complete RAG chain implementation
- **`langchain_mcp_server.py`** - LangChain-powered MCP server

### Supporting Components
- **`jira_operations.py`** - JIRA API operations
- **`action_detector.py`** - Action detection from user queries
- **`langchain_ingest.py`** - LangChain-based data ingestion

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export JIRA_BASE_URL="https://your-domain.atlassian.net"
export JIRA_USERNAME="your-email@domain.com"
export JIRA_API_TOKEN="your-api-token"
export HF_TOKEN="your-huggingface-token"  # Optional
```

### 3. Ingest JIRA Data
```bash
python3 langchain_ingest.py --project-key ALL --clear-existing
```

### 4. Start MCP Server
```bash
python3 langchain_mcp_server.py
```

## 🔧 Usage Examples

### Document Loading
```python
from langchain_jira_loader import JiraDocumentLoader

loader = JiraDocumentLoader(project_key="MYPROJECT")
documents = loader.load()
```

### Vector Store Operations
```python
from langchain_milvus_store import MilvusVectorStore

vector_store = MilvusVectorStore(collection_name="jira_data")
vector_store.add_documents(documents)
results = vector_store.similarity_search("authentication", k=5)
```

### RAG Chain
```python
from langchain_rag_chain import create_jira_rag_chain

rag_chain = create_jira_rag_chain()
result = rag_chain.process_query("What are the high priority stories?")
```

## 📊 Architecture

```
User Query → LangChain RAG Chain → Hybrid Retrieval → LangChain LLM → Response
                ↓
        [Vector Search + JQL + Metadata Filtering]
                ↓
        [JIRA Data + Sprint Context + JQL Info]
                ↓
        [Enhanced Prompt + Context → AI Response]
```

## 🛠️ Configuration

All components are configurable through:
- Environment variables
- Constructor parameters
- Configuration files

See individual module documentation for detailed configuration options.
