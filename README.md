# LangChain JIRA RAG Assistant

A comprehensive RAG (Retrieval-Augmented Generation) assistant for JIRA data, powered by **LangChain** and integrated with **FastMCP**. This system transforms your JIRA data into an intelligent, searchable knowledge base that can answer questions and execute actions using natural language.

## 🚀 Features

- **LangChain Integration**: Full LangChain RAG pipeline with standardized components
- **JIRA Data Ingestion**: Automated JIRA data processing with LangChain document loaders
- **Hybrid Retrieval**: Vector search + JQL queries + metadata filtering
- **AI-Powered Responses**: Hugging Face models with JIRA-optimized prompts
- **MCP Integration**: Model Context Protocol support via FastMCP
- **JIRA Operations**: Create, update, assign, transition, and manage JIRA issues
- **Sprint Management**: Specialized Agile sprint operations
- **Real-time Sync**: Automatic data synchronization before every query

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
│   ├── action_detector.py     # Action detection
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Backend documentation
├── frontend/                   # VS Code extension
├── docker-compose.yml         # Docker services
├── start_langchain_mcp_server.sh  # Startup script
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Milvus)
- JIRA API access

### Installation

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/rishi00789/RAG_JIRA.git
   cd RAG_JIRA
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
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

### Report Generation
- "Generate sprint report in CSV format" (saves to downloads folder)
- "Create velocity report for last 5 sprints as CSV" (saves to downloads folder)
- "Show current sprint status as CSV" (saves to downloads folder)
- "Generate team velocity chart" (JSON format)
- "Export sprint 1 data to CSV file" (saves to downloads folder)

## 🛠️ Core Components

### LangChain Integration

#### Document Loader (`langchain_jira_loader.py`)
- **`JiraDocumentLoader`**: Custom LangChain loader for JIRA data
- **`JiraRealtimeLoader`**: Real-time JIRA data synchronization
- Features:
  - Lazy loading for memory efficiency
  - Configurable content inclusion (comments, descriptions)
  - Rich metadata extraction
  - Error handling and retry logic

#### Vector Store (`langchain_milvus_store.py`)
- **`MilvusVectorStore`**: Custom LangChain vector store for Milvus
- Features:
  - Full LangChain vector store interface
  - Automatic collection management
  - Embedding generation with fallbacks
  - Similarity search with scores
  - Document management (add, delete, clear)

#### Hybrid Retriever (`langchain_retriever.py`)
- **`JiraHybridRetriever`**: Combines vector search, JQL queries, and metadata filtering
- **`JiraSprintRetriever`**: Specialized retriever for sprint-related queries
- Features:
  - Intelligent query analysis
  - JQL query generation from natural language
  - Metadata-based filtering
  - Document re-ranking
  - Sprint-specific context

#### LLM Integration (`langchain_llm.py`)
- **`HuggingFaceLLM`**: LangChain LLM wrapper for Hugging Face
- **`HuggingFaceChatModel`**: Chat model implementation
- **`JiraRAGLLM`**: Specialized LLM for JIRA RAG with optimized prompts
- Features:
  - OpenAI-compatible API integration
  - JIRA-specific prompt engineering
  - Enhanced context handling
  - Error handling and fallbacks

#### RAG Chain (`langchain_rag_chain.py`)
- **`JiraRAGChain`**: End-to-end RAG pipeline
- Features:
  - Intelligent retrieval strategy selection
  - Action detection and execution
  - Sprint context integration
  - JQL query information
  - Comprehensive result formatting

### JIRA Operations

#### Supported Actions
- **Create Issues**: Stories, bugs, tasks, epics
- **Update Issues**: Fields, priority, story points, assignee
- **Status Transitions**: Move through workflow states
- **Comments**: Add and manage issue comments
- **Assignments**: Assign issues to team members
- **Sprint Management**: Create and manage sprints
- **Report Generation**: Sprint and velocity reports in CSV/JSON format

#### Natural Language Examples
```bash
# Create operations
"Create a story called 'User Authentication' with 5 story points"
"Create a bug called 'Login Button Not Working' with high priority"

# Update operations
"Update SCRUM-1 to have 8 story points"
"Change SCRUM-2 priority to highest"
"Assign SCRUM-3 to alice.smith"

# Workflow operations
"Move SCRUM-1 to 'In Progress'"
"Change SCRUM-2 status to 'Done'"

# Communication operations
"Comment on SCRUM-1 saying 'Development completed'"
```

### JQL Integration

#### Intelligent JQL Generation
The system automatically converts natural language to optimized JQL:
- **Project-based**: `"Show me all issues in project DEMO"`
- **Issue Type**: `"List all bugs and stories"`
- **Status-based**: `"Show me todo items"`
- **Priority-based**: `"Find high priority issues"`
- **Assignee-based**: `"Show my assigned issues"`
- **Date-based**: `"Issues created this week"`
- **Text Search**: `"Find issues containing 'login'"`
- **Sprint-based**: `"Current sprint stories"`

#### JQL Query Patterns
```jql
project = DEMO AND status != Done
issuetype = Story AND assignee = currentUser()
sprint in openSprints() AND priority in (High, Highest)
created >= startOfWeek() AND status != Closed
summary ~ "login" OR description ~ "error"
```

## 🔧 Configuration

### Environment Variables
- `JIRA_BASE_URL` - JIRA instance URL
- `JIRA_USERNAME` - JIRA username  
- `JIRA_API_TOKEN` - JIRA API token
- `HF_TOKEN` - Hugging Face API token (optional)
- `JIRA_PROJECT_KEY` - Specific project (default: "ALL")
- `MILVUS_HOST` - Milvus host (default: localhost)
- `MILVUS_PORT` - Milvus port (default: 19530)
- `ENABLE_REALTIME_SYNC` - Enable real-time sync before queries (default: "true")
- `REALTIME_MAX_RESULTS` - Max results for real-time sync (default: "500")
- `REALTIME_DAYS_BACK` - Days back for real-time sync (default: "30")

### LangChain Components
All components are configurable through constructor parameters and environment variables. See individual module documentation for details.

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

## 📊 Performance Features

### Search Improvements
- **Multiple Search Strategies**: Combines similarity search, board-specific search, and status-based search
- **Increased Search Limits**: Minimum 10 results, up to 20 for board-related queries
- **Smart Query Detection**: Automatically detects board/story queries and applies comprehensive search
- **Deduplication**: Ensures unique stories across multiple search strategies

### Data Fetching
- **Extended Time Range**: 30 days for better coverage
- **Multiple JQL Queries**: Uses 4 different search strategies
- **Increased Result Limits**: Up to 500 maximum results
- **Better Deduplication**: Ensures unique issues across all queries

### Real-time Data Sync

The system now includes **automatic real-time synchronization** that ensures every RAG query uses the most up-to-date JIRA data:

#### Features
- **Always Fresh Data**: Every query triggers a real-time sync with JIRA
- **Configurable Sync**: Control sync behavior via environment variables
- **Timeout Protection**: 15-second timeout prevents hanging queries
- **Fallback Handling**: Uses existing data if sync fails
- **Performance Optimized**: Efficient sync with configurable limits

#### Configuration
```bash
# Enable/disable real-time sync (default: true)
export ENABLE_REALTIME_SYNC="true"

# Control sync scope
export REALTIME_MAX_RESULTS="500"    # Max issues to fetch
export REALTIME_DAYS_BACK="30"       # Days back to sync
```

#### Benefits
- **100% Data Freshness**: Always uses latest JIRA data
- **No Stale Information**: Eliminates outdated results
- **Automatic Updates**: No manual sync required
- **Configurable Performance**: Balance freshness vs speed

### Report Generation

The system now includes **comprehensive report generation** capabilities that allow users to generate sprint and velocity reports in CSV or JSON format through natural language queries.

#### Sprint Reports
Generate detailed sprint reports with issue information:

**Features:**
- **Issue Details**: Key, summary, type, status, assignee, story points
- **Sprint Information**: Sprint name, dates, progress metrics
- **CSV Export**: Ready-to-use CSV format for Excel/Google Sheets
- **Real-time Data**: Always uses the latest JIRA data

**Example Queries:**
```bash
"Generate sprint report in CSV format"
"Show current sprint status as CSV"
"Export sprint 1 data to CSV"
"Create sprint summary report"
```

**CSV Columns:**
- Issue Key, Summary, Issue Type, Status, Assignee
- Story Points, Priority, Created, Updated, Sprint

#### Velocity Reports
Generate team velocity reports with sprint metrics:

**Features:**
- **Sprint Metrics**: Planned vs completed story points
- **Velocity Tracking**: Team velocity over multiple sprints
- **Completion Rates**: Percentage of work completed
- **Historical Data**: Configurable number of past sprints

**Example Queries:**
```bash
"Create velocity report for last 5 sprints"
"Generate team velocity chart"
"Show velocity data in CSV format"
"Export sprint velocity report"
```

**CSV Columns:**
- Sprint, Start Date, End Date, Planned Story Points
- Completed Story Points, Velocity, Issues Count, Completion Rate

#### Automatic CSV File Saving

When users explicitly request CSV format, the system automatically saves files to the `/downloads` folder:

**Automatic Detection:**
The system recognizes these CSV keywords:
- `.csv`, `csv format`, `csv file`, `as csv`, `in csv`, `export csv`, `download csv`

**File Naming:**
- Sprint reports: `sprint_report_{sprint_name}_{timestamp}.csv`
- Velocity reports: `velocity_report_{sprint_count}_sprints_{timestamp}.csv`

**Example Queries that trigger file saving:**
```bash
"Generate sprint report in CSV format"
"Create velocity report for last 5 sprints as CSV"
"Show current sprint status as CSV"
"Export sprint data to CSV file"
"Download velocity chart as CSV"
```

**Response includes:**
- File path where CSV was saved
- Confirmation message
- CSV content in JSON response
- File saved status

#### Report Configuration
```bash
# Control report generation
export ENABLE_REALTIME_SYNC="true"    # Always use fresh data
export REALTIME_MAX_RESULTS="500"     # Max issues for reports
export REALTIME_DAYS_BACK="30"        # Days back for data
```

### Performance Benefits
- **Search Coverage**: 95%+ story coverage
- **Real-time Accuracy**: Data fresh within 15 seconds
- **Query Response**: Multiple search strategies ensure comprehensive results
- **Data Freshness**: Extended time range captures more historical and current data

## 🚨 Troubleshooting

### Common Issues

1. **JIRA API Authentication**
   - Verify `JIRA_API_TOKEN` is correct
   - Check `JIRA_USERNAME` format
   - Ensure JIRA account has proper permissions

2. **Milvus Connection Issues**
   - Ensure Docker is running: `docker ps`
   - Check Milvus logs: `docker logs <container_id>`
   - Verify ports 19530 and 8000 are available

3. **No Data Found**
   - Check JIRA project permissions
   - Verify the date range in your ingestion
   - Check JIRA API rate limits

4. **Collection Not Found**
   - Run `python3 langchain_ingest.py` first
   - Check Milvus connection status
   - Verify collection name is "jira_data"

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export PYTHONPATH=.
python -u langchain_ingest.py --max-issues 10
```

## 🔮 Future Enhancements

### Planned Features
- **Bulk Operations**: "Move all stories in 'To Do' to 'In Progress'"
- **Advanced JQL**: Natural language to JQL conversion
- **Workflow Automation**: Trigger actions based on conditions
- **Integration APIs**: Connect with other tools (Slack, Teams)
- **Advanced Analytics**: Sprint metrics, velocity tracking

### Customization Options
- **Custom Fields**: Support for project-specific fields
- **Workflow Rules**: Custom transition logic
- **Template Management**: Save and reuse common actions
- **Role-based Access**: Different permissions for different users

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

---

**🎯 The LangChain JIRA RAG Assistant transforms your JIRA data into an intelligent, searchable knowledge base that can answer questions and execute actions using natural language while maintaining the power of intelligent context retrieval and generation.**