# JIRA Integration with RAG Assistant

This guide explains how to set up and use the JIRA integration with your RAG Assistant to ingest and query JIRA data.

## Overview

The JIRA integration allows you to:
- Fetch issues, comments, and metadata from JIRA via the REST API
- Store this data in Milvus vector database with embeddings
- Query the data using natural language through the RAG system
- Get relevant JIRA issues and context for your questions

## Prerequisites

1. **JIRA Cloud Account**: You need access to a JIRA Cloud instance
2. **API Token**: Generate an API token from your Atlassian account
3. **Milvus Database**: Running locally or in Docker
4. **Python Environment**: With required dependencies installed

## Setup

### 1. Environment Configuration

Copy the example environment file and configure your JIRA credentials:

```bash
cd rag-assistant/backend
cp env.example .env
```

Edit `.env` with your actual values:

```bash
# Your JIRA instance URL
JIRA_BASE_URL=https://your-domain.atlassian.net

# Your JIRA username (usually your email)
JIRA_USERNAME=your-email@domain.com

# Your JIRA API token
JIRA_API_TOKEN=your-api-token-here

# Optional: Specific project key (set to "ALL" for all projects)
JIRA_PROJECT_KEY=ALL

# Hugging Face token for inference
HF_TOKEN=your-hf-token-here
```

### 2. Get JIRA API Token

1. Go to [Atlassian Account Settings](https://id.atlassian.com/manage-profile/security/api-tokens)
2. Click "Create API token"
3. Give it a label (e.g., "RAG Assistant")
4. Copy the generated token to your `.env` file

### 3. Start Milvus

Ensure Milvus is running:

```bash
cd rag-assistant
docker-compose up -d
```

## Usage

### 1. Ingest JIRA Data

Run the JIRA ingestion script to fetch and store data:

```bash
cd rag-assistant/backend

# Basic ingestion (last 365 days, max 1000 issues)
python jira_ingest.py

# Custom parameters
python jira_ingest.py --max-issues 500 --days-back 180

# Clear existing data and re-ingest
python jira_ingest.py --clear-existing
```

**Command Line Options:**
- `--max-issues`: Maximum number of issues to fetch (default: 1000)
- `--days-back`: Number of days back to fetch issues from (default: 365)
- `--clear-existing`: Clear existing data before ingestion

### 2. Start the RAG Server

```bash
cd rag-assistant/backend
python server.py
```

The server will automatically connect to Milvus and load the JIRA collection.

### 3. Query JIRA Data

Use the `/query` endpoint to ask questions about your JIRA data:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the recent bug reports?",
    "max_results": 5
  }'
```

**Example Questions:**
- "What are the open issues assigned to John?"
- "Show me bug reports from the last month"
- "What are the main problems in project ABC?"
- "Who is working on high-priority tasks?"

## Data Structure

The system ingests the following JIRA data:

- **Issue Summary**: Brief description of the issue
- **Issue Description**: Detailed description and context
- **Comments**: All comments and discussions
- **Metadata**: Issue type, status, priority, assignee, reporter, dates

Each piece of content is:
1. Split into manageable chunks (500 characters with 100 character overlap)
2. Embedded using vector representations
3. Stored in Milvus with full metadata
4. Searchable via semantic similarity

## API Endpoints

### Query Endpoint
- **POST** `/query` - Main RAG query endpoint
- **Body**: `{"question": "your question", "max_results": 3}`

### Status Endpoints
- **GET** `/` - Health check and system status
- **GET** `/status` - Detailed system status
- **POST** `/reconnect` - Reconnect to Milvus

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your JIRA credentials in `.env`
   - Ensure your API token is valid and not expired
   - Check that your username is correct

2. **Connection Issues**
   - Verify Milvus is running: `docker-compose ps`
   - Check Milvus logs: `docker-compose logs milvus`
   - Ensure ports 19530 and 8000 are available

3. **No Data Found**
   - Check JIRA project permissions
   - Verify the date range in your ingestion
   - Check JIRA API rate limits

4. **Collection Not Found**
   - Run `python jira_ingest.py` first
   - Check Milvus connection status
   - Verify collection name is "jira_data"

### Debug Mode

Enable debug logging by setting environment variable:

```bash
export PYTHONPATH=.
python -u jira_ingest.py --max-issues 10
```

## Advanced Configuration

### Custom JQL Queries

Modify the `fetch_jira_issues` method in `jira_ingest.py` to use custom JQL:

```python
# Example: Only fetch specific issue types
jql_query = "issuetype in (Bug, Task) AND status != Closed"

# Example: Fetch issues from specific components
jql_query = "component = 'Frontend' AND priority in (High, Critical)"
```

### Rate Limiting

JIRA has API rate limits. The script includes basic rate limiting, but you can adjust:

```python
import time

# Add delay between requests if needed
time.sleep(0.1)  # 100ms delay
```

### Custom Fields

To include custom JIRA fields, modify the `fetch_jira_issues` method:

```python
params = {
    'jql': jql_query,
    'maxResults': max_results,
    'fields': 'summary,description,comment,issuetype,project,status,priority,assignee,reporter,created,updated,labels,components,customfield_10000'
}
```

## Performance Tips

1. **Batch Size**: Adjust batch size in `insert_data` method for optimal performance
2. **Date Range**: Start with smaller date ranges for testing
3. **Project Scope**: Limit to specific projects initially
4. **Embedding Model**: Consider using production-grade embedding models for better search quality

## Security Considerations

1. **API Tokens**: Never commit API tokens to version control
2. **Access Control**: Ensure API tokens have minimal required permissions
3. **Data Privacy**: Be aware of what JIRA data you're ingesting
4. **Network Security**: Use HTTPS for all JIRA API calls

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review JIRA API documentation: https://developer.atlassian.com/cloud/jira/platform/rest/v2
3. Check Milvus logs and server logs
4. Verify environment configuration

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up environment
cd rag-assistant/backend
cp env.example .env
# Edit .env with your JIRA credentials

# 2. Start infrastructure
cd ..
docker-compose up -d

# 3. Ingest JIRA data
cd backend
python jira_ingest.py --max-issues 100 --days-back 30

# 4. Start RAG server
python server.py

# 5. Query your data
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the current open bugs?", "max_results": 5}'
```

This integration transforms your JIRA data into a searchable knowledge base that can answer questions about your projects, issues, and team activities!
