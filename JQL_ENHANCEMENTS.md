# JQL-Enhanced RAG System

## Overview

The RAG system has been significantly enhanced with **JQL (Jira Query Language)** capabilities, providing intelligent, precise, and optimized search functionality. This enhancement leverages the [JIRA JQL API](https://developer.atlassian.com/cloud/jira/platform/rest/v2/api-group-jql/#api-group-jql) to deliver more accurate and targeted results.

## 🚀 Key Features

### 1. **Intelligent JQL Query Generation**
- Automatically converts natural language queries to optimized JQL
- Supports complex search criteria including:
  - **Project-based**: `"Show me all issues in project DEMO"`
  - **Issue Type**: `"List all bugs and stories"`
  - **Status-based**: `"Show me todo items"`
  - **Priority-based**: `"Find high priority issues"`
  - **Assignee-based**: `"Show my assigned issues"`
  - **Date-based**: `"Issues created this week"`
  - **Text Search**: `"Find issues containing 'login'"`
  - **Sprint-based**: `"Current sprint stories"`
  - **Component/Label**: `"Issues with 'urgent' label"`

### 2. **Direct JQL Execution**
- Execute custom JQL queries directly via API
- Real-time validation and parsing
- Configurable result limits (up to 100 results)

### 3. **JQL Field Discovery**
- Dynamic field reference data
- Available operators and functions
- Reserved word identification

### 4. **Query Validation & Parsing**
- Syntax validation before execution
- Detailed error reporting
- Query structure analysis

## 🔧 API Endpoints

### **Main Query Endpoint (Enhanced)**
```http
POST /query
```
Now includes automatic JQL generation and execution for relevant queries.

### **Direct JQL Search**
```http
POST /jira/search/jql
```
Execute custom JQL queries directly.

**Request Body:**
```json
{
  "jql_query": "project = DEMO AND status != Done",
  "max_results": 50
}
```

### **JQL Suggestions**
```http
GET /jira/search/suggestions
```
Get available fields, functions, and operators for building JQL queries.

### **JQL Parsing**
```http
POST /jira/search/parse
```
Validate and parse JQL queries for syntax errors.

**Request Body:**
```json
{
  "jql_query": "project = DEMO AND status != Done"
}
```

## 🧠 How It Works

### **1. Natural Language Processing**
The system analyzes user questions and identifies search intent:
- **Keywords Detection**: Recognizes search criteria (project, status, priority, etc.)
- **Context Understanding**: Identifies relationships between search terms
- **Query Classification**: Determines the type of search needed

### **2. Intelligent JQL Generation**
Based on detected intent, the system generates optimized JQL queries:
```python
# Example: "Show me high priority bugs in the current sprint"
# Generated JQL:
"issuetype = Bug AND priority in ('High', 'Highest') AND sprint in openSprints() ORDER BY priority DESC, updated DESC"
```

### **3. Multi-Strategy Search**
The system combines multiple search approaches:
1. **Vector Search**: Semantic similarity in Milvus
2. **Board Search**: Project-specific searches
3. **Status Search**: Status-based filtering
4. **Sprint Search**: Sprint context awareness
5. **JQL Search**: Precise, targeted results ⭐ **NEW**

### **4. Result Optimization**
- **Deduplication**: Removes duplicate issues across search strategies
- **Priority Ranking**: JQL results get highest priority for precision
- **Context Enrichment**: Adds JQL query information to results

## 📊 Search Strategy Priority

1. **JQL Search** (Highest Priority) - Most precise results
2. **Sprint Search** - Sprint-aware results
3. **Status Search** - Status-filtered results
4. **Board Search** - Project-specific results
5. **Vector Search** - Semantic similarity results

## 💡 Usage Examples

### **Natural Language Queries**
```bash
# These automatically generate and execute JQL queries:

"Show me all high priority bugs in the current sprint"
"List stories assigned to me that are not done"
"Find issues with 'login' in the summary or description"
"Show me all tasks created this week"
"List unassigned stories in the backlog"
"Find issues with the 'urgent' label"
"Show me all bugs reported by John"
"List stories in project DEMO that are in progress"
```

### **Direct JQL Queries**
```bash
# Execute these directly via the JQL endpoint:

"project = DEMO AND status != Done"
"issuetype = Story AND assignee = currentUser()"
"sprint in openSprints() AND priority in (High, Highest)"
"created >= startOfWeek() AND status != Closed"
"summary ~ 'bug' OR description ~ 'error'"
"labels = 'urgent' AND status != Done"
```

## 🔍 JQL Query Patterns

### **Project Queries**
```jql
project = DEMO                    # Specific project
project in (DEMO, TEST)          # Multiple projects
```

### **Issue Type Queries**
```jql
issuetype = Story                # Single type
issuetype in (Story, Bug, Task)  # Multiple types
```

### **Status Queries**
```jql
status = 'To Do'                 # Specific status
status != Done                   # Exclude status
status in ('To Do', 'In Progress') # Multiple statuses
```

### **Priority Queries**
```jql
priority = High                  # Single priority
priority in ('High', 'Highest')  # Multiple priorities
priority is not EMPTY            # Has priority set
```

### **Assignee Queries**
```jql
assignee = currentUser()         # Current user
assignee = "John Doe"            # Specific user
assignee is EMPTY                # Unassigned
```

### **Date Queries**
```jql
created >= startOfDay()          # Today
created >= startOfWeek()         # This week
created >= startOfMonth()        # This month
updated >= -7d                   # Last 7 days
```

### **Text Search Queries**
```jql
summary ~ "login"                # Summary contains
description ~ "error"            # Description contains
summary ~ "login" OR description ~ "error"  # Either field
```

### **Sprint Queries**
```jql
sprint in openSprints()          # Current sprints
sprint in openSprints() AND status = 'To Do'  # Sprint backlog
```

### **Component & Label Queries**
```jql
component = "Frontend"           # Specific component
labels = "urgent"                # Specific label
labels in ("urgent", "blocker")  # Multiple labels
```

## 🚀 Performance Benefits

### **1. Precision**
- JQL queries return exact matches
- No false positives from semantic search
- Targeted results based on specific criteria

### **2. Speed**
- Direct database queries via JIRA API
- Reduced vector search overhead
- Optimized result filtering

### **3. Coverage**
- Combines multiple search strategies
- Ensures comprehensive result sets
- Eliminates gaps in search coverage

### **4. Real-time Accuracy**
- Forces real-time sync before JQL execution
- Always uses latest JIRA data
- No stale information

## 🔧 Configuration

### **Environment Variables**
```bash
# JQL-specific settings
JQL_MAX_RESULTS=100              # Maximum results per query
JQL_TIMEOUT=30                   # Query timeout in seconds
JQL_FORCE_SYNC=true             # Force sync before JQL queries
```

### **JQL Query Limits**
- **Max Query Length**: 1000 characters
- **Max Results**: 100 per query
- **Timeout**: 30 seconds
- **Rate Limiting**: Respects JIRA API limits

## 📈 Monitoring & Logging

### **Query Logging**
```bash
🔍 JQL Query generated: project = DEMO AND status != Done
📋 JQL search found 15 issues
✅ JQL-optimized answer generated
```

### **Performance Metrics**
- Query execution time
- Results count
- Success/failure rates
- API response times

## 🧪 Testing

Use the provided test script to verify JQL functionality:

```bash
python test_jql_functionality.py
```

This script tests:
- Direct JQL search execution
- JQL suggestions and field discovery
- Query parsing and validation
- Intelligent JQL generation

## 🔮 Future Enhancements

### **Planned Features**
1. **Query Templates**: Pre-built JQL templates for common searches
2. **Query History**: Save and reuse successful JQL queries
3. **Advanced Analytics**: Query performance metrics and optimization suggestions
4. **Query Builder UI**: Visual JQL query builder interface
5. **Query Optimization**: Automatic JQL query optimization for better performance

### **Integration Possibilities**
1. **Slack Integration**: JQL queries via Slack commands
2. **Email Integration**: Scheduled JQL reports
3. **Dashboard Integration**: Real-time JQL-based dashboards
4. **Webhook Support**: JQL-triggered webhooks for automation

## 📚 References

- [JIRA JQL API Documentation](https://developer.atlassian.com/cloud/jira/platform/rest/v2/api-group-jql/#api-group-jql)
- [JQL Syntax Reference](https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jql/)
- [JQL Functions](https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jql/#Functions)

## 🎯 Summary

The JQL-enhanced RAG system provides:

✅ **Intelligent Query Generation** - Natural language to JQL conversion
✅ **Precise Results** - Exact matches via JQL queries
✅ **Comprehensive Coverage** - Multiple search strategies combined
✅ **Real-time Accuracy** - Always up-to-date information
✅ **Performance Optimization** - Faster, more targeted searches
✅ **Developer-Friendly** - Direct JQL API access
✅ **Advanced Features** - Field discovery, query validation, parsing

This enhancement transforms the RAG system from a simple semantic search tool into a powerful, intelligent JIRA search engine that understands both natural language and technical JQL queries.
