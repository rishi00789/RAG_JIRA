# Enhanced JIRA RAG System

## Overview

The Enhanced JIRA RAG System is a comprehensive solution that combines **Retrieval-Augmented Generation (RAG)** with **JIRA write operations**. It can now not only read and analyze JIRA data but also create, update, delete, and manage JIRA issues through natural language commands.

## 🚀 Key Features

### 1. **Intelligent Action Detection**
- Automatically detects whether your request is a query or an action
- Parses natural language to extract action parameters
- Supports multiple action types: create, update, assign, transition, comment, delete

### 2. **Natural Language JIRA Operations**
- **Create Issues**: "Create a story called 'User Login Feature' with 5 story points"
- **Update Issues**: "Update SCRUM-1 to have high priority and 8 story points"
- **Assign Issues**: "Assign SCRUM-2 to john.doe"
- **Status Transitions**: "Move SCRUM-3 to 'In Progress'"
- **Add Comments**: "Comment on SCRUM-1 saying 'Development started'"
- **Delete Issues**: "Delete SCRUM-4"

### 3. **Enhanced RAG Capabilities**
- Real-time JIRA data synchronization
- Intelligent context extraction from JIRA issues
- Enhanced LLM prompts for better JIRA-related responses
- Action-aware responses with execution results

### 4. **Comprehensive API Endpoints**
- `/query` - Enhanced RAG endpoint with action detection
- `/jira/action` - Direct JIRA action execution
- `/jira/project` - Project information
- `/jira/issue-types` - Available issue types
- `/jira/search` - JQL-based search
- `/jira/issue/{key}` - Specific issue details

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Action Detector │───▶│  JIRA Ops      │
│                 │    │                  │    │  (POST/PUT/    │
└─────────────────┘    └──────────────────┘    │  DELETE)        │
                                               └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RAG Query     │───▶│   Milvus DB      │───▶│  JIRA Cloud    │
│   (Read)        │    │   (Vector)       │    │  (REST API)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Supported Actions

### Create Operations
```bash
# Create stories
"Create a story called 'User Authentication' with description 'Implement secure login system' and 5 story points"

# Create bugs
"Create a bug called 'Login Button Not Working' with high priority"

# Create tasks
"Create a task called 'Update Documentation' for john.doe"
```

### Update Operations
```bash
# Update story points
"Update SCRUM-1 to have 8 story points"

# Update priority
"Change SCRUM-2 priority to highest"

# Update assignee
"Assign SCRUM-3 to alice.smith"

# Update multiple fields
"Update SCRUM-4 to have high priority, 13 story points, and assign to bob.wilson"
```

### Workflow Operations
```bash
# Status transitions
"Move SCRUM-1 to 'In Progress'"
"Change SCRUM-2 status to 'Done'"
"Transition SCRUM-3 to 'Review'"
```

### Communication Operations
```bash
# Add comments
"Comment on SCRUM-1 saying 'Development completed, ready for testing'"
"Add comment to SCRUM-2: 'Blocked by dependency issue'"
```

### Management Operations
```bash
# Assign issues
"Assign SCRUM-1 to john.doe"
"Give SCRUM-2 to alice.smith"

# Delete issues
"Delete SCRUM-5"
"Remove SCRUM-6"
```

## 🔧 Setup and Configuration

### 1. Environment Variables
```bash
# Copy env.example to .env
cp env.example .env

# Edit .env with your JIRA credentials
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your-email@domain.com
JIRA_API_TOKEN=your-api-token-here
JIRA_PROJECT_KEY=SCRUM  # or "ALL" for all projects
HF_TOKEN=your-huggingface-token
SYNC_CACHE_DURATION=60
```

### 2. Install Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Start Services
```bash
# Start Milvus and dependencies
docker-compose up -d

# Start the enhanced RAG server
cd rag-assistant
source venv/bin/activate
uvicorn backend.server:app --reload --port 8000
```

## 🧪 Testing the System

### 1. Run the Test Suite
```bash
python test_enhanced_jira_rag.py
```

### 2. Manual Testing Examples

#### Test Action Detection
```bash
# Create a story
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Create a story called \"API Integration\" with 3 story points", "max_results": 3}'

# Update an issue
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Update SCRUM-1 to have high priority", "max_results": 3}'
```

#### Test Direct JIRA Actions
```bash
# Create issue directly
curl -X POST "http://localhost:8000/jira/action" \
  -H "Content-Type: application/json" \
  -d '{"action": "create", "parameters": {"summary": "Test Story", "description": "Test description", "issue_type": "Story"}}'

# Search issues
curl "http://localhost:8000/jira/search?jql=ORDER%20BY%20updated%20DESC&max_results=5"
```

## 🔍 How It Works

### 1. **Action Detection Process**
1. User sends a natural language request
2. Action detector analyzes the text using regex patterns
3. If an action is detected, parameters are extracted
4. Action is executed via JIRA API
5. Result is returned to user

### 2. **RAG Query Process**
1. If no action is detected, request is processed as a RAG query
2. Real-time sync ensures latest JIRA data
3. Query is embedded and searched in Milvus
4. Relevant context is retrieved
5. LLM generates answer using context

### 3. **Real-time Synchronization**
- Automatically syncs JIRA data before each query
- Detects content changes using hashing
- Maintains data freshness with intelligent caching

## 📊 API Reference

### Enhanced Query Endpoint
```http
POST /query
Content-Type: application/json

{
  "question": "string",
  "max_results": 3
}
```

**Response includes:**
- `answer`: Generated answer or action result
- `sources`: Source information
- `context`: Retrieved context
- `action_type`: Type of action detected (if any)
- `action_summary`: Summary of action executed

### JIRA Action Endpoint
```http
POST /jira/action
Content-Type: application/json

{
  "action": "create|update|assign|transition|comment|delete|story_points",
  "parameters": {...}
}
```

### JIRA Information Endpoints
- `GET /jira/project` - Project information
- `GET /jira/issue-types` - Available issue types
- `GET /jira/search?jql=...&max_results=...` - JQL search
- `GET /jira/issue/{issue_key}` - Specific issue details

## 🎯 Use Cases

### 1. **Scrum Masters**
- "Create a story for user authentication with 5 story points"
- "Move all stories in 'To Do' to 'In Progress'"
- "Assign the highest priority bug to our senior developer"

### 2. **Developers**
- "What's the status of the login feature story?"
- "Update my current story to have 8 story points"
- "Comment on SCRUM-1 that I've completed the backend"

### 3. **Project Managers**
- "Show me all high-priority issues"
- "Create an epic for the Q2 release"
- "What's the current sprint velocity?"

### 4. **QA Engineers**
- "Create a bug for the login page crash"
- "Move all tested stories to 'Done'"
- "What are the current blocking issues?"

## 🚨 Error Handling

### Common Issues and Solutions

1. **JIRA API Authentication**
   - Verify `JIRA_API_TOKEN` is correct
   - Check `JIRA_USERNAME` format
   - Ensure JIRA account has proper permissions

2. **Action Detection Failures**
   - Use clear, specific language
   - Include issue keys for updates (e.g., "SCRUM-1")
   - Specify action type clearly

3. **Milvus Connection Issues**
   - Ensure Docker Compose is running
   - Check Milvus service status
   - Verify collection exists

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

## 📚 Examples and Tutorials

### Getting Started
1. Set up your environment variables
2. Start the system
3. Try a simple query: "What's the status of our latest story?"
4. Try an action: "Create a story called 'Test Story'"
5. Explore more complex operations

### Advanced Usage
- Combine multiple actions in one request
- Use JQL for complex searches
- Automate repetitive tasks
- Build custom workflows

## 🤝 Support and Contributing

### Getting Help
- Check the logs for detailed error messages
- Verify your JIRA API permissions
- Test with simple queries first
- Use the test suite to validate functionality

### Contributing
- Report issues with detailed descriptions
- Suggest new action patterns
- Contribute to the action detector
- Enhance error handling and validation

---

**The Enhanced JIRA RAG System transforms your JIRA experience from read-only to fully interactive, allowing you to manage your projects using natural language while maintaining the power of intelligent context retrieval and generation.**
