# JIRA RAG MCP Server Implementation

## 🚀 Overview

This implementation converts your JIRA RAG Assistant into a **Model Context Protocol (MCP) Server** using `fastmcp`. The MCP server exposes all JIRA operations as tools that LLMs can automatically call based on user prompts, eliminating the need for manual action detection and execution.

## 🔧 What is MCP?

**Model Context Protocol (MCP)** is a standard that allows AI models to:
- Discover available tools
- Call tools with appropriate parameters
- Receive structured responses
- Automatically execute actions based on user intent

## 🎯 Key Benefits

1. **Automatic Tool Selection**: LLMs automatically choose the right JIRA tool based on user prompts
2. **No Manual Action Detection**: Eliminates the need for regex-based action detection
3. **Structured Tool Calls**: All JIRA operations are exposed as well-defined tools
4. **LLM-Driven Execution**: The AI model decides what actions to take and when
5. **Extensible Architecture**: Easy to add new tools and capabilities

## 🛠️ Available Tools

### Core JIRA Operations
- **`jira_get_issue`** - Get detailed issue information
- **`jira_search_issues`** - Search issues using JQL
- **`jira_create_issue`** - Create new issues
- **`jira_update_issue`** - Update issue fields
- **`jira_transition_issue`** - Change issue status
- **`jira_add_comment`** - Add comments to issues
- **`jira_assign_issue`** - Assign issues to users

### Sprint & Agile Management
- **`jira_get_agile_boards`** - Get all agile boards
- **`jira_get_board_issues`** - Get issues from specific boards
- **`jira_get_current_sprint`** - Get active sprint
- **`jira_get_sprint_stories`** - Get stories from sprints
- **`jira_create_sprint`** - Create new sprints

### Advanced Operations
- **`jira_get_worklog`** - Get time tracking data
- **`jira_add_worklog`** - Add time tracking entries
- **`jira_get_transitions`** - Get available status changes
- **`jira_link_to_epic`** - Link issues to epics
- **`jira_create_issue_link`** - Create links between issues

### Project Management
- **`jira_get_all_projects`** - Get all projects
- **`jira_get_project_issues`** - Get issues from projects
- **`jira_get_project_versions`** - Get project versions
- **`jira_create_version`** - Create new versions

### Utility Tools
- **`detect_user_intent`** - Analyze user prompts and suggest tools
- **`rag_query`** - Query the RAG system for context

## 🚀 How It Works

### 1. User Prompt Processing
```
User: "assign 3 story points to SCRUM-1"
```

### 2. LLM Tool Selection
The LLM automatically:
- Analyzes the user's intent
- Selects the appropriate tool (`jira_update_issue`)
- Determines the correct parameters

### 3. Tool Execution
```python
# LLM automatically calls:
await jira_update_issue(
    issue_key="SCRUM-1",
    field="story_points",
    value=3
)
```

### 4. Result Return
The tool executes and returns the result to the LLM, which then provides a natural language response to the user.

## 📁 File Structure

```
rag-assistant/
├── backend/
│   ├── mcp_server.py          # Main MCP server
│   ├── mcp_client.py          # Test client
│   ├── jira_operations.py     # JIRA API operations
│   ├── action_detector.py     # Legacy action detection
│   └── server.py              # Legacy FastAPI server
├── start_mcp_server.sh        # MCP server startup script
└── MCP_IMPLEMENTATION.md      # This file
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
cd rag-assistant
source venv/bin/activate
pip install fastmcp mcp
```

### 2. Start the MCP Server
```bash
# Option 1: Use the startup script
./start_mcp_server.sh

# Option 2: Manual start
cd backend
python mcp_server.py
```

### 3. Test the MCP Client
```bash
cd backend
python mcp_client.py
```

## 🧪 Testing Examples

### Example 1: Story Points Update
```
User: "assign 3 story points to SCRUM-1"
LLM Action: Calls jira_update_issue(issue_key="SCRUM-1", field="story_points", value=3)
Result: ✅ Successfully updated issue SCRUM-1
```

### Example 2: Issue Creation
```
User: "create a new story called 'User Authentication' in SCRUM project"
LLM Action: Calls jira_create_issue(project_key="SCRUM", issue_type="Story", summary="User Authentication")
Result: ✅ Created new story SCRUM-13
```

### Example 3: Status Transition
```
User: "move SCRUM-5 to In Progress"
LLM Action: Calls jira_transition_issue(issue_key="SCRUM-5", transition_name="Start Progress")
Result: ✅ Successfully transitioned SCRUM-5 to In Progress
```

## 🔌 Integration with LLMs

### Claude Integration
```python
# The LLM automatically discovers and uses tools
tools = await session.list_tools()
# LLM sees all available JIRA operations and chooses appropriately
```

### Custom LLM Integration
```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # LLM can now call any available tool
        result = await session.call_tool("jira_update_issue", {
            "issue_key": "SCRUM-1",
            "field": "story_points",
            "value": 3
        })
```

## 🎯 Use Cases

### 1. Natural Language JIRA Management
- "Create a bug report for the login issue"
- "Move all my tasks to Done"
- "Show me the current sprint progress"

### 2. Automated Workflow Execution
- "Start the development phase for SCRUM-10"
- "Add 2 hours of work to SCRUM-5"
- "Link SCRUM-8 to the authentication epic"

### 3. Contextual Information Retrieval
- "What's the status of my assigned issues?"
- "Show me all high-priority bugs"
- "List all stories in the current sprint"

## 🔧 Configuration

### Environment Variables
```bash
# Required for JIRA operations
JIRA_URL=your-jira-instance.com
JIRA_USERNAME=your-username
JIRA_API_TOKEN=your-api-token

# Optional for enhanced features
HUGGINGFACE_TOKEN=your-hf-token
```

### Tool Customization
You can easily add new tools by extending the `setup_tools()` method in `mcp_server.py`:

```python
@self.fastmcp.tool()
async def custom_jira_operation(param1: str, param2: int) -> str:
    """Custom JIRA operation description"""
    # Your custom logic here
    return "Operation result"
```

## 🚀 Migration from Legacy System

### Before (Legacy RAG)
1. User sends prompt
2. Action detector uses regex to identify intent
3. Manual execution of JIRA operations
4. Response generation

### After (MCP)
1. User sends prompt
2. LLM automatically selects appropriate tools
3. Tools execute JIRA operations
4. LLM generates natural response

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install fastmcp mcp
   ```

2. **JIRA Connection**: Verify environment variables are set correctly

3. **Tool Execution**: Check logs for detailed error messages

4. **Permission Issues**: Ensure the startup script is executable
   ```bash
   chmod +x start_mcp_server.sh
   ```

### Debug Mode
Enable detailed logging by modifying the logging level in `mcp_server.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🎉 Benefits Over Legacy System

| Feature | Legacy RAG | MCP Server |
|---------|------------|------------|
| Action Detection | Regex-based | LLM-driven |
| Tool Selection | Manual | Automatic |
| Extensibility | Limited | High |
| Error Handling | Basic | Comprehensive |
| User Experience | Good | Excellent |
| Maintenance | High | Low |

## 🚀 Next Steps

1. **Test the MCP server** with the provided client
2. **Integrate with your preferred LLM** (Claude, GPT, etc.)
3. **Customize tools** based on your specific needs
4. **Add new capabilities** as requirements evolve

## 📚 Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [JIRA REST API Documentation](https://developer.atlassian.com/cloud/jira/platform/rest/v3/)

---

**🎯 The MCP server transforms your JIRA RAG from a rule-based system to an intelligent, LLM-driven assistant that automatically executes the right actions based on user intent!**
