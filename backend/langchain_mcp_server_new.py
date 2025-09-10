#!/usr/bin/env python3
"""
LangChain-powered MCP Server for JIRA RAG
Simplified version with separate, focused tools
"""

import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime
import csv
import io
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    logger.warning("FastMCP not available")
    FASTMCP_AVAILABLE = False

try:
    from jira_operations import JiraOperations
    JIRA_OPS_AVAILABLE = True
except ImportError:
    logger.warning("JiraOperations not available")
    JIRA_OPS_AVAILABLE = False

try:
    from langchain_rag_chain import create_jira_rag_chain
    from langchain_milvus_store import create_milvus_vector_store
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain components not available")
    LANGCHAIN_AVAILABLE = False

class LangChainJIRARAGMCPServer:
    """Simplified LangChain-powered MCP Server with focused tools"""
    
    def __init__(self):
        self.fastmcp = FastMCP("jira-rag-assistant-simple")
        self.jira_ops = None
        self.rag_chain = None
        self.vector_store = None
        self.is_connected = False
        self.setup_tools()
        self.initialize_connections()
        
    def initialize_connections(self):
        """Initialize JIRA and LangChain connections"""
        try:
            # Initialize JIRA operations
            if JIRA_OPS_AVAILABLE:
                self.jira_ops = JiraOperations()
                logger.info("✅ JIRA operations initialized")
            
            # Initialize LangChain components
            if LANGCHAIN_AVAILABLE:
                self.vector_store = create_milvus_vector_store()
                self.rag_chain = create_jira_rag_chain()
                logger.info("✅ LangChain components initialized")
            
            self.is_connected = True
            logger.info("✅ All connections established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize connections: {e}")
            self.is_connected = False

    def setup_tools(self):
        """Setup focused MCP tools"""
        
        @self.fastmcp.tool()
        async def rag_query(question: str, max_results: int = 50) -> str:
            """
            Simple RAG query tool for JIRA questions and answers.
            Focuses only on Q&A without report generation complexity.
            
            Args:
                question: The user's question about JIRA data
                max_results: Maximum number of results to return (default: 50)
            
            Returns:
                JSON string with answer and sources
            """
            try:
                logger.info(f"🔍 Processing RAG query: {question}")
                
                if not self.is_connected or not self.rag_chain:
                    return json.dumps({
                        "answer": "❌ RAG system not initialized",
                        "sources": [],
                        "error": "System not ready"
                    })
                
                # Perform real-time sync if enabled
                sync_performed = False
                enable_realtime_sync = os.getenv("ENABLE_REALTIME_SYNC", "true").lower() == "true"
                
                if self.vector_store and enable_realtime_sync:
                    try:
                        logger.info("🔄 Performing real-time sync...")
                        from langchain_jira_loader import JiraRealtimeLoader
                        
                        realtime_loader = JiraRealtimeLoader(
                            project_key=os.getenv("JIRA_PROJECT_KEY", "ALL"),
                            max_results=int(os.getenv("REALTIME_MAX_RESULTS", "50")),
                            days_back=int(os.getenv("REALTIME_DAYS_BACK", "7"))
                        )
                        
                        recent_docs = realtime_loader.load_recent_updates()
                        
                        if recent_docs:
                            self.vector_store.clear_collection()
                            self.vector_store.add_documents(recent_docs)
                            sync_performed = True
                            logger.info(f"✅ Real-time sync completed - loaded {len(recent_docs)} documents")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Real-time sync failed: {e}")
                
                # Process the query
                result = self.rag_chain.process_query(question, max_results=max_results)
                
                return json.dumps({
                    "answer": result.get("answer", "No answer generated"),
                    "sources": result.get("sources", []),
                    "context": result.get("context", ""),
                    "sync_performed": sync_performed,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Error in RAG query: {e}")
                return json.dumps({
                    "answer": f"Error processing query: {str(e)}",
                    "sources": [],
                    "error": str(e)
                })
        
        @self.fastmcp.tool()
        async def sprint_report(sprint_name: str = "current", format: str = "json", save_file: bool = False) -> str:
            """
            Generate sprint report with issues and details.
            
            Args:
                sprint_name: Sprint name or "current" for active sprint (default: "current")
                format: Output format - "json" or "csv" (default: "json")
                save_file: Whether to save CSV file to downloads folder (default: False)
            
            Returns:
                JSON string with sprint data or CSV content
            """
            try:
                logger.info(f"📊 Generating sprint report for: {sprint_name}")
                
                if not self.jira_ops:
                    return json.dumps({
                        "success": False,
                        "error": "JIRA operations not available"
                    })
                
                # Get sprint data
                sprint_data = self._get_sprint_data(sprint_name)
                
                if not sprint_data:
                    return json.dumps({
                        "success": False,
                        "error": "No sprint data found"
                    })
                
                if format.lower() == "csv":
                    csv_content = self._generate_sprint_csv(sprint_data)
                    
                    if save_file:
                        file_path = self._save_csv_to_downloads(csv_content, f"sprint_report_{sprint_name}")
                        return json.dumps({
                            "success": True,
                            "format": "csv",
                            "content": csv_content,
                            "file_saved": True,
                            "file_path": file_path,
                            "sprint_name": sprint_name,
                            "issues_count": len(sprint_data)
                        })
                    else:
                        return json.dumps({
                            "success": True,
                            "format": "csv",
                            "content": csv_content,
                            "file_saved": False,
                            "sprint_name": sprint_name,
                            "issues_count": len(sprint_data)
                        })
                else:
                    return json.dumps({
                        "success": True,
                        "format": "json",
                        "data": sprint_data,
                        "sprint_name": sprint_name,
                        "issues_count": len(sprint_data)
                    })
                
            except Exception as e:
                logger.error(f"❌ Error generating sprint report: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        @self.fastmcp.tool()
        async def velocity_report(sprint_count: int = 5, format: str = "json", save_file: bool = False) -> str:
            """
            Generate velocity report for multiple sprints.
            
            Args:
                sprint_count: Number of recent sprints to analyze (default: 5)
                format: Output format - "json" or "csv" (default: "json")
                save_file: Whether to save CSV file to downloads folder (default: False)
            
            Returns:
                JSON string with velocity data or CSV content
            """
            try:
                logger.info(f"📈 Generating velocity report for last {sprint_count} sprints")
                
                if not self.jira_ops:
                    return json.dumps({
                        "success": False,
                        "error": "JIRA operations not available"
                    })
                
                # Get velocity data
                velocity_data = self._get_velocity_data(sprint_count)
                
                if not velocity_data:
                    return json.dumps({
                        "success": False,
                        "error": "No velocity data found"
                    })
                
                if format.lower() == "csv":
                    csv_content = self._generate_velocity_csv(velocity_data)
                    
                    if save_file:
                        file_path = self._save_csv_to_downloads(csv_content, f"velocity_report_{sprint_count}_sprints")
                        return json.dumps({
                            "success": True,
                            "format": "csv",
                            "content": csv_content,
                            "file_saved": True,
                            "file_path": file_path,
                            "sprint_count": sprint_count,
                            "sprints_analyzed": len(velocity_data)
                        })
                    else:
                        return json.dumps({
                            "success": True,
                            "format": "csv",
                            "content": csv_content,
                            "file_saved": False,
                            "sprint_count": sprint_count,
                            "sprints_analyzed": len(velocity_data)
                        })
                else:
                    return json.dumps({
                        "success": True,
                        "format": "json",
                        "data": velocity_data,
                        "sprint_count": sprint_count,
                        "sprints_analyzed": len(velocity_data)
                    })
                
            except Exception as e:
                logger.error(f"❌ Error generating velocity report: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
        
        @self.fastmcp.tool()
        async def jira_action(action: str, issue_key: str = None, summary: str = None, description: str = None, assignee: str = None, status: str = None, comment: str = None) -> str:
            """
            Execute JIRA actions like create, update, assign, transition issues.
            
            Args:
                action: Action to perform (create, update, assign, transition, comment)
                issue_key: JIRA issue key (required for most actions)
                summary: Issue summary (for create/update)
                description: Issue description (for create/update)
                assignee: Assignee username (for assign)
                status: New status (for transition)
                comment: Comment text (for comment)
            
            Returns:
                JSON string with action result
            """
            try:
                logger.info(f"🔧 Executing JIRA action: {action}")
                
                if not self.jira_ops:
                    return json.dumps({
                        "success": False,
                        "error": "JIRA operations not available"
                    })
                
                # Prepare parameters based on action
                params = {}
                if summary:
                    params['summary'] = summary
                if description:
                    params['description'] = description
                if assignee:
                    params['assignee'] = assignee
                if status:
                    params['status'] = status
                if comment:
                    params['comment'] = comment
                
                result = self.jira_ops.execute_action(action, issue_key, **params)
                
                return json.dumps({
                    "success": True,
                    "action": action,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"❌ Error executing JIRA action: {e}")
                return json.dumps({
                    "success": False,
                    "action": action,
                    "error": str(e)
                })
        
        @self.fastmcp.tool()
        async def get_collection_stats() -> str:
            """Get Milvus collection statistics"""
            try:
                if not self.vector_store:
                    return json.dumps({
                        "success": False,
                        "error": "Vector store not available"
                    })
                
                stats = self.vector_store.get_collection_stats()
                return json.dumps({
                    "success": True,
                    "stats": stats
                })
                
            except Exception as e:
                logger.error(f"❌ Error getting collection stats: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
    
    def _get_sprint_data(self, sprint_name: str) -> List[Dict]:
        """Get sprint data from JIRA"""
        try:
            boards = self.jira_ops.get_agile_boards()
            if not boards:
                return []
            
            sprint_id = None
            if sprint_name == "current":
                board_id = boards[0]['id']
                current_sprint = self.jira_ops.get_current_sprint(board_id)
                if current_sprint:
                    sprint_id = current_sprint['id']
                else:
                    # Fallback to most recent sprint
                    all_sprints = self.jira_ops.get_sprints(board_id)
                    if all_sprints:
                        all_sprints.sort(key=lambda x: x.get('startDate', ''), reverse=True)
                        sprint_id = all_sprints[0]['id']
            else:
                # Find sprint by name
                for board in boards:
                    sprints = self.jira_ops.get_sprints(board['id'])
                    for sprint in sprints:
                        if sprint.get('name', '').lower() == sprint_name.lower():
                            sprint_id = sprint['id']
                            break
                    if sprint_id:
                        break
            
            if not sprint_id:
                return []
            
            # Get sprint issues
            issues = self.jira_ops.get_sprint_stories(sprint_id)
            
            # Format the data
            sprint_data = []
            for issue in issues:
                fields = issue.get('fields', {})
                sprint_data.append({
                    'key': issue.get('key', ''),
                    'summary': fields.get('summary', ''),
                    'issue_type': fields.get('issuetype', {}).get('name', ''),
                    'status': fields.get('status', {}).get('name', ''),
                    'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned'),
                    'story_points': fields.get('customfield_10016', ''),
                    'priority': fields.get('priority', {}).get('name', ''),
                    'created': fields.get('created', ''),
                    'updated': fields.get('updated', ''),
                    'sprint': sprint_name
                })
            
            return sprint_data
            
        except Exception as e:
            logger.error(f"❌ Error getting sprint data: {e}")
            return []
    
    def _get_velocity_data(self, sprint_count: int) -> List[Dict]:
        """Get velocity data for multiple sprints"""
        try:
            boards = self.jira_ops.get_agile_boards()
            if not boards:
                return []
            
            board_id = boards[0]['id']
            all_sprints = self.jira_ops.get_sprints(board_id)
            
            if not all_sprints:
                return []
            
            # Sort by start date and get recent sprints
            all_sprints.sort(key=lambda x: x.get('startDate', ''), reverse=True)
            recent_sprints = all_sprints[:sprint_count]
            
            velocity_data = []
            for sprint in recent_sprints:
                sprint_id = sprint['id']
                issues = self.jira_ops.get_sprint_stories(sprint_id)
                
                planned_points = 0
                completed_points = 0
                completed_issues = 0
                
                for issue in issues:
                    fields = issue.get('fields', {})
                    story_points = fields.get('customfield_10016', 0)
                    status = fields.get('status', {}).get('name', '').lower()
                    
                    if story_points:
                        planned_points += story_points
                        if 'done' in status or 'closed' in status or 'resolved' in status:
                            completed_points += story_points
                            completed_issues += 1
                
                velocity = completed_points
                completion_rate = (completed_points / planned_points * 100) if planned_points > 0 else 0
                
                velocity_data.append({
                    'sprint': sprint.get('name', ''),
                    'start_date': sprint.get('startDate', ''),
                    'end_date': sprint.get('endDate', ''),
                    'planned_story_points': planned_points,
                    'completed_story_points': completed_points,
                    'velocity': velocity,
                    'issues_count': len(issues),
                    'completion_rate': round(completion_rate, 2)
                })
            
            return velocity_data
            
        except Exception as e:
            logger.error(f"❌ Error getting velocity data: {e}")
            return []
    
    def _generate_sprint_csv(self, sprint_data: List[Dict]) -> str:
        """Generate CSV content for sprint data"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Issue Key', 'Summary', 'Issue Type', 'Status', 'Assignee', 
            'Story Points', 'Priority', 'Created', 'Updated', 'Sprint'
        ])
        
        # Write data rows
        for issue in sprint_data:
            writer.writerow([
                issue.get('key', ''),
                issue.get('summary', ''),
                issue.get('issue_type', ''),
                issue.get('status', ''),
                issue.get('assignee', ''),
                issue.get('story_points', ''),
                issue.get('priority', ''),
                issue.get('created', ''),
                issue.get('updated', ''),
                issue.get('sprint', '')
            ])
        
        return output.getvalue()
    
    def _generate_velocity_csv(self, velocity_data: List[Dict]) -> str:
        """Generate CSV content for velocity data"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Sprint', 'Start Date', 'End Date', 'Planned Story Points', 
            'Completed Story Points', 'Velocity', 'Issues Count', 'Completion Rate'
        ])
        
        # Write data rows
        for sprint in velocity_data:
            writer.writerow([
                sprint.get('sprint', ''),
                sprint.get('start_date', ''),
                sprint.get('end_date', ''),
                sprint.get('planned_story_points', ''),
                sprint.get('completed_story_points', ''),
                sprint.get('velocity', ''),
                sprint.get('issues_count', ''),
                sprint.get('completion_rate', '')
            ])
        
        return output.getvalue()
    
    def _save_csv_to_downloads(self, csv_content: str, filename: str) -> str:
        """Save CSV content to downloads folder"""
        try:
            # Create downloads directory
            downloads_dir = Path("downloads")
            downloads_dir.mkdir(exist_ok=True)
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_filename = f"{filename}_{timestamp}.csv"
            file_path = downloads_dir / full_filename
            
            # Write CSV content
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)
            
            logger.info(f"✅ CSV file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save CSV file: {e}")
            return None
    
    def start(self):
        """Start the MCP server"""
        try:
            logger.info("✅ Simplified LangChain JIRA RAG MCP Server starting...")
            
            self.fastmcp.run(
                transport="streamable-http",
                host="127.0.0.1",
                port=8003,
                path="/mcp"
            )
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server: {e}")
            traceback.print_exc()
            raise

# Create server instance
mcp = LangChainJIRARAGMCPServer()

if __name__ == "__main__":
    mcp.start()
