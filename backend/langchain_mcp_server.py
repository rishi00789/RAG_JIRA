#!/usr/bin/env python3
"""
LangChain-powered MCP Server for JIRA RAG Assistant
Refactored to use LangChain components while maintaining FastMCP interface
"""

import asyncio
import json
import logging
import os
import traceback
import csv
import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangChain RAG components
try:
    from langchain_milvus_store import MilvusVectorStore
    from langchain_rag_chain import JiraRAGChain, create_jira_rag_chain
    from langchain_jira_loader import JiraDocumentLoader, JiraRealtimeLoader
    from jira_operations import get_jira_operations
    from action_detector import detect_jira_action, ActionType
    LANGCHAIN_AVAILABLE = True
    JIRA_OPS_AVAILABLE = True
    logger.info("✅ All LangChain modules loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    LANGCHAIN_AVAILABLE = False
    JIRA_OPS_AVAILABLE = False

class LangChainJIRARAGMCPServer:
    """LangChain-powered MCP Server with comprehensive RAG functionality"""
    
    def __init__(self):
        self.fastmcp = FastMCP("jira-rag-assistant-langchain")
        self.jira_ops = None
        self.rag_chain = None
        self.vector_store = None
        self.is_connected = False
        self.setup_tools()
        self.initialize_connections()
        
    def generate_sprint_report_csv(self, sprint_data: List[Dict]) -> str:
        """Generate CSV report for sprint data"""
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
    
    def generate_velocity_report_csv(self, velocity_data: List[Dict]) -> str:
        """Generate CSV report for velocity data"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Sprint', 'Start Date', 'End Date', 'Planned Story Points', 
            'Completed Story Points', 'Velocity', 'Issues Count', 'Completion Rate'
        ])
        
        # Write data rows
        for sprint in velocity_data:
            completion_rate = (sprint.get('completed_points', 0) / sprint.get('planned_points', 1)) * 100 if sprint.get('planned_points', 0) > 0 else 0
            writer.writerow([
                sprint.get('name', ''),
                sprint.get('start_date', ''),
                sprint.get('end_date', ''),
                sprint.get('planned_points', 0),
                sprint.get('completed_points', 0),
                sprint.get('velocity', 0),
                sprint.get('issues_count', 0),
                f"{completion_rate:.1f}%"
            ])
        
        return output.getvalue()
    
    def detect_report_request(self, question: str) -> Dict[str, Any]:
        """Detect if the question is requesting a report generation"""
        question_lower = question.lower()
        
        # Check for explicit CSV format requests
        is_csv_request = any(csv_keyword in question_lower for csv_keyword in [
            '.csv', 'csv format', 'csv file', 'as csv', 'in csv', 'export csv', 'download csv'
        ])
        
        # Sprint report detection
        if any(keyword in question_lower for keyword in ['sprint report', 'sprint summary', 'sprint status', 'current sprint', 'sprint data']):
            return {
                'type': 'sprint_report',
                'sprint_name': self.extract_sprint_name(question),
                'format': 'csv' if is_csv_request else 'json',
                'save_file': is_csv_request
            }
        
        # Velocity report detection
        if any(keyword in question_lower for keyword in ['velocity report', 'velocity chart', 'sprint velocity', 'team velocity', 'velocity data']):
            return {
                'type': 'velocity_report',
                'sprint_count': self.extract_sprint_count(question),
                'format': 'csv' if is_csv_request else 'json',
                'save_file': is_csv_request
            }
        
        return {'type': 'regular_query'}

    def extract_sprint_name(self, question: str) -> Optional[str]:
        """Extract sprint name from question"""
        import re
        question_lower = question.lower()
        
        # If it's just asking for sprint report without specific sprint, return current
        if any(phrase in question_lower for phrase in ['sprint report', 'sprint status', 'sprint summary', 'current sprint']):
            return 'current'
        
        # Look for specific sprint patterns
        patterns = [
            r'sprint\s+(\d+)',
            r'sprint\s+([a-zA-Z0-9\s]+)',
            r'current\s+sprint',
            r'active\s+sprint'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1) if match.groups() else 'current'
        
        # Default to current sprint if no specific sprint mentioned
        return 'current'

    def extract_sprint_count(self, question: str) -> int:
        """Extract number of sprints for velocity report"""
        import re
        question_lower = question.lower()
        
        # If no specific count mentioned, default to 5
        patterns = [
            r'last\s+(\d+)\s+sprints',
            r'(\d+)\s+sprints',
            r'past\s+(\d+)\s+sprints',
            r'(\d+)\s+recent\s+sprints'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return int(match.group(1))
        
        # Default to last 5 sprints if no specific count mentioned
        return 5

    def save_csv_to_downloads(self, csv_content: str, filename: str) -> str:
        """Save CSV content to downloads folder and return the file path"""
        try:
            # Create downloads directory if it doesn't exist
            downloads_dir = Path("downloads")
            downloads_dir.mkdir(exist_ok=True)
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = filename.replace(" ", "_").lower()
            full_filename = f"{base_name}_{timestamp}.csv"
            file_path = downloads_dir / full_filename
            
            # Write CSV content to file
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)
            
            logger.info(f"📁 CSV file saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save CSV file: {e}")
            return None

    def generate_sprint_data(self, sprint_name: Optional[str] = None) -> List[Dict]:
        """Generate sprint data from JIRA"""
        try:
            if not self.jira_ops:
                raise Exception("JIRA operations not available")
            
            # Get current sprint or specified sprint
            boards = self.jira_ops.get_agile_boards()
            if not boards:
                raise Exception("No agile boards found")
            
            sprint_id = None
            sprint_name_final = sprint_name
            
            if sprint_name == 'current' or sprint_name is None:
                # Get current active sprint from the first board
                board_id = boards[0]['id']
                current_sprint = self.jira_ops.get_current_sprint(board_id)
                if current_sprint:
                    sprint_id = current_sprint['id']
                    sprint_name_final = current_sprint.get('name', 'Current Sprint')
                else:
                    # Fallback: get the most recent sprint
                    all_sprints = self.jira_ops.get_sprints(board_id)
                    if all_sprints:
                        # Sort by start date and get the most recent
                        all_sprints.sort(key=lambda x: x.get('startDate', ''), reverse=True)
                        sprint_id = all_sprints[0]['id']
                        sprint_name_final = all_sprints[0].get('name', 'Recent Sprint')
                    else:
                        raise Exception("No sprints found")
            else:
                # Find sprint by name
                for board in boards:
                    board_sprints = self.jira_ops.get_sprints(board['id'])
                    for sprint in board_sprints:
                        if sprint.get('name', '').lower() == sprint_name.lower():
                            sprint_id = sprint['id']
                            sprint_name_final = sprint['name']
                            break
                    if sprint_id:
                        break
                
                if not sprint_id:
                    raise Exception(f"Sprint '{sprint_name}' not found")
            
            # Get sprint issues
            sprint_issues = self.jira_ops.get_sprint_stories(sprint_id)
            
            # Format sprint data
            sprint_data = []
            for issue in sprint_issues:
                sprint_data.append({
                    'key': issue.get('key', ''),
                    'summary': issue.get('summary', ''),
                    'issue_type': issue.get('issue_type', ''),
                    'status': issue.get('status', ''),
                    'assignee': issue.get('assignee', ''),
                    'story_points': issue.get('story_points', 0),
                    'priority': issue.get('priority', ''),
                    'created': issue.get('created', ''),
                    'updated': issue.get('updated', ''),
                    'sprint': sprint_name_final
                })
            
            return sprint_data
            
        except Exception as e:
            logger.error(f"Failed to generate sprint data: {e}")
            return []

    def generate_velocity_data(self, sprint_count: int = 5) -> List[Dict]:
        """Generate velocity data from JIRA"""
        try:
            if not self.jira_ops:
                raise Exception("JIRA operations not available")
            
            # Get agile boards
            boards = self.jira_ops.get_agile_boards()
            if not boards:
                raise Exception("No agile boards found")
            
            velocity_data = []
            
            # Get sprints from the first board
            board_id = boards[0]['id']
            sprints = self.jira_ops.get_sprints(board_id)
            
            if not sprints:
                raise Exception("No sprints found")
            
            # Sort sprints by start date (most recent first)
            sprints.sort(key=lambda x: x.get('startDate', ''), reverse=True)
            
            # Take the specified number of sprints
            recent_sprints = sprints[:sprint_count]
            
            for sprint in recent_sprints:
                sprint_id = sprint['id']
                sprint_name = sprint.get('name', 'Unknown Sprint')
                start_date = sprint.get('startDate', '')
                end_date = sprint.get('endDate', '')
                
                # Get sprint issues
                sprint_issues = self.jira_ops.get_sprint_stories(sprint_id)
                
                # Calculate metrics
                planned_points = 0
                completed_points = 0
                issues_count = len(sprint_issues)
                
                for issue in sprint_issues:
                    story_points = issue.get('story_points', 0)
                    status = issue.get('status', '').lower()
                    
                    planned_points += story_points
                    
                    if status in ['done', 'closed', 'resolved']:
                        completed_points += story_points
                
                velocity = completed_points
                completion_rate = (completed_points / planned_points * 100) if planned_points > 0 else 0
                
                velocity_data.append({
                    'name': sprint_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'planned_points': planned_points,
                    'completed_points': completed_points,
                    'velocity': velocity,
                    'issues_count': issues_count,
                    'completion_rate': completion_rate
                })
            
            return velocity_data
            
        except Exception as e:
            logger.error(f"Failed to generate velocity data: {e}")
            return []

    def initialize_connections(self):
        """Initialize Milvus, JIRA connections, and LangChain components"""
        try:
            # Initialize JIRA operations
            if JIRA_OPS_AVAILABLE:
                self.jira_ops = get_jira_operations()
                logger.info("✅ JIRA operations initialized")
            
            # Initialize LangChain components
            if LANGCHAIN_AVAILABLE:
                # Create vector store
                self.vector_store = MilvusVectorStore(collection_name="jira_data")
                logger.info("✅ Milvus vector store initialized")
                
                # Create RAG chain
                self.rag_chain = JiraRAGChain(
                    vector_store=self.vector_store,
                    jira_ops=self.jira_ops,
                    use_hybrid_retrieval=True,
                    use_sprint_retrieval=True,
                    max_retrieval_results=50
                )
                logger.info("✅ LangChain RAG chain initialized")
                
                self.is_connected = True
            else:
                logger.error("❌ LangChain components not available")
                self.is_connected = False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize connections: {e}")
            self.is_connected = False

    def setup_tools(self):
        """Setup the comprehensive rag_query tool using LangChain components"""
        
        @self.fastmcp.tool()
        async def rag_query(question: str, max_results: int = 50, fast_mode: bool = False) -> str:
            """
            Comprehensive RAG query tool powered by LangChain.
            Handles JIRA queries, actions, real-time sync, AI-powered responses, and report generation.
            
            Features:
            - Natural language JIRA queries with real-time data sync
            - Sprint report generation (CSV/JSON format)
            - Velocity report generation with charts (CSV/JSON format)
            - JIRA action execution (create, update, assign, transition)
            - Hybrid retrieval with vector search and JQL queries
            
            Report Generation Examples:
            - "Generate sprint report in CSV format" (saves to downloads folder)
            - "Create velocity report for last 5 sprints as CSV" (saves to downloads folder)
            - "Show current sprint status as CSV" (saves to downloads folder)
            - "Generate team velocity chart" (JSON format)
            - "Export sprint data to CSV file" (saves to downloads folder)
            
            Args:
                question: The user's question, request, or report generation command
                max_results: Maximum number of results to return (default: 50)
                fast_mode: Skip heavy operations for faster response (default: False)
            
            Returns:
                JSON string with comprehensive response including answer, sources, context, 
                action results, and CSV content for reports
            """
            import asyncio
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("RAG query timed out")
            
            # Set a 90-second timeout for the entire operation
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(90)
            
            try:
                logger.info(f"🔍 Processing query with LangChain: {question} (fast_mode: {fast_mode})")
                
                if not self.is_connected or not self.rag_chain:
                    return json.dumps({
                        "answer": "❌ LangChain RAG system not properly initialized. Please check the system configuration.",
                        "sources": [],
                        "context": [],
                        "success": False,
                        "error": "LangChain components not available"
                    }, indent=2)
                
                # Adjust timeout based on fast_mode
                if fast_mode:
                    signal.alarm(30)  # 30 seconds for fast mode
                    logger.info("⚡ Fast mode enabled - reduced timeout to 30 seconds")
                
                # 🔄 REAL-TIME SYNC: Always fetch latest JIRA data before query
                sync_performed = False
                enable_realtime_sync = os.getenv("ENABLE_REALTIME_SYNC", "true").lower() == "true"
                
                if self.vector_store and enable_realtime_sync:
                    try:
                        logger.info("🔄 Performing real-time sync for latest JIRA data...")
                        
                        # Use timeout for sync operation
                        def sync_timeout_handler(signum, frame):
                            raise TimeoutError("Sync operation timed out")
                        
                        signal.signal(signal.SIGALRM, sync_timeout_handler)
                        signal.alarm(5)  # 5 second timeout for sync
                        
                        try:
                            # Use LangChain document loader for real-time sync
                            from langchain_jira_loader import JiraRealtimeLoader
                            
                            realtime_loader = JiraRealtimeLoader(
                                project_key=os.getenv("JIRA_PROJECT_KEY", "ALL"),
                                max_results=int(os.getenv("REALTIME_MAX_RESULTS", "50")),
                                days_back=int(os.getenv("REALTIME_DAYS_BACK", "7"))
                            )
                            
                            # Load recent documents
                            recent_docs = realtime_loader.load_recent_updates()
                            
                            if recent_docs:
                                # Clear existing collection and add new documents
                                self.vector_store.clear_collection()
                                self.vector_store.add_documents(recent_docs)
                                sync_performed = True
                                logger.info(f"✅ Real-time sync completed - loaded {len(recent_docs)} documents")
                            else:
                                logger.info("ℹ️ No recent updates to sync")
                                
                        finally:
                            signal.alarm(0)  # Cancel the alarm
                    except TimeoutError:
                        logger.warning("⚠️ Real-time sync timed out, using existing data")
                    except Exception as e:
                        logger.warning(f"⚠️ Real-time sync failed, using existing data: {e}")
                else:
                    if not enable_realtime_sync:
                        logger.info("ℹ️ Real-time sync disabled, using existing data")
                
                # 📊 REPORT GENERATION: Check if this is a report request
                report_request = self.detect_report_request(question)
                
                # Skip real-time sync for report generation to avoid timeouts
                if report_request['type'] != 'regular_query':
                    sync_performed = False
                    logger.info("ℹ️ Skipping real-time sync for report generation to ensure fast response")
                
                if report_request['type'] == 'sprint_report':
                    logger.info("📊 Generating sprint report...")
                    try:
                        # Generate sprint data (synchronous call)
                        sprint_data = self.generate_sprint_data(report_request.get('sprint_name'))
                        
                        if report_request['format'] == 'csv':
                            csv_content = self.generate_sprint_report_csv(sprint_data)
                            
                            # Save CSV file if requested
                            file_path = None
                            if report_request.get('save_file', False):
                                sprint_name = report_request.get('sprint_name', 'current_sprint')
                                filename = f"sprint_report_{sprint_name}"
                                file_path = self.save_csv_to_downloads(csv_content, filename)
                            
                            response_message = f"📊 Sprint Report Generated\n\nReport contains {len(sprint_data)} issues from the sprint."
                            if file_path:
                                response_message += f"\n\n📁 CSV file saved to: {file_path}"
                            
                            return json.dumps({
                                "answer": response_message,
                                "sources": sprint_data,
                                "context": [],
                                "success": True,
                                "report_type": "sprint_report",
                                "report_format": "csv",
                                "csv_content": csv_content,
                                "file_saved": file_path is not None,
                                "file_path": file_path,
                                "sync_performed": sync_performed,
                                "sync_message": "Real-time sync completed - using latest data" if sync_performed else "Using existing data"
                            }, indent=2)
                        else:
                            return json.dumps({
                                "answer": f"📊 Sprint Report Generated\n\nReport contains {len(sprint_data)} issues from the sprint.",
                                "sources": sprint_data,
                                "context": [],
                                "success": True,
                                "report_type": "sprint_report",
                                "report_format": "json",
                                "sync_performed": sync_performed,
                                "sync_message": "Real-time sync completed - using latest data" if sync_performed else "Using existing data"
                            }, indent=2)
                    except Exception as e:
                        logger.error(f"❌ Sprint report generation failed: {e}")
                        return json.dumps({
                            "answer": f"❌ Failed to generate sprint report: {str(e)}",
                            "sources": [],
                            "context": [],
                            "success": False,
                            "error": str(e)
                        }, indent=2)
                
                elif report_request['type'] == 'velocity_report':
                    logger.info("📊 Generating velocity report...")
                    try:
                        # Generate velocity data (synchronous call)
                        velocity_data = self.generate_velocity_data(report_request.get('sprint_count', 5))
                        
                        if report_request['format'] == 'csv':
                            csv_content = self.generate_velocity_report_csv(velocity_data)
                            
                            # Save CSV file if requested
                            file_path = None
                            if report_request.get('save_file', False):
                                sprint_count = report_request.get('sprint_count', 5)
                                filename = f"velocity_report_{sprint_count}_sprints"
                                file_path = self.save_csv_to_downloads(csv_content, filename)
                            
                            response_message = f"📊 Velocity Report Generated\n\nReport contains data for {len(velocity_data)} sprints."
                            if file_path:
                                response_message += f"\n\n📁 CSV file saved to: {file_path}"
                            
                            return json.dumps({
                                "answer": response_message,
                                "sources": velocity_data,
                                "context": [],
                                "success": True,
                                "report_type": "velocity_report",
                                "report_format": "csv",
                                "csv_content": csv_content,
                                "file_saved": file_path is not None,
                                "file_path": file_path,
                                "sync_performed": sync_performed,
                                "sync_message": "Real-time sync completed - using latest data" if sync_performed else "Using existing data"
                            }, indent=2)
                        else:
                            return json.dumps({
                                "answer": f"📊 Velocity Report Generated\n\nReport contains data for {len(velocity_data)} sprints.",
                                "sources": velocity_data,
                                "context": [],
                                "success": True,
                                "report_type": "velocity_report",
                                "report_format": "json",
                                "sync_performed": sync_performed,
                                "sync_message": "Real-time sync completed - using latest data" if sync_performed else "Using existing data"
                            }, indent=2)
                    except Exception as e:
                        logger.error(f"❌ Velocity report generation failed: {e}")
                        return json.dumps({
                            "answer": f"❌ Failed to generate velocity report: {str(e)}",
                            "sources": [],
                            "context": [],
                            "success": False,
                            "error": str(e)
                        }, indent=2)
                
                # Process query using LangChain RAG chain
                logger.info("🔍 Processing query with LangChain RAG chain...")
                
                # Configure RAG chain based on fast_mode
                if fast_mode:
                    # Use simpler retrieval for fast mode
                    self.rag_chain.max_retrieval_results = min(max_results, 10)
                    self.rag_chain.use_hybrid_retrieval = False
                else:
                    # Use full hybrid retrieval for comprehensive results
                    self.rag_chain.max_retrieval_results = min(max_results, 50)
                    self.rag_chain.use_hybrid_retrieval = True
                
                # Process the query
                result = self.rag_chain.process_query(
                    question=question,
                    include_sprint_context=not fast_mode,
                    include_jql_info=not fast_mode
                )
                
                # Add processing information to result
                if sync_performed:
                    result["sync_performed"] = True
                    result["sync_message"] = "Real-time sync completed - using latest data"
                else:
                    result["sync_performed"] = False
                    result["sync_message"] = "Using existing vector store data"
                
                logger.info(f"✅ LangChain RAG query completed successfully")
                return json.dumps(result, indent=2)
                
            except TimeoutError as te:
                logger.error(f"⏰ Query timed out: {te}")
                return json.dumps({
                    "answer": "Query timed out. The system is processing a large amount of data. Please try a more specific query or try again later.",
                    "sources": [],
                    "context": [],
                    "success": False,
                    "error": "timeout",
                    "message": "Request timed out after 45 seconds"
                }, indent=2)
            except Exception as e:
                logger.error(f"❌ Error processing query: {e}")
                return json.dumps({
                    "answer": f"Error processing query: {str(e)}",
                    "sources": [],
                    "context": [],
                    "success": False,
                    "error": str(e)
                }, indent=2)
            finally:
                # Always cancel the alarm
                signal.alarm(0)
        
        @self.fastmcp.tool()
        async def ingest_jira_data(
            project_key: str = "ALL",
            max_issues: int = 1000,
            days_back: int = 365,
            clear_existing: bool = False
        ) -> str:
            """
            Ingest JIRA data using LangChain document loader.
            
            Args:
                project_key: JIRA project key to ingest (default: "ALL")
                max_issues: Maximum number of issues to fetch
                days_back: Number of days back to fetch issues from
                clear_existing: Whether to clear existing data before ingestion
            
            Returns:
                JSON string with ingestion results
            """
            try:
                logger.info(f"🔄 Starting JIRA data ingestion with LangChain...")
                
                if not self.is_connected or not self.vector_store:
                    return json.dumps({
                        "success": False,
                        "message": "LangChain components not available",
                        "error": "System not properly initialized"
                    }, indent=2)
                
                # Set timeout for ingestion operation
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Data ingestion timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 minutes timeout for ingestion
                
                # Create JIRA document loader
                loader = JiraDocumentLoader(
                    project_key=project_key,
                    max_results=max_issues,
                    days_back=days_back,
                    include_comments=True,
                    include_descriptions=True
                )
                
                # Clear existing data if requested
                if clear_existing:
                    self.vector_store.clear_collection()
                    logger.info("🗑️ Cleared existing collection data")
                
                # Load documents
                documents = loader.load()
                
                if not documents:
                    return json.dumps({
                        "success": False,
                        "message": "No JIRA documents found to ingest",
                        "documents_loaded": 0
                    }, indent=2)
                
                # Add documents to vector store
                doc_ids = self.vector_store.add_documents(documents)
                
                # Get collection stats
                stats = self.vector_store.get_collection_stats()
                
                result = {
                    "success": True,
                    "message": f"Successfully ingested {len(documents)} JIRA documents",
                    "documents_loaded": len(documents),
                    "document_ids": doc_ids[:10] if doc_ids else [],  # Show first 10 IDs
                    "collection_stats": stats,
                    "project_key": project_key,
                    "max_issues": max_issues,
                    "days_back": days_back
                }
                
                logger.info(f"✅ JIRA data ingestion completed: {len(documents)} documents")
                return json.dumps(result, indent=2)
                
            except Exception as e:
                logger.error(f"❌ Error during JIRA data ingestion: {e}")
                return json.dumps({
                    "success": False,
                    "message": f"Error during ingestion: {str(e)}",
                    "error": str(e)
                }, indent=2)
        
        @self.fastmcp.tool()
        async def get_collection_stats() -> str:
            """
            Get statistics about the JIRA data collection.
            
            Returns:
                JSON string with collection statistics
            """
            try:
                if not self.vector_store:
                    return json.dumps({
                        "success": False,
                        "message": "Vector store not available"
                    }, indent=2)
                
                stats = self.vector_store.get_collection_stats()
                
                result = {
                    "success": True,
                    "collection_stats": stats,
                    "rag_chain_available": self.rag_chain is not None,
                    "jira_ops_available": self.jira_ops is not None,
                    "langchain_available": LANGCHAIN_AVAILABLE
                }
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                logger.error(f"❌ Error getting collection stats: {e}")
                return json.dumps({
                    "success": False,
                    "message": f"Error getting stats: {str(e)}",
                    "error": str(e)
                }, indent=2)
    
    def start(self):
        """Start the LangChain-powered MCP server"""
        try:
            logger.info("✅ LangChain JIRA RAG MCP Server initialized successfully")
            
            # Start the FastMCP server using streamable-http transport (blocks)
            self.fastmcp.run(
                transport="streamable-http",
                host="127.0.0.1",
                port=8003,  # Different port to avoid conflicts
                path="/mcp"
            )
        except Exception as e:
            logger.error(f"❌ Failed to start LangChain MCP server: {e}")
            traceback.print_exc()
            raise

# Create server instance for FastMCP dev command
mcp = LangChainJIRARAGMCPServer()

if __name__ == "__main__":
    mcp.start()
