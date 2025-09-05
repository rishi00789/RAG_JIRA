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
from typing import Any, Dict, List, Optional, Union
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
    from langchain_jira_loader import JiraDocumentLoader
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
            Handles JIRA queries, actions, real-time sync, and AI-powered responses using LangChain components.
            
            Args:
                question: The user's question or request
                max_results: Maximum number of results to return (default: 50)
                fast_mode: Skip heavy operations for faster response (default: False)
            
            Returns:
                JSON string with comprehensive response including answer, sources, context, and action results
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
                
                # Process query using existing vector store data
                
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
