#!/usr/bin/env python3
"""
LangChain-based JIRA Data Ingestion Script
Uses LangChain document loaders and vector stores for comprehensive JIRA data ingestion
"""

import os
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangChain components
try:
    from langchain_jira_loader import JiraDocumentLoader, JiraRealtimeLoader
    from langchain_milvus_store import MilvusVectorStore
    from langchain_rag_chain import create_jira_rag_chain
    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain components loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import LangChain components: {e}")
    LANGCHAIN_AVAILABLE = False

class LangChainJiraIngestor:
    """LangChain-based JIRA data ingestor"""
    
    def __init__(
        self,
        collection_name: str = "jira_data",
        project_key: str = "ALL",
        max_issues: int = 1000,
        days_back: int = 365,
        include_comments: bool = True,
        include_descriptions: bool = True
    ):
        """
        Initialize the LangChain JIRA ingestor
        
        Args:
            collection_name: Name of the Milvus collection
            project_key: JIRA project key to ingest
            max_issues: Maximum number of issues to fetch
            days_back: Number of days back to fetch issues from
            include_comments: Whether to include issue comments
            include_descriptions: Whether to include issue descriptions
        """
        self.collection_name = collection_name
        self.project_key = project_key
        self.max_issues = max_issues
        self.days_back = days_back
        self.include_comments = include_comments
        self.include_descriptions = include_descriptions
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain components are not available. Please install required dependencies.")
        
        # Initialize components
        self.vector_store = None
        self.loader = None
        self.rag_chain = None
    
    def initialize_components(self):
        """Initialize LangChain components"""
        try:
            # Create vector store
            self.vector_store = MilvusVectorStore(collection_name=self.collection_name)
            logger.info(f"✅ Vector store initialized: {self.collection_name}")
            
            # Create document loader
            self.loader = JiraDocumentLoader(
                project_key=self.project_key,
                max_results=self.max_issues,
                days_back=self.days_back,
                include_comments=self.include_comments,
                include_descriptions=self.include_descriptions
            )
            logger.info(f"✅ Document loader initialized for project: {self.project_key}")
            
            # Create RAG chain for testing
            self.rag_chain = create_jira_rag_chain(collection_name=self.collection_name)
            logger.info("✅ RAG chain initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def ingest_data(self, clear_existing: bool = False) -> dict:
        """
        Ingest JIRA data using LangChain components
        
        Args:
            clear_existing: Whether to clear existing data before ingestion
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info("🚀 Starting LangChain-based JIRA data ingestion...")
            
            # Initialize components
            self.initialize_components()
            
            # Clear existing data if requested
            if clear_existing:
                logger.info("🗑️ Clearing existing collection data...")
                self.vector_store.clear_collection()
                logger.info("✅ Collection cleared")
            
            # Load documents using LangChain loader
            logger.info("📚 Loading JIRA documents...")
            documents = self.loader.load()
            
            if not documents:
                return {
                    "success": False,
                    "message": "No JIRA documents found to ingest",
                    "documents_loaded": 0
                }
            
            logger.info(f"📄 Loaded {len(documents)} documents from JIRA")
            
            # Add documents to vector store
            logger.info("📥 Adding documents to vector store...")
            doc_ids = self.vector_store.add_documents(documents)
            
            # Get collection statistics
            stats = self.vector_store.get_collection_stats()
            
            # Test the RAG chain with a sample query
            test_result = self._test_rag_chain()
            
            result = {
                "success": True,
                "message": f"Successfully ingested {len(documents)} JIRA documents using LangChain",
                "documents_loaded": len(documents),
                "document_ids": doc_ids[:10] if doc_ids else [],  # Show first 10 IDs
                "collection_stats": stats,
                "project_key": self.project_key,
                "max_issues": self.max_issues,
                "days_back": self.days_back,
                "include_comments": self.include_comments,
                "include_descriptions": self.include_descriptions,
                "test_query_result": test_result
            }
            
            logger.info(f"✅ LangChain ingestion completed successfully: {len(documents)} documents")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during LangChain ingestion: {e}")
            return {
                "success": False,
                "message": f"Error during ingestion: {str(e)}",
                "error": str(e)
            }
    
    def _test_rag_chain(self) -> dict:
        """Test the RAG chain with a sample query"""
        try:
            if not self.rag_chain:
                return {"success": False, "message": "RAG chain not available"}
            
            # Test with a simple query
            test_query = "What are the recent issues?"
            result = self.rag_chain.process_query(test_query)
            
            return {
                "success": True,
                "test_query": test_query,
                "results_found": result.get("total_results", 0),
                "retrieval_strategy": result.get("retrieval_strategy", "unknown"),
                "answer_preview": result.get("answer", "")[:100] + "..." if result.get("answer") else "No answer"
            }
            
        except Exception as e:
            logger.warning(f"⚠️ RAG chain test failed: {e}")
            return {
                "success": False,
                "message": f"RAG chain test failed: {str(e)}",
                "error": str(e)
            }
    
    def ingest_recent_updates(self, days_back: int = 7) -> dict:
        """
        Ingest only recent JIRA updates using real-time loader
        
        Args:
            days_back: Number of days back to fetch recent updates
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"🔄 Starting recent updates ingestion (last {days_back} days)...")
            
            # Initialize components
            self.initialize_components()
            
            # Create real-time loader
            realtime_loader = JiraRealtimeLoader(
                project_key=self.project_key,
                max_results=self.max_issues,
                days_back=days_back,
                include_comments=self.include_comments,
                include_descriptions=self.include_descriptions
            )
            
            # Load recent documents
            recent_docs = realtime_loader.load_recent_updates()
            
            if not recent_docs:
                return {
                    "success": True,
                    "message": "No recent updates found",
                    "documents_loaded": 0
                }
            
            # Clear existing data and add new documents
            logger.info("🗑️ Clearing existing data for fresh ingestion...")
            self.vector_store.clear_collection()
            
            # Add recent documents
            doc_ids = self.vector_store.add_documents(recent_docs)
            
            # Get collection statistics
            stats = self.vector_store.get_collection_stats()
            
            result = {
                "success": True,
                "message": f"Successfully ingested {len(recent_docs)} recent JIRA documents",
                "documents_loaded": len(recent_docs),
                "document_ids": doc_ids[:10] if doc_ids else [],
                "collection_stats": stats,
                "days_back": days_back,
                "ingestion_type": "recent_updates"
            }
            
            logger.info(f"✅ Recent updates ingestion completed: {len(recent_docs)} documents")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during recent updates ingestion: {e}")
            return {
                "success": False,
                "message": f"Error during recent updates ingestion: {str(e)}",
                "error": str(e)
            }

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="LangChain-based JIRA data ingestion")
    parser.add_argument("--project-key", type=str, default="ALL", help="JIRA project key to ingest")
    parser.add_argument("--max-issues", type=int, default=1000, help="Maximum number of issues to fetch")
    parser.add_argument("--days-back", type=int, default=365, help="Number of days back to fetch issues from")
    parser.add_argument("--collection-name", type=str, default="jira_data", help="Milvus collection name")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing data before ingestion")
    parser.add_argument("--recent-only", action="store_true", help="Ingest only recent updates")
    parser.add_argument("--recent-days", type=int, default=7, help="Days back for recent updates (used with --recent-only)")
    parser.add_argument("--no-comments", action="store_true", help="Exclude issue comments")
    parser.add_argument("--no-descriptions", action="store_true", help="Exclude issue descriptions")
    
    args = parser.parse_args()
    
    try:
        # Create ingestor
        ingestor = LangChainJiraIngestor(
            collection_name=args.collection_name,
            project_key=args.project_key,
            max_issues=args.max_issues,
            days_back=args.days_back,
            include_comments=not args.no_comments,
            include_descriptions=not args.no_descriptions
        )
        
        # Perform ingestion
        if args.recent_only:
            result = ingestor.ingest_recent_updates(days_back=args.recent_days)
        else:
            result = ingestor.ingest_data(clear_existing=args.clear_existing)
        
        # Print results
        print("\n" + "="*60)
        print("LANGCHAIN JIRA DATA INGESTION RESULTS")
        print("="*60)
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        
        if result['success']:
            print(f"Documents Loaded: {result['documents_loaded']}")
            print(f"Project Key: {result.get('project_key', 'N/A')}")
            print(f"Collection: {args.collection_name}")
            
            if 'collection_stats' in result:
                stats = result['collection_stats']
                print(f"Collection Entities: {stats.get('num_entities', 'N/A')}")
            
            if 'test_query_result' in result and result['test_query_result']['success']:
                test = result['test_query_result']
                print(f"RAG Test Query: {test['test_query']}")
                print(f"RAG Test Results: {test['results_found']} documents found")
                print(f"RAG Retrieval Strategy: {test['retrieval_strategy']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print("="*60)
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease set the following environment variables:")
        print("JIRA_BASE_URL=https://your-domain.atlassian.net")
        print("JIRA_USERNAME=your-email@domain.com")
        print("JIRA_API_TOKEN=your-api-token")
        print("HF_TOKEN=your-huggingface-token (optional)")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease install required dependencies:")
        print("pip install langchain langchain-community langchain-openai")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
