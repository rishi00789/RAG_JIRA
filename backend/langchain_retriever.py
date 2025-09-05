#!/usr/bin/env python3
"""
LangChain-based Retriever for JIRA RAG
Custom retriever that combines multiple search strategies for comprehensive JIRA data retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore

from langchain_milvus_store import MilvusVectorStore
from jira_operations import get_jira_operations

logger = logging.getLogger(__name__)

class JiraHybridRetriever(BaseRetriever):
    """Hybrid retriever that combines vector search, JQL queries, and metadata filtering"""
    
    vector_store: Optional[MilvusVectorStore] = None
    jira_ops: Optional[Any] = None
    k: int = 10
    jql_k: int = 20
    use_jql: bool = True
    use_metadata_filter: bool = True
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        jira_ops: Optional[Any] = None,
        k: int = 10,
        jql_k: int = 20,
        use_jql: bool = True,
        use_metadata_filter: bool = True,
        **kwargs
    ):
        """
        Initialize the hybrid retriever
        
        Args:
            vector_store: Milvus vector store instance
            jira_ops: JIRA operations instance
            k: Number of vector search results
            jql_k: Number of JQL search results
            use_jql: Whether to use JQL queries
            use_metadata_filter: Whether to use metadata filtering
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.jira_ops = jira_ops or get_jira_operations()
        self.k = k
        self.jql_k = jql_k
        self.use_jql = use_jql
        self.use_metadata_filter = use_metadata_filter
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents using hybrid approach"""
        try:
            logger.info(f"🔍 Hybrid retrieval for query: {query}")
            
            all_documents = []
            seen_issue_keys = set()
            
            # 1. Vector similarity search
            vector_docs = self._vector_search(query)
            logger.info(f"📊 Vector search found {len(vector_docs)} documents")
            
            for doc in vector_docs:
                issue_key = doc.metadata.get('issue_key', '')
                if issue_key and issue_key not in seen_issue_keys:
                    all_documents.append(doc)
                    seen_issue_keys.add(issue_key)
            
            # 2. JQL-based search (if enabled and JIRA ops available)
            if self.use_jql and self.jira_ops:
                jql_docs = self._jql_search(query)
                logger.info(f"🔍 JQL search found {len(jql_docs)} documents")
                
                for doc in jql_docs:
                    issue_key = doc.metadata.get('issue_key', '')
                    if issue_key and issue_key not in seen_issue_keys:
                        all_documents.append(doc)
                        seen_issue_keys.add(issue_key)
            
            # 3. Metadata-based filtering (if enabled)
            if self.use_metadata_filter:
                filtered_docs = self._metadata_filter(all_documents, query)
                logger.info(f"🏷️ Metadata filtering: {len(all_documents)} -> {len(filtered_docs)} documents")
                all_documents = filtered_docs
            
            # 4. Re-rank and limit results
            final_docs = self._rerank_documents(all_documents, query)
            
            logger.info(f"✅ Hybrid retrieval completed: {len(final_docs)} unique documents")
            return final_docs
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid retrieval: {e}")
            return []
    
    def _vector_search(self, query: str) -> List[Document]:
        """Perform vector similarity search"""
        try:
            return self.vector_store.similarity_search(query, k=self.k)
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []
    
    def _jql_search(self, query: str) -> List[Document]:
        """Perform JQL-based search"""
        try:
            # Generate JQL query from natural language
            jql_query = self._generate_jql_query(query)
            if not jql_query:
                return []
            
            logger.info(f"🔍 Generated JQL: {jql_query}")
            
            # Execute JQL search
            jql_results = self.jira_ops.search_issues(jql_query, max_results=self.jql_k)
            
            if not jql_results or not jql_results.get("success"):
                logger.warning("⚠️ JQL search failed or returned no results")
                return []
            
            # Convert JQL results to documents
            documents = []
            for issue in jql_results.get("issues", []):
                doc = self._jql_result_to_document(issue)
                if doc:
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ JQL search error: {e}")
            return []
    
    def _generate_jql_query(self, query: str) -> Optional[str]:
        """Generate JQL query from natural language"""
        query_lower = query.lower()
        jql_parts = []
        
        # Project-based queries
        if any(keyword in query_lower for keyword in ['project', 'in project', 'from project']):
            import re
            project_match = re.search(r'project\s+(\w+)', query_lower)
            if project_match:
                project_key = project_match.group(1).upper()
                jql_parts.append(f'project = {project_key}')
        
        # Issue type queries
        if any(keyword in query_lower for keyword in ['story', 'stories', 'bug', 'bugs', 'task', 'tasks', 'epic', 'epics']):
            if 'story' in query_lower:
                jql_parts.append("issuetype = Story")
            elif 'bug' in query_lower:
                jql_parts.append("issuetype = Bug")
            elif 'task' in query_lower:
                jql_parts.append("issuetype = Task")
            elif 'epic' in query_lower:
                jql_parts.append("issuetype = Epic")
        
        # Status-based queries
        if any(keyword in query_lower for keyword in ['todo', 'to do', 'in progress', 'inprogress', 'done', 'completed', 'blocked']):
            if any(keyword in query_lower for keyword in ['todo', 'to do']):
                jql_parts.append("status = 'To Do'")
            elif 'in progress' in query_lower or 'inprogress' in query_lower:
                jql_parts.append("status = 'In Progress'")
            elif any(keyword in query_lower for keyword in ['done', 'completed']):
                jql_parts.append("status = Done")
            elif 'blocked' in query_lower:
                jql_parts.append("status = Blocked")
        
        # Assignee-based queries
        if any(keyword in query_lower for keyword in ['assigned to', 'assignedto', 'my issues', 'myissues', 'unassigned']):
            if any(keyword in query_lower for keyword in ['my issues', 'myissues']):
                jql_parts.append("assignee = currentUser()")
            elif 'unassigned' in query_lower:
                jql_parts.append("assignee is EMPTY")
        
        # Date-based queries
        if any(keyword in query_lower for keyword in ['today', 'yesterday', 'this week', 'thisweek', 'this month', 'thismonth', 'recent', 'recently']):
            if 'today' in query_lower:
                jql_parts.append("created >= startOfDay()")
            elif 'yesterday' in query_lower:
                jql_parts.append("created >= startOfDay(-1d) AND created < startOfDay()")
            elif 'this week' in query_lower or 'thisweek' in query_lower:
                jql_parts.append("created >= startOfWeek()")
            elif 'this month' in query_lower or 'thismonth' in query_lower:
                jql_parts.append("created >= startOfMonth()")
            elif any(keyword in query_lower for keyword in ['recent', 'recently']):
                jql_parts.append("updated >= -7d")
        
        # Sprint-based queries
        if any(keyword in query_lower for keyword in ['sprint', 'current sprint', 'currentsprint', 'sprint backlog', 'sprintbacklog']):
            if 'current sprint' in query_lower or 'currentsprint' in query_lower:
                jql_parts.append("sprint in openSprints()")
            elif 'sprint backlog' in query_lower or 'sprintbacklog' in query_lower:
                jql_parts.append("sprint in openSprints() AND status = 'To Do'")
            else:
                jql_parts.append("sprint in openSprints()")
        
        # If no specific criteria found, create a general active issues query
        if not jql_parts:
            jql_parts.append("status != Done AND status != Closed AND status != Resolved")
        
        # Build the JQL query
        jql_query = " AND ".join(jql_parts)
        if jql_parts and jql_query.strip():
            jql_query += " ORDER BY updated DESC"
        
        return jql_query
    
    def _jql_result_to_document(self, issue: Dict) -> Optional[Document]:
        """Convert JQL result to LangChain document"""
        try:
            fields = issue.get('fields', {}) or {}
            assignee = fields.get('assignee') or {}
            reporter = fields.get('reporter') or {}
            issuetype = fields.get('issuetype') or {}
            project = fields.get('project') or {}
            status = fields.get('status') or {}
            priority = fields.get('priority') or {}
            
            # Create document content
            content_parts = []
            if fields.get('summary'):
                content_parts.append(f"Summary: {fields['summary']}")
            if fields.get('description'):
                content_parts.append(f"Description: {fields['description']}")
            
            if not content_parts:
                return None
            
            content = "\n\n".join(content_parts)
            
            # Create metadata
            metadata = {
                'issue_key': issue.get('key', ''),
                'issue_type': issuetype.get('name', ''),
                'content_type': 'jql_search',
                'project_key': project.get('key', ''),
                'status': status.get('name', ''),
                'priority': priority.get('name', ''),
                'assignee': assignee.get('displayName', 'Unassigned'),
                'reporter': reporter.get('displayName', 'Unknown'),
                'created_date': fields.get('created', ''),
                'updated_date': fields.get('updated', ''),
                'source': 'jql'
            }
            
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            logger.error(f"❌ Error converting JQL result to document: {e}")
            return None
    
    def _metadata_filter(self, documents: List[Document], query: str) -> List[Document]:
        """Filter documents based on metadata relevance"""
        query_lower = query.lower()
        filtered_docs = []
        
        for doc in documents:
            metadata = doc.metadata
            relevance_score = 0
            
            # Check issue type relevance
            issue_type = metadata.get('issue_type', '').lower()
            if any(keyword in query_lower for keyword in ['story', 'stories']) and 'story' in issue_type:
                relevance_score += 2
            elif any(keyword in query_lower for keyword in ['bug', 'bugs']) and 'bug' in issue_type:
                relevance_score += 2
            elif any(keyword in query_lower for keyword in ['task', 'tasks']) and 'task' in issue_type:
                relevance_score += 2
            
            # Check status relevance
            status = metadata.get('status', '').lower()
            if any(keyword in query_lower for keyword in ['todo', 'to do']) and 'to do' in status:
                relevance_score += 1
            elif any(keyword in query_lower for keyword in ['in progress', 'inprogress']) and 'in progress' in status:
                relevance_score += 1
            elif any(keyword in query_lower for keyword in ['done', 'completed']) and 'done' in status:
                relevance_score += 1
            
            # Check priority relevance
            priority = metadata.get('priority', '').lower()
            if any(keyword in query_lower for keyword in ['high', 'urgent', 'critical']) and 'high' in priority:
                relevance_score += 1
            elif any(keyword in query_lower for keyword in ['low', 'minor']) and 'low' in priority:
                relevance_score += 1
            
            # Check project relevance
            project_key = metadata.get('project_key', '').lower()
            if project_key in query_lower:
                relevance_score += 2
            
            # Include document if it has some relevance or if no specific criteria match
            if relevance_score > 0 or not any(keyword in query_lower for keyword in ['story', 'bug', 'task', 'todo', 'done', 'high', 'low']):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _rerank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Re-rank documents based on relevance to query"""
        try:
            # Simple re-ranking based on content and metadata relevance
            query_lower = query.lower()
            scored_docs = []
            
            for doc in documents:
                score = 0
                content = doc.page_content.lower()
                metadata = doc.metadata
                
                # Content relevance
                query_words = query_lower.split()
                for word in query_words:
                    if word in content:
                        score += 1
                
                # Metadata relevance
                if metadata.get('content_type') == 'summary':
                    score += 2  # Prioritize summaries
                elif metadata.get('content_type') == 'description':
                    score += 1
                
                # Status relevance
                status = metadata.get('status', '').lower()
                if 'in progress' in status:
                    score += 1  # Prioritize active issues
                
                # Priority relevance
                priority = metadata.get('priority', '').lower()
                if 'high' in priority:
                    score += 1
                
                scored_docs.append((doc, score))
            
            # Sort by score (descending) and return documents
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            logger.error(f"❌ Error re-ranking documents: {e}")
            return documents

class JiraSprintRetriever(BaseRetriever):
    """Specialized retriever for sprint-related queries"""
    
    vector_store: Optional[MilvusVectorStore] = None
    jira_ops: Optional[Any] = None
    k: int = 15
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        jira_ops: Optional[Any] = None,
        k: int = 15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.jira_ops = jira_ops or get_jira_operations()
        self.k = k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        """Retrieve sprint-related documents"""
        try:
            query_lower = query.lower()
            documents = []
            
            # Check if it's a current sprint query
            if any(keyword in query_lower for keyword in ['current sprint', 'this sprint', 'active sprint']):
                sprint_info = self.jira_ops.get_current_sprint()
                if sprint_info.get("success") and sprint_info.get("current_sprint"):
                    logger.info(f"🏃 Current sprint found: {sprint_info['current_sprint'].get('name', 'Unknown')}")
            
            # Check if it's a sprint backlog query
            elif any(keyword in query_lower for keyword in ['sprint backlog', 'backlog', 'not started']):
                backlog_info = self.jira_ops.get_sprint_backlog()
                if backlog_info.get("success"):
                    logger.info(f"📋 Sprint backlog has {backlog_info.get('total_backlog', 0)} stories")
            
            # Check if it's a sprint progress query
            elif any(keyword in query_lower for keyword in ['sprint progress', 'progress', 'completion', 'metrics']):
                progress_info = self.jira_ops.get_sprint_progress()
                if progress_info.get("success"):
                    progress = progress_info.get("progress", {})
                    logger.info(f"📊 Sprint progress: {progress.get('completion_percentage', 0)}% complete")
            
            # Perform vector search for sprint-related content
            sprint_docs = self.vector_store.similarity_search(query, k=self.k)
            documents.extend(sprint_docs)
            
            logger.info(f"🏃 Sprint retrieval found {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error in sprint retrieval: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Test the retrievers
    try:
        # Create vector store
        vector_store = MilvusVectorStore(collection_name="test_jira_data")
        
        # Create hybrid retriever
        hybrid_retriever = JiraHybridRetriever(
            vector_store=vector_store,
            k=5,
            jql_k=10
        )
        
        # Test retrieval
        query = "Show me all high priority stories in progress"
        results = hybrid_retriever._get_relevant_documents(query)
        
        print(f"\n🔍 Hybrid retrieval results for: '{query}'")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
        
    except Exception as e:
        print(f"❌ Error testing retrievers: {e}")
