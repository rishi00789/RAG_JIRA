#!/usr/bin/env python3
"""
LangChain-based JIRA Document Loader
Custom document loader for JIRA data that integrates with LangChain's document processing pipeline
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

# Load environment variables
load_dotenv()

class JiraDocumentLoader(BaseLoader):
    """Custom LangChain document loader for JIRA data"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        api_token: Optional[str] = None,
        project_key: str = "ALL",
        max_results: int = 1000,
        days_back: int = 365,
        include_comments: bool = True,
        include_descriptions: bool = True
    ):
        """
        Initialize the JIRA document loader
        
        Args:
            base_url: JIRA base URL (defaults to JIRA_BASE_URL env var)
            username: JIRA username (defaults to JIRA_USERNAME env var)
            api_token: JIRA API token (defaults to JIRA_API_TOKEN env var)
            project_key: JIRA project key to filter by (defaults to "ALL")
            max_results: Maximum number of issues to fetch
            days_back: Number of days back to fetch issues from
            include_comments: Whether to include issue comments
            include_descriptions: Whether to include issue descriptions
        """
        self.base_url = base_url or os.getenv("JIRA_BASE_URL")
        self.username = username or os.getenv("JIRA_USERNAME")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN")
        self.project_key = project_key
        self.max_results = max_results
        self.days_back = days_back
        self.include_comments = include_comments
        self.include_descriptions = include_descriptions
        
        if not all([self.base_url, self.username, self.api_token]):
            raise ValueError(
                "Missing required JIRA environment variables. Please set JIRA_BASE_URL, "
                "JIRA_USERNAME, and JIRA_API_TOKEN or pass them as parameters."
            )
        
        # Setup session
        self.session = requests.Session()
        self.session.auth = (self.username, self.api_token)
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        # Set timeout for all requests
        self.session.timeout = (10, 30)  # (connect timeout, read timeout)
    
    def load(self) -> List[Document]:
        """Load JIRA documents"""
        documents = []
        
        try:
            # Fetch JIRA issues
            issues = self._fetch_jira_issues()
            
            for issue in issues:
                # Create documents for each issue
                issue_docs = self._create_issue_documents(issue)
                documents.extend(issue_docs)
            
            print(f"✅ Loaded {len(documents)} documents from {len(issues)} JIRA issues")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading JIRA documents: {e}")
            return []
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load JIRA documents (memory efficient for large datasets)"""
        try:
            # Fetch JIRA issues
            issues = self._fetch_jira_issues()
            
            for issue in issues:
                # Create documents for each issue
                issue_docs = self._create_issue_documents(issue)
                for doc in issue_docs:
                    yield doc
                    
        except Exception as e:
            print(f"❌ Error lazy loading JIRA documents: {e}")
            return
    
    def _fetch_jira_issues(self) -> List[Dict]:
        """Fetch JIRA issues from the API"""
        print(f"🔍 Fetching JIRA issues from the last {self.days_back} days...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        # JQL query to get issues
        jql_query = f"updated >= '{start_date.strftime('%Y-%m-%d')}' ORDER BY updated DESC"
        
        if self.project_key != "ALL":
            jql_query = f"project = {self.project_key} AND {jql_query}"
        
        url = f"{self.base_url}/rest/api/2/search"
        params = {
            'jql': jql_query,
            'maxResults': self.max_results,
            'fields': 'summary,description,comment,issuetype,project,status,priority,assignee,reporter,created,updated,labels,components'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            issues = data.get('issues', [])
            
            print(f"✅ Fetched {len(issues)} JIRA issues")
            return issues
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching JIRA issues: {e}")
            return []
    
    def _fetch_jira_comments(self, issue_key: str) -> List[Dict]:
        """Fetch comments for a specific issue"""
        if not self.include_comments:
            return []
            
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/comment"
        
        try:
            response = self.session.get(url, timeout=(5, 15))  # Shorter timeout for comments
            response.raise_for_status()
            
            data = response.json()
            return data.get('comments', [])
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching comments for {issue_key}: {e}")
            return []
    
    def _create_issue_documents(self, issue: Dict) -> List[Document]:
        """Create LangChain documents from a JIRA issue"""
        documents = []
        issue_key = issue['key']
        fields = issue['fields']
        
        # Extract metadata
        metadata = {
            'issue_key': issue_key,
            'issue_type': fields.get('issuetype', {}).get('name', 'Unknown') if fields.get('issuetype') else 'Unknown',
            'project_key': fields.get('project', {}).get('key', 'Unknown') if fields.get('project') else 'Unknown',
            'status': fields.get('status', {}).get('name', 'Unknown') if fields.get('status') else 'Unknown',
            'priority': fields.get('priority', {}).get('name', 'Unknown') if fields.get('priority') else 'Unknown',
            'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned') if fields.get('assignee') else 'Unassigned',
            'reporter': fields.get('reporter', {}).get('displayName', 'Unknown') if fields.get('reporter') else 'Unknown',
            'created_date': fields.get('created', ''),
            'updated_date': fields.get('updated', ''),
            'labels': [label for label in fields.get('labels', [])],
            'components': [comp.get('name', '') for comp in fields.get('components', []) if comp],
            'source': 'jira'
        }
        
        # Create document for issue summary
        if fields.get('summary'):
            summary_metadata = metadata.copy()
            summary_metadata['content_type'] = 'summary'
            
            documents.append(Document(
                page_content=fields['summary'],
                metadata=summary_metadata
            ))
        
        # Create document for issue description
        if self.include_descriptions and fields.get('description'):
            description_metadata = metadata.copy()
            description_metadata['content_type'] = 'description'
            
            documents.append(Document(
                page_content=fields['description'],
                metadata=description_metadata
            ))
        
        # Create documents for comments
        if self.include_comments:
            comments = self._fetch_jira_comments(issue_key)
            for comment in comments:
                if comment.get('body'):
                    comment_metadata = metadata.copy()
                    comment_metadata['content_type'] = 'comment'
                    comment_metadata['comment_author'] = comment.get('author', {}).get('displayName', 'Unknown') if comment.get('author') else 'Unknown'
                    comment_metadata['comment_created'] = comment.get('created', '')
                    comment_metadata['comment_updated'] = comment.get('updated', '')
                    
                    documents.append(Document(
                        page_content=comment['body'],
                        metadata=comment_metadata
                    ))
        
        return documents

class JiraRealtimeLoader(JiraDocumentLoader):
    """Real-time JIRA document loader for live data synchronization"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override defaults for real-time sync
        self.max_results = kwargs.get('max_results', 500)
        self.days_back = kwargs.get('days_back', 30)
    
    def load_recent_updates(self) -> List[Document]:
        """Load only recently updated JIRA documents"""
        print("🔄 Loading recent JIRA updates...")
        
        # Calculate date range for recent updates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        # JQL query for recent updates
        jql_query = f"updated >= '{start_date.strftime('%Y-%m-%d')}' ORDER BY updated DESC"
        
        if self.project_key != "ALL":
            jql_query = f"project = {self.project_key} AND {jql_query}"
        
        url = f"{self.base_url}/rest/api/2/search"
        params = {
            'jql': jql_query,
            'maxResults': self.max_results,
            'fields': 'summary,description,comment,issuetype,project,status,priority,assignee,reporter,created,updated,labels,components'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=(10, 30))
            response.raise_for_status()
            
            data = response.json()
            issues = data.get('issues', [])
            
            documents = []
            for issue in issues:
                issue_docs = self._create_issue_documents(issue)
                documents.extend(issue_docs)
            
            print(f"✅ Loaded {len(documents)} recent documents from {len(issues)} updated issues")
            return documents
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error loading recent JIRA updates: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Test the loader
    try:
        loader = JiraDocumentLoader(
            project_key="ALL",
            max_results=10,
            days_back=30
        )
        
        # Load documents
        documents = loader.load()
        
        print(f"\n📚 Loaded {len(documents)} documents:")
        for i, doc in enumerate(documents[:3]):  # Show first 3
            print(f"\nDocument {i+1}:")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease set the following environment variables:")
        print("JIRA_BASE_URL=https://your-domain.atlassian.net")
        print("JIRA_USERNAME=your-email@domain.com")
        print("JIRA_API_TOKEN=your-api-token")
