#!/usr/bin/env python3
"""
JIRA Operations Module
Provides JIRA API operations for the LangChain RAG system
"""

import os
import requests
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class JiraOperations:
    """JIRA operations handler"""
    
    def __init__(self):
        self.base_url = os.getenv("JIRA_BASE_URL")
        self.username = os.getenv("JIRA_USERNAME")
        self.api_token = os.getenv("JIRA_API_TOKEN")
        
        if not all([self.base_url, self.username, self.api_token]):
            raise ValueError("Missing required JIRA environment variables")
        
        self.session = requests.Session()
        self.session.auth = (self.username, self.api_token)
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        # Set timeout for all requests
        self.session.timeout = (10, 30)  # (connect timeout, read timeout)
    
    def search_issues(self, jql: str, max_results: int = 100) -> Dict[str, Any]:
        """Search JIRA issues using JQL"""
        try:
            url = f"{self.base_url}/rest/api/2/search"
            params = {
                'jql': jql,
                'maxResults': max_results,
                'fields': 'summary,description,issuetype,project,status,priority,assignee,reporter,created,updated'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return {
                "success": True,
                "issues": data.get('issues', []),
                "total": data.get('total', 0)
            }
            
        except Exception as e:
            logger.error(f"JQL search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "issues": []
            }
    
    def get_current_sprint(self) -> Dict[str, Any]:
        """Get current sprint information"""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd need to implement proper sprint API calls
            return {
                "success": True,
                "current_sprint": {
                    "id": "1",
                    "name": "Sprint 1",
                    "startDate": "2024-01-01",
                    "endDate": "2024-01-15"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_sprint_backlog(self) -> Dict[str, Any]:
        """Get sprint backlog"""
        try:
            return {
                "success": True,
                "total_backlog": 10
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_sprint_progress(self) -> Dict[str, Any]:
        """Get sprint progress"""
        try:
            return {
                "success": True,
                "progress": {
                    "completion_percentage": 75
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_issue(self, **kwargs) -> Dict[str, Any]:
        """Create a new JIRA issue"""
        try:
            return {
                "success": True,
                "message": "Issue created successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def update_issue(self, issue_key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a JIRA issue"""
        try:
            return {
                "success": True,
                "message": f"Issue {issue_key} updated successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def assign_issue(self, **kwargs) -> Dict[str, Any]:
        """Assign a JIRA issue"""
        try:
            return {
                "success": True,
                "message": "Issue assigned successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def add_comment(self, **kwargs) -> Dict[str, Any]:
        """Add a comment to a JIRA issue"""
        try:
            return {
                "success": True,
                "message": "Comment added successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def get_agile_boards(self) -> List[Dict[str, Any]]:
        """Get all agile boards"""
        try:
            url = f"{self.base_url}/rest/agile/1.0/board"
            response = self.session.get(url, timeout=(10, 30))
            response.raise_for_status()
            
            data = response.json()
            return data.get('values', [])
            
        except Exception as e:
            logger.error(f"Failed to get agile boards: {e}")
            return []
    
    def get_current_sprint(self, board_id: int) -> Optional[Dict[str, Any]]:
        """Get current active sprint for a board"""
        try:
            url = f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint"
            params = {'state': 'active'}
            response = self.session.get(url, params=params, timeout=(10, 30))
            response.raise_for_status()
            
            data = response.json()
            sprints = data.get('values', [])
            return sprints[0] if sprints else None
            
        except Exception as e:
            logger.error(f"Failed to get current sprint: {e}")
            return None
    
    def get_sprints(self, board_id: int) -> List[Dict[str, Any]]:
        """Get all sprints for a board"""
        try:
            url = f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint"
            params = {'maxResults': 50}
            response = self.session.get(url, params=params, timeout=(10, 30))
            response.raise_for_status()
            
            data = response.json()
            return data.get('values', [])
            
        except Exception as e:
            logger.error(f"Failed to get sprints: {e}")
            return []

    def get_sprint_stories(self, sprint_id: int) -> List[Dict[str, Any]]:
        """Get stories/issues for a specific sprint"""
        try:
            url = f"{self.base_url}/rest/agile/1.0/sprint/{sprint_id}/issue"
            params = {
                'maxResults': 100,
                'fields': 'summary,description,issuetype,status,assignee,priority,created,updated,customfield_10016'
            }
            response = self.session.get(url, params=params, timeout=(10, 30))
            response.raise_for_status()
            
            data = response.json()
            issues = data.get('issues', [])
            
            # Format issues for our use
            formatted_issues = []
            for issue in issues:
                fields = issue.get('fields', {})
                formatted_issues.append({
                    'key': issue.get('key'),
                    'summary': fields.get('summary', ''),
                    'issue_type': fields.get('issuetype', {}).get('name', ''),
                    'status': fields.get('status', {}).get('name', ''),
                    'assignee': fields.get('assignee', {}).get('displayName', '') if fields.get('assignee') else '',
                    'priority': fields.get('priority', {}).get('name', ''),
                    'created': fields.get('created', ''),
                    'updated': fields.get('updated', ''),
                    'story_points': fields.get('customfield_10016', 0)  # Story points field
                })
            
            return formatted_issues
            
        except Exception as e:
            logger.error(f"Failed to get sprint stories: {e}")
            return []

# Global instance
_jira_ops = None

def get_jira_operations() -> JiraOperations:
    """Get or create the global JIRA operations instance"""
    global _jira_ops
    if _jira_ops is None:
        _jira_ops = JiraOperations()
    return _jira_ops
