#!/usr/bin/env python3
"""
Action Detector Module
Detects JIRA actions from user queries for the LangChain RAG system
"""

import re
from enum import Enum
from typing import Dict, Any, Tuple

class ActionType(Enum):
    """Types of JIRA actions"""
    QUERY = "query"
    CREATE = "create"
    UPDATE = "update"
    ASSIGN = "assign"
    COMMENT = "comment"

def detect_jira_action(query: str) -> Tuple[ActionType, Dict[str, Any]]:
    """
    Detect JIRA action from user query
    
    Args:
        query: User's query string
        
    Returns:
        Tuple of (action_type, action_params)
    """
    query_lower = query.lower()
    
    # Query patterns (should be checked first to avoid false positives)
    query_patterns = [
        'list', 'show', 'find', 'search', 'get', 'retrieve', 'display', 'what', 'which', 'how many',
        'assigned to', 'created by', 'updated by', 'reported by', 'status is', 'priority is',
        'in sprint', 'in project', 'due date', 'created date', 'updated date'
    ]
    
    if any(pattern in query_lower for pattern in query_patterns):
        return ActionType.QUERY, {"original_text": query}
    
    # Create action patterns (more specific)
    create_patterns = ['create new', 'add new', 'make new', 'new issue', 'new ticket']
    if any(pattern in query_lower for pattern in create_patterns):
        return ActionType.CREATE, {"original_text": query}
    
    # Update action patterns (more specific)
    update_patterns = ['update', 'change', 'modify', 'edit', 'set status', 'change priority']
    if any(pattern in query_lower for pattern in update_patterns):
        return ActionType.UPDATE, {"original_text": query}
    
    # Assign action patterns (more specific - avoid "assigned to" queries)
    assign_patterns = ['assign to', 'give to', 'hand over to', 'reassign to']
    if any(pattern in query_lower for pattern in assign_patterns):
        return ActionType.ASSIGN, {"original_text": query}
    
    # Comment action patterns
    comment_patterns = ['add comment', 'write comment', 'add note', 'add remark']
    if any(pattern in query_lower for pattern in comment_patterns):
        return ActionType.COMMENT, {"original_text": query}
    
    # Default to query for anything else
    return ActionType.QUERY, {"original_text": query}
