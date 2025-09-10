#!/usr/bin/env python3
"""
LangChain RAG Chain for JIRA Assistant
End-to-end RAG pipeline that combines retrieval, generation, and JIRA-specific processing
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManagerForChainRun

from langchain_milvus_store import MilvusVectorStore
from langchain_retriever import JiraHybridRetriever, JiraSprintRetriever
from langchain_llm import JiraRAGLLM, get_jira_rag_llm
from jira_operations import get_jira_operations
from action_detector import detect_jira_action, ActionType

logger = logging.getLogger(__name__)

class JiraRAGChain:
    """Complete RAG chain for JIRA data processing"""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        llm: Optional[JiraRAGLLM] = None,
        jira_ops: Optional[Any] = None,
        use_hybrid_retrieval: bool = True,
        use_sprint_retrieval: bool = True,
        max_retrieval_results: int = 20,
        **kwargs
    ):
        """
        Initialize the JIRA RAG chain
        
        Args:
            vector_store: Milvus vector store instance
            llm: JIRA RAG LLM instance
            jira_ops: JIRA operations instance
            use_hybrid_retrieval: Whether to use hybrid retrieval
            use_sprint_retrieval: Whether to use sprint-specific retrieval
            max_retrieval_results: Maximum number of retrieval results
            **kwargs: Additional arguments
        """
        self.vector_store = vector_store
        self.llm = llm or get_jira_rag_llm()
        self.jira_ops = jira_ops or get_jira_operations()
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.use_sprint_retrieval = use_sprint_retrieval
        self.max_retrieval_results = max_retrieval_results
        
        # Initialize retrievers
        self.hybrid_retriever = None
        self.sprint_retriever = None
        
        if self.use_hybrid_retrieval:
            self.hybrid_retriever = JiraHybridRetriever(
                vector_store=self.vector_store,
                jira_ops=self.jira_ops,
                k=min(max_retrieval_results, 10),
                jql_k=min(max_retrieval_results, 20)
            )
        
        if self.use_sprint_retrieval:
            self.sprint_retriever = JiraSprintRetriever(
                vector_store=self.vector_store,
                jira_ops=self.jira_ops,
                k=min(max_retrieval_results, 15)
            )
        
        # Initialize prompt templates
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for different types of queries"""
        
        # Main RAG prompt
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a JIRA project management assistant. You help users understand their projects, tasks, and work items by analyzing JIRA data.

Your responses should be:
- Clear and actionable
- Focused on the specific question asked
- Professional but friendly
- Include relevant details from the JIRA data
- Provide insights and recommendations when appropriate
- Use bullet points and formatting for better readability
- Include issue keys and status information when relevant

Always base your answers on the JIRA data provided in the context."""),
            ("human", """Question: {question}

JIRA Data Context:
{context}

{sprint_context}

{jql_info}

Please provide a comprehensive answer that directly addresses the question using the JIRA data provided.""")
        ])
        
        # Sprint-specific prompt
        self.sprint_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a JIRA sprint management assistant. You help users understand sprint progress, backlog, and sprint-related information.

Your responses should be:
- Focus on sprint-specific information
- Include sprint metrics and progress
- Provide actionable insights about sprint management
- Use sprint terminology appropriately"""),
            ("human", """Sprint Question: {question}

Sprint Context: {sprint_context}

JIRA Data Context:
{context}

Please provide a comprehensive answer about the sprint-related question using the JIRA data provided.""")
        ])
        
        # Action detection prompt
        self.action_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a JIRA action detection assistant. Analyze user queries to determine if they want to perform JIRA actions.

Action types:
- CREATE: Creating new issues
- UPDATE: Updating existing issues
- ASSIGN: Assigning issues to users
- COMMENT: Adding comments to issues
- QUERY: Just asking questions (no action needed)

Respond with the action type and any relevant parameters."""),
            ("human", """User Query: {query}

Determine if this is an action request or just a query. If it's an action, specify the action type and parameters.""")
        ])
    
    def process_query(
        self,
        question: str,
        include_sprint_context: bool = True,
        include_jql_info: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            question: User's question
            include_sprint_context: Whether to include sprint context
            include_jql_info: Whether to include JQL query information
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing answer, sources, context, and metadata
        """
        try:
            logger.info(f"🔍 Processing query: {question}")
            
            # Step 1: Check for action requests
            action_result = self._check_for_actions(question)
            if action_result.get("is_action"):
                return action_result
            
            # Step 2: Determine retrieval strategy
            retrieval_strategy = self._determine_retrieval_strategy(question)
            logger.info(f"📋 Using retrieval strategy: {retrieval_strategy}")
            
            # Step 3: Retrieve relevant documents
            documents = self._retrieve_documents(question, retrieval_strategy)
            logger.info(f"📚 Retrieved {len(documents)} documents")
            
            if not documents:
                return {
                    "answer": "No relevant JIRA data found for your question. Please try rephrasing your query or check if the data has been properly ingested.",
                    "sources": [],
                    "context": [],
                    "success": True,
                    "total_results": 0
                }
            
            # Step 4: Prepare context and metadata
            context_data = self._prepare_context(documents, question)
            
            # Step 5: Get additional context (sprint, JQL info)
            additional_context = self._get_additional_context(question, include_sprint_context, include_jql_info)
            
            # Step 6: Generate response
            answer = self._generate_response(
                question, 
                context_data, 
                additional_context, 
                retrieval_strategy
            )
            
            # Step 7: Prepare final result
            result = {
                "answer": answer,
                "sources": context_data["sources"],
                "context": context_data["context_parts"],
                "success": True,
                "total_results": len(documents),
                "retrieval_strategy": retrieval_strategy,
                **additional_context
            }
            
            logger.info(f"✅ Query processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context": [],
                "success": False,
                "error": str(e)
            }
    
    def _check_for_actions(self, question: str) -> Dict[str, Any]:
        """Check if the query is requesting a JIRA action"""
        try:
            action_type, action_params = detect_jira_action(question)
            
            if action_type != ActionType.QUERY:
                logger.info(f"🎯 Detected action: {action_type.value}")
                
                # Execute the action
                action_response = self._execute_action(action_type, action_params)
                
                if action_response.get("success"):
                    return {
                        "answer": f"✅ {action_response.get('message')}\n\nAction Type: {action_type.value.capitalize()}\nAction Summary: {action_params}",
                        "sources": [f"Action executed: {action_type.value}"],
                        "context": [f"Action parameters: {action_params}"],
                        "action_type": action_type.value,
                        "action_summary": str(action_params),
                        "success": True,
                        "is_action": True
                    }
                else:
                    return {
                        "answer": f"❌ Action execution failed: {action_response.get('message', 'Unknown error')}",
                        "sources": [],
                        "context": [],
                        "success": False,
                        "is_action": True,
                        "error": action_response.get('message', 'Unknown error')
                    }
            
            return {"is_action": False}
            
        except Exception as e:
            logger.warning(f"⚠️ Action detection failed: {e}")
            return {"is_action": False}
    
    def _execute_action(self, action_type: ActionType, action_params: Dict) -> Dict[str, Any]:
        """Execute a JIRA action"""
        try:
            # Filter out non-JIRA parameters
            filtered_params = {k: v for k, v in action_params.items() 
                             if k not in ['original_text'] and v is not None}
            
            if action_type == ActionType.CREATE:
                return self.jira_ops.create_issue(**filtered_params)
            elif action_type == ActionType.UPDATE:
                issue_key = filtered_params.get("issue_key")
                updates = filtered_params.get("updates", {})
                if issue_key and updates:
                    return self.jira_ops.update_issue(issue_key, updates)
                else:
                    return {"success": False, "message": "Missing issue_key or updates"}
            elif action_type == ActionType.ASSIGN:
                return self.jira_ops.assign_issue(**filtered_params)
            elif action_type == ActionType.COMMENT:
                return self.jira_ops.add_comment(**filtered_params)
            else:
                return {"success": False, "message": f"Unsupported action type: {action_type}"}
                
        except Exception as e:
            logger.error(f"❌ Action execution error: {e}")
            return {"success": False, "message": str(e)}
    
    def _determine_retrieval_strategy(self, question: str) -> str:
        """Determine the best retrieval strategy for the question"""
        question_lower = question.lower()
        
        # Sprint-related queries
        if any(keyword in question_lower for keyword in [
            'sprint', 'current sprint', 'sprint backlog', 'sprint progress', 
            'sprint stories', 'sprint metrics', 'sprint completion'
        ]):
            return "sprint"
        
        # General queries that benefit from hybrid retrieval
        if any(keyword in question_lower for keyword in [
            'all', 'show me', 'list', 'find', 'search', 'what are'
        ]):
            return "hybrid"
        
        # Specific queries that can use simple vector search
        return "vector"
    
    def _retrieve_documents(self, question: str, strategy: str) -> List[Document]:
        """Retrieve documents based on the determined strategy"""
        try:
            if strategy == "sprint" and self.sprint_retriever:
                return self.sprint_retriever._get_relevant_documents(question, run_manager=None)
            elif strategy == "hybrid" and self.hybrid_retriever:
                return self.hybrid_retriever._get_relevant_documents(question, run_manager=None)
            else:
                # Fallback to simple vector search
                return self.vector_store.similarity_search(question, k=self.max_retrieval_results)
                
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            return []
    
    def _prepare_context(self, documents: List[Document], question: str) -> Dict[str, Any]:
        """Prepare context from retrieved documents"""
        context_parts = []
        sources = []
        
        for doc in documents:
            metadata = doc.metadata
            content = doc.page_content
            
            # Create context entry
            context_entry = f"""Issue: {metadata.get('issue_key', 'Unknown')}
Type: {metadata.get('issue_type', 'Unknown')}
Content Type: {metadata.get('content_type', 'Unknown')}
Project: {metadata.get('project_key', 'Unknown')}
Status: {metadata.get('status', 'Unknown')}
Priority: {metadata.get('priority', 'Unknown')}
Assignee: {metadata.get('assignee', 'Unknown')}
Reporter: {metadata.get('reporter', 'Unknown')}
Content: {content}"""
            
            context_parts.append(context_entry)
            
            # Create source entry
            source_info = f"{metadata.get('issue_key', 'Unknown')} ({metadata.get('issue_type', 'Unknown')}) - {metadata.get('content_type', 'Unknown')}"
            if metadata.get('project_key'):
                source_info += f" | Project: {metadata['project_key']}"
            if metadata.get('status'):
                source_info += f" | Status: {metadata['status']}"
            if metadata.get('assignee'):
                source_info += f" | Assignee: {metadata['assignee']}"
            
            sources.append(source_info)
        
        return {
            "context_parts": context_parts,
            "sources": sources,
            "full_context": "\n\n---\n\n".join(context_parts)
        }
    
    def _get_additional_context(
        self, 
        question: str, 
        include_sprint_context: bool, 
        include_jql_info: bool
    ) -> Dict[str, Any]:
        """Get additional context like sprint information and JQL queries"""
        additional_context = {}
        
        # Sprint context
        if include_sprint_context:
            try:
                sprint_info = self.jira_ops.get_current_sprint()
                if sprint_info.get("success") and sprint_info.get("current_sprint"):
                    additional_context["sprint_context"] = sprint_info["current_sprint"]
            except Exception as e:
                logger.warning(f"⚠️ Could not get sprint context: {e}")
        
        # JQL query information
        if include_jql_info and self.hybrid_retriever:
            try:
                jql_query = self.hybrid_retriever._generate_jql_query(question)
                if jql_query:
                    additional_context["jql_query"] = jql_query
            except Exception as e:
                logger.warning(f"⚠️ Could not generate JQL query: {e}")
        
        return additional_context
    
    def _generate_response(
        self, 
        question: str, 
        context_data: Dict, 
        additional_context: Dict, 
        strategy: str
    ) -> str:
        """Generate the final response using the LLM"""
        try:
            # Choose appropriate prompt based on strategy
            if strategy == "sprint":
                prompt = self.sprint_prompt
                sprint_context = additional_context.get("sprint_context", {})
                context_str = f"Sprint: {sprint_context.get('name', 'Unknown')} (ID: {sprint_context.get('id', 'Unknown')})"
            else:
                prompt = self.rag_prompt
                context_str = context_data["full_context"]
            
            # Prepare prompt variables
            prompt_vars = {
                "question": question,
                "context": context_str,
                "sprint_context": f"Sprint Context: {additional_context.get('sprint_context', {})}" if additional_context.get('sprint_context') else "",
                "jql_info": f"JQL Query Used: {additional_context.get('jql_query', '')}" if additional_context.get('jql_query') else ""
            }
            
            # Format the prompt
            formatted_prompt = prompt.format(**prompt_vars)
            
            # Generate response using the specialized JIRA LLM
            response = self.llm.generate_jira_response(
                question=question,
                context=context_data["full_context"],
                sources=context_data["sources"],
                sprint_context=additional_context.get("sprint_context"),
                jql_query=additional_context.get("jql_query")
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return f"Error generating response: {str(e)}"

# Factory function for easy instantiation
def create_jira_rag_chain(
    collection_name: str = "jira_data",
    **kwargs
) -> JiraRAGChain:
    """Create a JIRA RAG chain with default configuration"""
    vector_store = MilvusVectorStore(collection_name=collection_name)
    return JiraRAGChain(vector_store=vector_store, **kwargs)

# Example usage
if __name__ == "__main__":
    # Test the RAG chain
    try:
        print("🧪 Testing JIRA RAG Chain...")
        
        # Create RAG chain
        rag_chain = create_jira_rag_chain(collection_name="test_jira_data")
        
        # Test query
        test_question = "What are the high priority stories in progress?"
        result = rag_chain.process_query(test_question)
        
        print(f"\n🔍 Query: {test_question}")
        print(f"📊 Results: {result['total_results']} documents found")
        print(f"📋 Strategy: {result.get('retrieval_strategy', 'unknown')}")
        print(f"💬 Answer: {result['answer']}")
        print(f"📚 Sources: {len(result['sources'])} sources")
        
    except Exception as e:
        print(f"❌ Error testing RAG chain: {e}")
