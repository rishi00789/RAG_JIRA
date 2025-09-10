#!/usr/bin/env python3
"""
LangChain LLM Integration for JIRA RAG
Custom LLM implementation that integrates Hugging Face models with LangChain's LLM interface
"""

import os
import logging
from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
from dotenv import load_dotenv

from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation, ChatResult, ChatGeneration

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class HuggingFaceLLM(LLM):
    """Custom LangChain LLM wrapper for Hugging Face models"""
    
    model_name: str = "openai/gpt-oss-20b"
    base_url: str = "https://router.huggingface.co/v1"
    api_key: Optional[str] = None
    max_tokens: int = 800
    temperature: float = 0.7
    timeout: int = 30
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("Hugging Face API token is required. Set HF_TOKEN environment variable.")
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Hugging Face model"""
        try:
            from openai import OpenAI
            
            # Initialize OpenAI client with Hugging Face inference provider
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            
            # Create completion
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                **kwargs
            )
            
            # Extract the response
            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            else:
                return "No response generated"
                
        except ImportError:
            return "Error: OpenAI client not installed. Run: pip install openai"
        except Exception as e:
            logger.error(f"❌ Hugging Face LLM error: {e}")
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters"""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }

class HuggingFaceChatModel(BaseChatModel):
    """Custom LangChain Chat Model wrapper for Hugging Face models"""
    
    model_name: str = "openai/gpt-oss-20b"
    base_url: str = "https://router.huggingface.co/v1"
    api_key: Optional[str] = None
    max_tokens: int = 800
    temperature: float = 0.7
    timeout: int = 30
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("Hugging Face API token is required. Set HF_TOKEN environment variable.")
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response"""
        try:
            from openai import OpenAI
            
            # Initialize OpenAI client with Hugging Face inference provider
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            
            # Convert LangChain messages to OpenAI format
            openai_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    openai_messages.append({
                        "role": "system",
                        "content": message.content
                    })
                elif isinstance(message, HumanMessage):
                    openai_messages.append({
                        "role": "user",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    openai_messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
            
            # Create completion
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                **kwargs
            )
            
            # Extract the response
            if completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                message = AIMessage(content="No response generated")
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
                
        except ImportError:
            message = AIMessage(content="Error: OpenAI client not installed. Run: pip install openai")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"❌ Hugging Face Chat Model error: {e}")
            message = AIMessage(content=f"Error: {str(e)}")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat response"""
        # For now, just call the sync version
        # In a real implementation, you'd use async HTTP client
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters"""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }

class JiraRAGLLM(HuggingFaceChatModel):
    """Specialized LLM for JIRA RAG with optimized prompts and parameters"""
    
    def __init__(self, **kwargs):
        # Set optimized defaults for JIRA RAG
        default_kwargs = {
            "model_name": "openai/gpt-oss-20b",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 45
        }
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
    
    def generate_jira_response(
        self,
        question: str,
        context: str,
        sources: List[str] = None,
        sprint_context: Dict = None,
        jql_query: str = None
    ) -> str:
        """Generate a specialized JIRA response with enhanced context"""
        try:
            # Build enhanced system prompt
            system_prompt = self._build_system_prompt()
            
            # Build enhanced user prompt
            user_prompt = self._build_user_prompt(
                question, context, sources, sprint_context, jql_query
            )
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Generate response
            result = self._generate(messages)
            
            if result.generations and len(result.generations) > 0:
                return result.generations[0].message.content
            else:
                return "No response generated"
                
        except Exception as e:
            logger.error(f"❌ Error generating JIRA response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for JIRA RAG"""
        return """You are a JIRA project management assistant. You help users understand their projects, tasks, and work items by analyzing JIRA data.

Your responses should be:
- Clear and actionable
- Focused on the specific question asked
- Professional but friendly
- Include relevant details from the JIRA data
- Provide insights and recommendations when appropriate
- Use bullet points and formatting for better readability
- Include issue keys and status information when relevant

Always base your answers on the JIRA data provided in the context. If you don't have enough information to answer a question, say so and suggest what additional information might be helpful."""
    
    def _build_user_prompt(
        self,
        question: str,
        context: str,
        sources: List[str] = None,
        sprint_context: Dict = None,
        jql_query: str = None
    ) -> str:
        """Build the user prompt with enhanced context"""
        prompt_parts = [
            f"Question: {question}",
            "",
            "JIRA Data Context:",
            context
        ]
        
        if sources:
            prompt_parts.extend([
                "",
                f"Available Sources: {', '.join(sources)}"
            ])
        
        if sprint_context and sprint_context.get("current_sprint"):
            current_sprint = sprint_context["current_sprint"]
            prompt_parts.extend([
                "",
                "Sprint Context:",
                f"- Current Sprint: {current_sprint.get('name', 'Unknown')}",
                f"- Sprint ID: {current_sprint.get('id', 'Unknown')}",
                f"- Start Date: {current_sprint.get('startDate', 'Unknown')}",
                f"- End Date: {current_sprint.get('endDate', 'Unknown')}"
            ])
        
        if jql_query:
            prompt_parts.extend([
                "",
                f"JQL Query Used: {jql_query}"
            ])
        
        prompt_parts.extend([
            "",
            "Please provide a comprehensive answer that directly addresses the question using the JIRA data provided."
        ])
        
        return "\n".join(prompt_parts)

# Factory functions for easy instantiation
def get_huggingface_llm(**kwargs) -> HuggingFaceLLM:
    """Get a Hugging Face LLM instance"""
    return HuggingFaceLLM(**kwargs)

def get_huggingface_chat_model(**kwargs) -> HuggingFaceChatModel:
    """Get a Hugging Face Chat Model instance"""
    return HuggingFaceChatModel(**kwargs)

def get_jira_rag_llm(**kwargs) -> JiraRAGLLM:
    """Get a specialized JIRA RAG LLM instance"""
    return JiraRAGLLM(**kwargs)

# Example usage
if __name__ == "__main__":
    # Test the LLM implementations
    try:
        # Test basic LLM
        print("🧪 Testing Hugging Face LLM...")
        llm = get_huggingface_llm()
        response = llm("What is the capital of France?")
        print(f"LLM Response: {response}")
        
        # Test chat model
        print("\n🧪 Testing Hugging Face Chat Model...")
        chat_model = get_huggingface_chat_model()
        messages = [
            HumanMessage(content="Hello, how are you?")
        ]
        chat_result = chat_model._generate(messages)
        print(f"Chat Response: {chat_result.generations[0].message.content}")
        
        # Test JIRA RAG LLM
        print("\n🧪 Testing JIRA RAG LLM...")
        jira_llm = get_jira_rag_llm()
        
        sample_context = """
        Issue: TEST-1
        Type: Story
        Project: TEST
        Status: In Progress
        Priority: High
        Assignee: John Doe
        Content: Implement user authentication system with secure login functionality.
        """
        
        sample_sources = ["TEST-1 (Story) - summary", "TEST-1 (Story) - description"]
        
        jira_response = jira_llm.generate_jira_response(
            question="What is the status of the authentication implementation?",
            context=sample_context,
            sources=sample_sources
        )
        print(f"JIRA RAG Response: {jira_response}")
        
    except Exception as e:
        print(f"❌ Error testing LLM implementations: {e}")
        print("Make sure to set HF_TOKEN environment variable with your Hugging Face API token")
