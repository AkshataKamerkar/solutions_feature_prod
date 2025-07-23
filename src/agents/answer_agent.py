from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any
import logging
from src.config import config
from src.prompts.templates import AGENT_ANSWER_PROMPT

logger = logging.getLogger(__name__)

class AnswerAgent:
    """Agent responsible for generating educational answers"""
    
    def __init__(self):
        """Initialize the answer agent"""
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        self.system_message = SystemMessage(content="""
        You are an Educational Content Expert with deep knowledge across various subjects and boards. 
        You excel at creating clear, structured answers that help students understand concepts thoroughly.
        You always align your answers with the specific board's curriculum standards and use appropriate terminology.
        """)
    
    def generate_answer(self, context: Dict[str, Any]) -> str:
        """
        Generate an answer based on the provided context
        
        Args:
            context: Dictionary containing question, chapter info, etc.
            
        Returns:
            Generated answer string
        """
        try:
            # Format the prompt
            prompt = AGENT_ANSWER_PROMPT.format(**context)
            
            # Create message
            human_message = HumanMessage(content=prompt)
            
            # Generate response
            messages = [self.system_message, human_message]
            result = self.llm(messages)
            
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise