from typing import Dict, Any, List, Optional
import logging
from langchain_community.chat_models import ChatOpenAI
from src.config import config
from src.database.pinecone_client import PineconeClient
from src.agents.answer_agent import AnswerAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.prompts.templates import DIRECT_ANSWER_PROMPT
from src.evaluation.metrics import CBSEEvaluationMetrics

logger = logging.getLogger(__name__)

class QAPipeline:
    """Main pipeline for processing educational Q&A with CBSE evaluation"""
    
    def __init__(self):
        """Initialize the QA pipeline"""
        self.pinecone_client = PineconeClient()
        self.answer_agent = AnswerAgent()
        self.evaluation_agent = EvaluationAgent()
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
    
    def process_question(
        self, 
        question: str, 
        marks: int,
        subject_filter: Optional[str] = None,
        board_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a question through the complete pipeline
        
        Args:
            question: The user's question
            marks: Marks allocated for the question (1-5 for CBSE)
            subject_filter: Optional subject filter
            board_filter: Optional board filter
            
        Returns:
            Dictionary containing all three answers and metadata
        """
        try:
            # Validate marks for CBSE
            if board_filter == "CBSE" and marks not in range(1, 6):
                return {
                    'success': False,
                    'error': 'For CBSE board, marks must be between 1 and 5',
                    'direct_answer': None,
                    'agent_answer': None,
                    'final_answer': None
                }
            
            # Step 1: Get direct LLM answer
            logger.info("Generating direct LLM answer...")
            direct_answer = self._get_direct_answer(question, marks)
            
            # Step 2: Search for relevant content in Pinecone
            logger.info(f"Searching for relevant content in board: {board_filter}")
            search_results = self._search_relevant_content(
                question, 
                board_filter,
                subject_filter
            )
            
            if not search_results:
                logger.warning("No relevant content found in database")
                return {
                    'success': False,
                    'error': 'No relevant content found for the question',
                    'direct_answer': direct_answer,
                    'agent_answer': None,
                    'final_answer': None
                }
            
            # Step 3: Extract context from search results
            context = self._extract_context(search_results[0], question, marks)
            
            # Log what we're using
            logger.info(f"Using context from: {context['chapter_name']} - {context['concept_title']}")
            
            # Step 4: Generate agent answer with context
            logger.info("Generating agent answer...")
            agent_answer = self.answer_agent.generate_answer(context)
            
            # Step 5: Evaluate and improve the answer using CBSE metrics
            logger.info(f"Evaluating answer using CBSE {marks}-mark criteria...")
            evaluation_context = {
                'question': question,
                'answer': agent_answer,
                'marks': marks,
                'board': context['board'],
                'subject': context['subject'],
                'chapter_name': context['chapter_name']
            }
            evaluation_result = self.evaluation_agent.evaluate_answer(evaluation_context)
            
            # Get CBSE evaluation metrics for display
            cbse_metrics = None
            if board_filter == "CBSE":
                cbse_metrics = CBSEEvaluationMetrics.get_evaluation_metrics(marks)
            
            # Compile results
            result = {
                'success': True,
                'question': question,
                'marks': marks,
                'context': {
                    'chapter': context['chapter_name'],
                    'subject': context['subject'],
                    'board': context['board'],
                    'concept_title': context['concept_title'],
                    'summary_text': context['summary_text'],
                    'keywords': context['keywords']
                },
                'answers': {
                    'direct_answer': direct_answer,
                    'agent_answer': agent_answer,
                    'final_answer': evaluation_result['final_answer']
                },
                'evaluation': {
                    'score': evaluation_result['score'],
                    'criterion_scores': evaluation_result.get('criterion_scores', {}),
                    'strengths': evaluation_result['strengths'],
                    'improvements': evaluation_result['improvements'],
                    'detailed_feedback': evaluation_result.get('detailed_feedback', ''),
                    'cbse_metrics': cbse_metrics
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in QA pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'direct_answer': None,
                'agent_answer': None,
                'final_answer': None
            }
    
    def _get_direct_answer(self, question: str, marks: int) -> str:
        """Generate a direct answer without context"""
        try:
            prompt = DIRECT_ANSWER_PROMPT.format(
                question=question,
                marks=marks
            )
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating direct answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _search_relevant_content(
        self, 
        question: str, 
        board_filter: Optional[str] = None,
        subject_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant content in Pinecone"""
        filter_dict = {}
        if subject_filter:
            filter_dict['subject'] = subject_filter
        
        logger.info(f"Searching content - Board: {board_filter}, Subject: {subject_filter}")
        
        results = self.pinecone_client.search_similar_content(
            query=question,
            board=board_filter,
            filter_dict=filter_dict if filter_dict else None
        )
        
        logger.info(f"Search returned {len(results)} results")
        
        return results
    
    def _extract_context(
        self, 
        search_result: Dict[str, Any], 
        question: str, 
        marks: int
    ) -> Dict[str, Any]:
            """Extract context from search result - Updated for new Pinecone structure"""
            metadata = search_result['metadata']
            
            # Log what we're extracting
            logger.info(f"Extracting context from: {metadata.get('concept_title', 'Unknown')}")
            
            # Extract all required fields from Pinecone metadata
            return {
                'question': question,
                'chapter_name': metadata.get('chapter_name', 'Unknown'),
                'subject': metadata.get('subject', 'Unknown'),
                'board': metadata.get('board', 'Unknown'),
                'summary_text': metadata.get('summary_text', ''),
                'concept_title': metadata.get('concept_title', 'Unknown'),
                'keywords': metadata.get('keywords', ''),
                'marks': marks
            }