from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any
import logging
import json
import re
from src.config import config
from src.prompts.templates import get_evaluation_prompt
from src.evaluation.metrics import CBSEEvaluationMetrics

logger = logging.getLogger(__name__)

class EvaluationAgent:
    """Agent responsible for evaluating and improving answers using CBSE metrics"""
    
    def __init__(self):
        """Initialize the evaluation agent"""
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=0.3,  # Lower temperature for more consistent evaluation
            max_tokens=config.MAX_TOKENS
        )
        
        self.system_message = SystemMessage(content="""
        You are a senior CBSE Board examiner with years of experience evaluating 
        student answers. You:
        - Follow CBSE marking schemes strictly
        - Provide fair but rigorous evaluation
        - Give constructive feedback for improvement
        - Create model answers that exemplify board standards
        - Consider the specific mark allocation for each question type
        """)
    
    def evaluate_answer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate and improve an answer using CBSE metrics
        
        Args:
            context: Dictionary containing question, answer, marks, and metadata
            
        Returns:
            Dictionary with evaluation results and improved answer
        """
        try:
            # Get CBSE-specific evaluation prompt
            prompt = get_evaluation_prompt(
                question=context['question'],
                answer=context['answer'],
                marks=context['marks'],
                board=context['board'],
                subject=context['subject'],
                chapter_name=context['chapter_name']
            )
            
            # Create message
            human_message = HumanMessage(content=prompt)
            
            # Generate response
            messages = [self.system_message, human_message]
            result = self.llm(messages)
            
            # Parse the result with enhanced parsing
            evaluation_result = self._parse_cbse_evaluation(result.content, context['marks'])
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            raise
    
    def _parse_cbse_evaluation(self, evaluation_text: str, total_marks: int) -> Dict[str, Any]:
        """Parse CBSE evaluation text into structured format"""
        try:
            result = {
                'score': None,
                'criterion_scores': {},
                'strengths': [],
                'improvements': [],
                'final_answer': '',
                'detailed_feedback': ''
            }
            
            # Extract total score
            score_pattern = r'Total Score[:\s]+(\d+(?:\.\d+)?)\s*/\s*\d+'
            score_match = re.search(score_pattern, evaluation_text, re.IGNORECASE)
            if score_match:
                result['score'] = float(score_match.group(1))
            else:
                # Try alternate pattern
                score_pattern2 = r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*' + str(total_marks)
                score_match2 = re.search(score_pattern2, evaluation_text)
                if score_match2:
                    result['score'] = float(score_match2.group(1))
            
            # Extract criterion-wise scores
            criterion_pattern = r'Criterion[:\s]+(.+?)[:\s]+(\d+(?:\.\d+)?)\s*(?:out of|/)\s*(\d+(?:\.\d+)?)'
            criterion_matches = re.finditer(criterion_pattern, evaluation_text, re.IGNORECASE)
            for match in criterion_matches:
                criterion_name = match.group(1).strip()
                score = float(match.group(2))
                result['criterion_scores'][criterion_name] = score
            
            # Extract sections
            sections = {
                'strengths': r'Strengths[:\s]*\n((?:[-•]\s*.+\n?)+)',
                'improvements': r'(?:Areas for Improvement|Improvements)[:\s]*\n((?:[-•]\s*.+\n?)+)',
                'model_answer': r'Model Answer[:\s]*\n((?:.|\n)+?)(?=\n\n|\Z)'
            }
            
            for section, pattern in sections.items():
                match = re.search(pattern, evaluation_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if section in ['strengths', 'improvements']:
                        # Extract bullet points
                        items = re.findall(r'[-•]\s*(.+)', match.group(1))
                        result[section] = [item.strip() for item in items]
                    elif section == 'model_answer':
                        result['final_answer'] = match.group(1).strip()
            
            # If model answer not found with above pattern, try to find it
            if not result['final_answer']:
                # Look for content after "Model Answer" or "Final Answer"
                answer_start = evaluation_text.lower().find('model answer')
                if answer_start == -1:
                    answer_start = evaluation_text.lower().find('final answer')
                
                if answer_start != -1:
                    # Find the start of actual answer (after colon or newline)
                    answer_text = evaluation_text[answer_start:]
                    colon_pos = answer_text.find(':')
                    newline_pos = answer_text.find('\n')
                    
                    if colon_pos != -1:
                        start_pos = colon_pos + 1
                    elif newline_pos != -1:
                        start_pos = newline_pos + 1
                    else:
                        start_pos = 0
                    
                    result['final_answer'] = answer_text[start_pos:].strip()
            
            # Store the full evaluation for reference
            result['detailed_feedback'] = evaluation_text
            
            # Validate score
            if result['score'] is None:
                logger.warning("Could not extract score from evaluation")
                # Try to calculate from criterion scores
                if result['criterion_scores']:
                    result['score'] = sum(result['criterion_scores'].values())
                    result['score'] = min(result['score'], total_marks)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing CBSE evaluation: {str(e)}")
            # Return a basic structure with the original text
            return {
                'score': None,
                'criterion_scores': {},
                'strengths': [],
                'improvements': [],
                'final_answer': evaluation_text,
                'detailed_feedback': evaluation_text
            }
    
    def get_marks_distribution(self, marks: int) -> Dict[str, Any]:
        """Get the marks distribution for a specific question type"""
        try:
            metrics = CBSEEvaluationMetrics.get_evaluation_metrics(marks)
            distribution = {}
            
            for criterion in metrics['evaluation_criteria']:
                if not criterion.get('internal_use_only', False):
                    distribution[criterion['criteria_name']] = {
                        'max_marks': criterion['max_marks'],
                        'description': criterion['description']
                    }
            
            return distribution
        except Exception as e:
            logger.error(f"Error getting marks distribution: {str(e)}")
            return {}