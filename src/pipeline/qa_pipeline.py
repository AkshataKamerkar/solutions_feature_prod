# qa_pipeline.py

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone  # Updated import
from langchain_openai import OpenAIEmbeddings

from src.database.pinecone_client import PineconeClient
from src.agents.answer_agent import AnswerAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.prompts.templates import DIRECT_ANSWER_PROMPT
from src.evaluation.metrics import CBSEEvaluationMetrics
from src.config import config

# New imports for CBSE compliance
from src.agents.answer_refinement_agent import (
    AnswerRefinementAgent,
    create_refinement_context,
    SubjectSection,
    QuestionType
)

logger = logging.getLogger(__name__)


class QAPipeline:
    """Enhanced pipeline for processing educational Q&A with CBSE compliance"""

    def __init__(self):
        """Initialize the enhanced QA pipeline"""
        # Your existing components
        self.pinecone_client = PineconeClient()
        self.answer_agent = AnswerAgent()
        self.evaluation_agent = EvaluationAgent()
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )

        # New CBSE refinement agent
        self.refinement_agent = AnswerRefinementAgent()

        # Initialize embeddings for direct Pinecone access
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL
        )

        # Initialize direct Pinecone client (new syntax)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self._direct_index = None

        # Subject mapping for standardization
        self.subject_mapping = {
            'Physics': 'Science',
            'Chemistry': 'Science',
            'Biology': 'Science',
            'Science': 'Science',
            'Mathematics': 'Mathematics',
            'Math': 'Mathematics',
            'History': 'Social Science',
            'Geography': 'Social Science',
            'Political Science': 'Social Science',
            'Economics': 'Social Science',
            'Civics': 'Social Science',
            'Social Science': 'Social Science',
            'English': 'English',
            'Hindi': 'Hindi'
        }

    def _get_direct_pinecone_index(self):
        """Get direct access to Pinecone index"""
        if self._direct_index is None:
            self._direct_index = self.pc.Index(config.PINECONE_INDEX_NAME)
        return self._direct_index

    def process_question(
            self,
            question: str,
            marks: int,
            subject_filter: Optional[str] = None,
            board_filter: Optional[str] = None,
            include_cbse_refinement: bool = True,
            question_section: str = "general"
    ) -> Dict[str, Any]:
        """
        Enhanced process_question with CBSE compliance features

        Args:
            question: The user's question
            marks: Marks allocated for the question (1-5 for CBSE)
            subject_filter: Optional subject filter
            board_filter: Optional board filter
            include_cbse_refinement: Whether to include CBSE refinement
            question_section: Section type (for English/Hindi)

        Returns:
            Dictionary containing all answers, refinement, and metadata
        """
        try:
            # Validate marks for CBSE
            if board_filter == "CBSE" and marks not in range(1, 6):
                return {
                    'success': False,
                    'error': 'For CBSE board, marks must be between 1 and 5',
                    'answers': {
                        'direct_answer': None,
                        'agent_answer': None,
                        'final_answer': None,
                        'cbse_refined_answer': None
                    }
                }

            logger.info(f"Processing {marks}-mark {board_filter or 'Unknown'} question")

            # Step 1: Get direct LLM answer
            logger.info("Generating direct LLM answer...")
            direct_answer = self._get_direct_answer(question, marks)

            # Step 2: Search for relevant content in Pinecone (using raw results)
            logger.info(f"Searching for relevant content in board: {board_filter}")
            search_results = self._search_relevant_content_raw(
                question,
                board_filter,
                subject_filter
            )

            if not search_results:
                logger.warning("No relevant content found in database")
                return self._create_no_content_response(
                    question, marks, board_filter, direct_answer
                )

            # Step 3: Extract context from search results
            context = self._extract_context(search_results[0], question, marks)

            # Log what we're using
            logger.info(f"Using context: {context['chapter_name']} - {context['concept_title']}")

            # Step 4: Generate agent answer with context
            logger.info("Generating agent answer...")
            agent_answer = self.answer_agent.generate_answer(context)

            # Step 5: CBSE Refinement (New Feature)
            cbse_refined_answer = None
            refinement_info = None

            if include_cbse_refinement and board_filter == "CBSE":
                logger.info("Applying CBSE refinement...")
                cbse_refined_answer, refinement_info = self._apply_cbse_refinement(
                    question, agent_answer, marks, context, question_section
                )

            # Step 6: Evaluate the final answer
            final_answer_for_evaluation = cbse_refined_answer if cbse_refined_answer else agent_answer

            logger.info(f"Evaluating answer using CBSE {marks}-mark criteria...")
            evaluation_context = {
                'question': question,
                'answer': final_answer_for_evaluation,
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

            # Compile comprehensive results
            result = self._compile_comprehensive_result(
                question, marks, context, direct_answer, agent_answer,
                cbse_refined_answer, refinement_info, evaluation_result,
                cbse_metrics, search_results
            )

            logger.info(f"Pipeline completed successfully. CBSE Compliant: {result.get('cbse_compliant', False)}")
            return result

        except Exception as e:
            logger.error(f"Error in QA pipeline: {str(e)}")
            return self._create_error_response(question, marks, board_filter, str(e))

    def _search_relevant_content_raw(
            self,
            question: str,
            board_filter: Optional[str] = None,
            subject_filter: Optional[str] = None,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant content using raw Pinecone results (NO FILTERING)"""

        try:
            logger.info(f"Direct Pinecone search - Board: {board_filter}, Subject: {subject_filter}")

            # Get embeddings for the question
            query_embedding = self.embeddings.embed_query(question)

            # Get direct access to Pinecone index
            index = self._get_direct_pinecone_index()

            # Try multiple search strategies
            search_attempts = [
                # Attempt 1: With both filters
                {
                    'namespace': board_filter,
                    'filter': {'subject': subject_filter} if subject_filter else None,
                    'description': f"with board={board_filter}, subject={subject_filter}"
                },
                # Attempt 2: Board only (no subject filter)
                {
                    'namespace': board_filter,
                    'filter': None,
                    'description': f"with board={board_filter}, no subject filter"
                },
                # Attempt 3: Subject only (no board/namespace)
                {
                    'namespace': None,
                    'filter': {'subject': subject_filter} if subject_filter else None,
                    'description': f"with subject={subject_filter}, no board filter"
                },
                # Attempt 4: No filters at all
                {
                    'namespace': None,
                    'filter': None,
                    'description': "with no filters"
                }
            ]

            raw_results = []

            for attempt in search_attempts:
                try:
                    logger.info(f"Trying search {attempt['description']}")

                    query_params = {
                        'vector': query_embedding,
                        'top_k': top_k,
                        'include_metadata': True
                    }

                    if attempt['namespace']:
                        query_params['namespace'] = attempt['namespace']

                    if attempt['filter']:
                        query_params['filter'] = attempt['filter']

                    query_response = index.query(**query_params)

                    # Convert to our format immediately (NO PROCESSING)
                    for match in query_response.matches:
                        result = {
                            'id': match.id,
                            'score': float(match.score),
                            'metadata': dict(match.metadata) if match.metadata else {}
                        }
                        raw_results.append(result)

                    if raw_results:
                        logger.info(f"SUCCESS: Found {len(raw_results)} results {attempt['description']}")
                        break
                    else:
                        logger.info(f"No results {attempt['description']}")

                except Exception as attempt_error:
                    logger.warning(f"Search attempt failed {attempt['description']}: {str(attempt_error)}")
                    continue

            # Apply minimal similarity threshold only
            if raw_results:
                # Keep results above similarity threshold, but if none meet threshold, keep top 3
                filtered_results = [r for r in raw_results if r['score'] >= config.SIMILARITY_THRESHOLD]

                final_results = filtered_results if filtered_results else raw_results[:3]

                logger.info(f"Returning {len(final_results)} results (threshold: {config.SIMILARITY_THRESHOLD})")

                # Log what we're returning for debugging
                for i, result in enumerate(final_results[:2]):  # Log first 2 results
                    metadata = result.get('metadata', {})
                    logger.info(f"Result {i + 1}: score={result['score']:.3f}, "
                                f"subject={metadata.get('subject', 'N/A')}, "
                                f"chapter={metadata.get('chapter_name', metadata.get('chapter', 'N/A'))}")

                return final_results

            logger.warning("No results found with any search strategy")
            return []

        except Exception as e:
            logger.error(f"Critical error in raw search: {str(e)}")
            return []

    def _apply_cbse_refinement(self, question: str, answer: str, marks: int,
                               context: Dict[str, Any], question_section: str) -> tuple:
        """Apply CBSE refinement to the answer"""

        try:
            # Map subject to standardized format
            standardized_subject = self.subject_mapping.get(
                context['subject'], context['subject']
            )

            # Create refinement context
            refinement_context = create_refinement_context(
                question=question,
                answer=answer,
                marks=marks,
                subject=standardized_subject,
                board=context['board'],
                chapter_name=context['chapter_name'],
                concept_title=context['concept_title'],
                keywords=context.get('keywords', '').split(',') if context.get('keywords') else [],
                subject_section=question_section
            )

            # Execute refinement
            refined_result = self.refinement_agent.refine_answer(refinement_context)

            if refined_result:
                logger.info(
                    f"CBSE Refinement: {refined_result.original_word_count} → {refined_result.refined_word_count} words")
            else:
                logger.warning("CBSE Refinement failed: No refined result returned.")
            return refined_result.refined_answer, refined_result

        except Exception as e:
            logger.error(f"Error in CBSE refinement: {str(e)}")
            return None, None

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
        """Legacy search method - now calls the raw search method"""
        return self._search_relevant_content_raw(question, board_filter, subject_filter)

    def _extract_context(
            self,
            search_result: Dict[str, Any],
            question: str,
            marks: int
    ) -> Dict[str, Any]:
        """Extract context from search result - Updated for your Pinecone structure"""
        metadata = search_result.get('metadata', {})

        # Handle both old and new metadata structures
        chapter_name = metadata.get('chapter_name', metadata.get('chapter', 'Unknown'))
        concept_title = metadata.get('concept_title', metadata.get('section_type', 'Unknown'))
        summary_text = metadata.get('summary_text', metadata.get('summary', ''))
        keywords = metadata.get('keywords', '')

        logger.info(f"Extracting context from: {concept_title}")

        return {
            'question': question,
            'chapter_name': chapter_name,
            'subject': metadata.get('subject', 'Unknown'),
            'board': metadata.get('board', 'Unknown'),
            'summary_text': summary_text,
            'concept_title': concept_title,
            'keywords': keywords,
            'marks': marks
        }

    def _compile_comprehensive_result(self, question: str, marks: int, context: Dict[str, Any],
                                      direct_answer: str, agent_answer: str, cbse_refined_answer: str,
                                      refinement_info: Any, evaluation_result: Dict[str, Any],
                                      cbse_metrics: Any, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile comprehensive result with all processing information"""

        # Determine the final answer
        final_answer = cbse_refined_answer if cbse_refined_answer else agent_answer

        # Check CBSE compliance
        cbse_compliant = False
        if refinement_info:
            cbse_compliant = refinement_info.meets_cbse_standards

        return {
            'success': True,
            'question': question,
            'marks': marks,
            'board': context['board'],
            'subject': context['subject'],
            'chapter': context['chapter_name'],

            # All answer variations
            'answers': {
                'direct_answer': direct_answer,
                'agent_answer': agent_answer,
                'cbse_refined_answer': cbse_refined_answer,
                'final_answer': final_answer
            },

            # CBSE Compliance Information
            'cbse_compliant': cbse_compliant,
            'ready_for_exam': cbse_compliant and evaluation_result.get('score', 0) >= (marks * 0.7),

            # Context Information
            'context': {
                'chapter': context['chapter_name'],
                'subject': context['subject'],
                'board': context['board'],
                'concept_title': context['concept_title'],
                'summary_text': context['summary_text'],
                'keywords': context['keywords'],
                'similarity_score': search_results[0].get('score', 0) if search_results else 0
            },

            # Refinement Information
            'refinement_info': {
                'applied': cbse_refined_answer is not None,
                'original_word_count': refinement_info.original_word_count if refinement_info else len(
                    agent_answer.split()),
                'refined_word_count': refinement_info.refined_word_count if refinement_info else len(
                    agent_answer.split()),
                'strategy_used': refinement_info.refinement_strategy if refinement_info else 'none',
                'quality_score': refinement_info.quality_score if refinement_info else 0.0,
                'compliance_notes': refinement_info.cbse_compliance_notes if refinement_info else 'Not applied'
            } if refinement_info else None,

            # Evaluation Results
            'evaluation': {
                'total_score': evaluation_result.get('score', 0),
                'max_possible': marks,
                'percentage': (evaluation_result.get('score', 0) / marks * 100) if marks > 0 else 0,
                'criterion_scores': evaluation_result.get('criterion_scores', {}),
                'strengths': evaluation_result.get('strengths', []),
                'improvements': evaluation_result.get('improvements', []),
                'detailed_feedback': evaluation_result.get('detailed_feedback', ''),
                'cbse_metrics': getattr(cbse_metrics, '__dict__', cbse_metrics) if cbse_metrics else None

            },

            # Processing Metadata
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0_enhanced',
                'pinecone_results_count': len(search_results),
                'cbse_refinement_applied': cbse_refined_answer is not None,
                'processing_steps': [
                    'direct_answer_generated',
                    'pinecone_search_completed',
                    'agent_answer_generated',
                    'cbse_refinement_applied' if cbse_refined_answer else 'cbse_refinement_skipped',
                    'evaluation_completed'
                ]
            }
        }

    def _create_no_content_response(self, question: str, marks: int,
                                    board_filter: str, direct_answer: str) -> Dict[str, Any]:
        """Create response when no content is found"""

        return {
            'success': False,
            'question': question,
            'marks': marks,
            'board': board_filter or 'Unknown',
            'error': 'No relevant content found in knowledge base',
            'answers': {
                'direct_answer': direct_answer,
                'agent_answer': None,
                'cbse_refined_answer': None,
                'final_answer': direct_answer
            },
            'cbse_compliant': False,
            'suggestions': [
                'Try rephrasing the question with different keywords',
                'Check if the topic is covered in the syllabus',
                'Verify the subject and board filters',
                'Ensure correct spelling of key terms'
            ],
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0_enhanced',
                'no_content_found': True
            }
        }

    def _create_error_response(self, question: str, marks: int,
                               board_filter: str, error: str) -> Dict[str, Any]:
        """Create response for processing errors"""

        return {
            'success': False,
            'question': question,
            'marks': marks,
            'board': board_filter or 'Unknown',
            'error': f'Processing error: {error}',
            'answers': {
                'direct_answer': None,
                'agent_answer': None,
                'cbse_refined_answer': None,
                'final_answer': None
            },
            'cbse_compliant': False,
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0_enhanced',
                'error_occurred': True,
                'error_details': error
            }
        }

    def batch_process_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple questions in batch"""

        results = []

        for i, question_data in enumerate(questions):
            try:
                logger.info(f"Processing batch question {i + 1}/{len(questions)}")

                result = self.process_question(
                    question=question_data.get('question', ''),
                    marks=question_data.get('marks', 1),
                    subject_filter=question_data.get('subject', None),
                    board_filter=question_data.get('board', 'CBSE'),
                    include_cbse_refinement=question_data.get('include_refinement', True),
                    question_section=question_data.get('section', 'general')
                )

                # Add batch metadata
                result['batch_info'] = {
                    'batch_index': i,
                    'total_questions': len(questions)
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing batch question {i + 1}: {str(e)}")
                results.append(self._create_error_response(
                    question_data.get('question', ''),
                    question_data.get('marks', 1),
                    question_data.get('board', 'CBSE'),
                    str(e)
                ))

        return results

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration and capabilities"""

        return {
            "pipeline_version": "2.0_enhanced_raw",
            "features": [
                "Direct LLM answers",
                "Context-aware agent answers",
                "CBSE compliance refinement",
                "Comprehensive evaluation",
                "Batch processing",
                "Raw Pinecone search (no filtering)"
            ],
            "supported_boards": ["CBSE", "ICSE", "STATE_BOARD", "IB", "CAMBRIDGE"],
            "supported_subjects": list(self.subject_mapping.keys()),
            "cbse_marks_range": [1, 2, 3, 4, 5],
            "agents": {
                "answer_agent": "initialized",
                "evaluation_agent": "initialized",
                "refinement_agent": "initialized"
            },
            "models": {
                "llm_model": config.LLM_MODEL,
                "embedding_model": config.EMBEDDING_MODEL
            },
            "database": {
                "pinecone_index": config.PINECONE_INDEX_NAME,
                "dimension": config.PINECONE_DIMENSION,
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "search_method": "direct_raw_access"
            }
        }

    def validate_input(self, question: str, marks: int, board: str = "CBSE") -> Dict[str, Any]:
        """Validate input parameters before processing"""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Validate question
        if not question or not question.strip():
            validation_result["valid"] = False
            validation_result["errors"].append("Question cannot be empty")

        # Validate marks
        if not isinstance(marks, int) or marks < 1:
            validation_result["valid"] = False
            validation_result["errors"].append("Marks must be a positive integer")

        if board == "CBSE" and marks not in range(1, 6):
            validation_result["valid"] = False
            validation_result["errors"].append("For CBSE board, marks must be between 1 and 5")

        # Validate board
        supported_boards = ["CBSE", "ICSE", "STATE_BOARD", "IB", "CAMBRIDGE"]
        if board not in supported_boards:
            validation_result["warnings"].append(f"Board '{board}' not in supported list: {supported_boards}")

        return validation_result

    def get_content_statistics(self, board: str = "CBSE") -> Dict[str, Any]:
        """Get statistics about available content in the database"""

        try:
            # Get a sample of data to analyze using raw search
            sample_query = "educational content"
            sample_results = self._search_relevant_content_raw(sample_query, board, None, top_k=100)

            # Analyze the sample
            subjects = set()
            chapters = set()
            section_types = set()

            for result in sample_results:
                metadata = result.get('metadata', {})
                subjects.add(metadata.get('subject', 'Unknown'))

                # Handle both old and new metadata structures
                chapter = metadata.get('chapter_name', metadata.get('chapter', 'Unknown'))
                chapters.add(chapter)

                section_type = metadata.get('concept_title', metadata.get('section_type', 'Unknown'))
                section_types.add(section_type)

            stats = {
                'board': board,
                'sample_size': len(sample_results),
                'unique_subjects': len(subjects),
                'unique_chapters': len(chapters),
                'unique_section_types': len(section_types),
                'subjects_list': sorted(list(subjects)),
                'sample_chapters': sorted(list(chapters))[:20],  # First 20 chapters
                'section_types_list': sorted(list(section_types))[:20],  # First 20 types
                'database_status': 'healthy' if sample_results else 'no_content',
                'last_checked': datetime.now().isoformat(),
                'search_method': 'direct_raw_access'
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting content statistics: {str(e)}")
            return {
                'board': board,
                'error': str(e),
                'database_status': 'error',
                'last_checked': datetime.now().isoformat()
            }

    def search_content(self, query: str, board: str = "CBSE",
                       subject: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Search educational content directly in the database"""

        try:
            # Search content using our raw search method
            results = self._search_relevant_content_raw(query, board, subject, top_k)

            # Format results
            formatted_results = []
            for result in results:
                metadata = result.get('metadata', {})
                formatted_results.append({
                    'id': result.get('id', 'unknown'),
                    'score': result.get('score', 0),
                    'chapter': metadata.get('chapter_name', metadata.get('chapter', 'Unknown')),
                    'subject': metadata.get('subject', 'Unknown'),
                    'section_type': metadata.get('concept_title', metadata.get('section_type', 'Unknown')),
                    'summary': metadata.get('summary_text', metadata.get('summary', ''))[
                               :200] + '...' if metadata.get('summary_text', metadata.get('summary',
                                                                                          '')) else 'No summary available',
                    'keywords': metadata.get('keywords', '')
                })

            return {
                "search_query": query,
                "board": board,
                "subject_filter": subject,
                "results_count": len(formatted_results),
                "results": formatted_results,
                "searched_at": datetime.now().isoformat(),
                "search_method": "direct_raw_access"
            }

        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return {
                "search_query": query,
                "board": board,
                "subject_filter": subject,
                "results_count": 0,
                "results": [],
                "error": str(e),
                "searched_at": datetime.now().isoformat()
            }

    def test_pipeline_health(self) -> Dict[str, Any]:
        """Test the health of all pipeline components"""

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0_enhanced_raw",
            "components": {}
        }

        # Test Pinecone connection
        try:
            test_results = self._search_relevant_content_raw("test", "CBSE", None, top_k=1)
            health_status["components"]["pinecone"] = {
                "status": "healthy",
                "test_results_count": len(test_results),
                "search_method": "direct_raw_access"
            }
        except Exception as e:
            health_status["components"]["pinecone"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        # Test OpenAI connection
        try:
            test_response = self.llm.predict("Test")
            health_status["components"]["openai"] = {
                "status": "healthy",
                "test_response_length": len(test_response)
            }
        except Exception as e:
            health_status["components"]["openai"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        # Test agents
        try:
            health_status["components"]["agents"] = {
                "answer_agent": "initialized" if self.answer_agent else "not_initialized",
                "evaluation_agent": "initialized" if self.evaluation_agent else "not_initialized",
                "refinement_agent": "initialized" if self.refinement_agent else "not_initialized"
            }
        except Exception as e:
            health_status["components"]["agents"] = {
                "status": "error",
                "error": str(e)
            }

        # Overall status
        unhealthy_components = [
            comp for comp, data in health_status["components"].items()
            if isinstance(data, dict) and data.get("status") == "unhealthy"
        ]

        if unhealthy_components:
            health_status["status"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components

        return health_status

# Utility functions for the pipeline
def create_demo_questions() -> List[Dict[str, Any]]:
    """Create demo questions for testing the pipeline"""
    return [
        {
            "question": "State the laws of reflection of light.",
            "marks": 2,
            "board": "CBSE",
            "subject": "Physics",
            "section": "general"
        },
        {
            "question": "What is photosynthesis? Explain the process.",
            "marks": 3,
            "board": "CBSE",
            "subject": "Biology",
            "section": "general"
        },
        {
            "question": "Find the discriminant of 2x² + 3x + 1 = 0",
            "marks": 2,
            "board": "CBSE",
            "subject": "Mathematics",
            "section": "general"
        },
        {
            "question": "Explain the causes of the First World War.",
            "marks": 5,
            "board": "CBSE",
            "subject": "History",
            "section": "general"
        },
        {
            "question": "What is Ohm's law? State its applications.",
            "marks": 3,
            "board": "CBSE",
            "subject": "Physics",
            "section": "general"
        }
    ]

def test_pipeline_with_demo() -> Dict[str, Any]:
    """Test the pipeline with demo questions"""

    try:
        pipeline = QAPipeline()
        demo_questions = create_demo_questions()

        # Test with first demo question
        test_question = demo_questions[0]
        result = pipeline.process_question(
            question=test_question["question"],
            marks=test_question["marks"],
            subject_filter=test_question["subject"],
            board_filter=test_question["board"],
            include_cbse_refinement=True,
            question_section=test_question["section"]
        )

        return {
            "test_status": "success" if result.get('success', False) else "failed",
            "demo_question": test_question,
            "result_preview": {
                "question": result.get('question', ''),
                "success": result.get('success', False),
                "cbse_compliant": result.get('cbse_compliant', False),
                "final_answer_preview": result.get('answers', {}).get('final_answer', '')[:200] + '...' if result.get(
                    'answers', {}).get('final_answer') else '',
                "processing_time": result.get('metadata', {}).get('processing_timestamp', ''),
                "search_results_count": result.get('metadata', {}).get('pinecone_results_count', 0)
            },
            "available_demo_questions": demo_questions,
            "message": "Demo test completed successfully!" if result.get('success', False) else "Demo test failed"
        }

    except Exception as e:
        return {
            "test_status": "error",
            "error": str(e),
            "available_demo_questions": create_demo_questions(),
            "message": f"Demo test failed with error: {str(e)}"
        }


# Export the main class and utility functions
__all__ = [
    'QAPipeline',
    'create_demo_questions',
    'test_pipeline_with_demo'
]

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    print("🚀 Initializing QA Pipeline with Raw Search...")
    pipeline = QAPipeline()

    # Test pipeline health
    print("🔍 Testing pipeline health...")
    health = pipeline.test_pipeline_health()
    print(f"Pipeline Status: {health['status']}")

    if health['status'] == 'healthy':
        print("✅ All components are healthy!")

        # Test raw search directly
        print("\n🔍 Testing raw search functionality...")
        test_search = pipeline._search_relevant_content_raw("What is Ohm's law?", "CBSE", "Physics")
        print(f"Raw search returned {len(test_search)} results")

        if test_search:
            print("✅ Raw search is working!")
            for i, result in enumerate(test_search[:2]):
                metadata = result.get('metadata', {})
                print(f"  Result {i + 1}: score={result['score']:.3f}, "
                      f"subject={metadata.get('subject', 'N/A')}, "
                      f"chapter={metadata.get('chapter_name', metadata.get('chapter', 'N/A'))}")
        else:
            print("❌ Raw search returned no results")

    # Run demo test
    print("\n🎮 Running demo test...")
    demo_result = test_pipeline_with_demo()
    print(f"Demo Status: {demo_result['test_status']}")

    if demo_result['test_status'] == 'success':
        print("✅ Pipeline is working correctly!")
        print(f"Demo Question: {demo_result['demo_question']['question']}")
        print(f"CBSE Compliant: {demo_result['result_preview']['cbse_compliant']}")
        print(f"Search Results Found: {demo_result['result_preview']['search_results_count']}")
    else:
        print("❌ Pipeline test failed!")
        if 'error' in demo_result:
            print(f"Error: {demo_result['error']}")

    # Show available demo questions
    print("\n📚 Available Demo Questions:")
    for i, q in enumerate(create_demo_questions(), 1):
        print(f"{i}. [{q['subject']} - {q['marks']}M] {q['question']}")

    # Test content statistics
    print("\n📊 Getting content statistics...")
    try:
        stats = pipeline.get_content_statistics("CBSE")
        print(f"Database Status: {stats.get('database_status', 'unknown')}")
        print(f"Sample Size: {stats.get('sample_size', 0)}")
        print(f"Unique Subjects: {stats.get('unique_subjects', 0)}")
        print(f"Subjects Available: {stats.get('subjects_list', [])}")
    except Exception as e:
        print(f"❌ Error getting statistics: {str(e)}")