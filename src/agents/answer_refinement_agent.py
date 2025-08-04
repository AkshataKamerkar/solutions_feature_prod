# answer_refinement_agent.py

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import re
from dataclasses import dataclass
from enum import Enum
import json
from src.config import config
from src.evaluation.metrics import CBSEEvaluationMetrics

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions in CBSE"""
    MCQ = "mcq"
    SUBJECTIVE = "subjective"
    FILL_BLANK = "fill_blank"
    TRUE_FALSE = "true_false"
    ERROR_CORRECTION = "error_correction"
    REPORTING = "reporting"
    SUB_QUESTION = "sub_question"


class SubjectSection(Enum):
    """Sections within subjects"""
    # English sections
    LITERATURE = "literature"
    GRAMMAR = "grammar"
    WRITING = "writing"
    READING = "reading"

    # Hindi sections
    SAHITYA = "साहित्य"
    VYAKARAN = "व्याकरण"
    RACHNATMAK_LEKHAN = "रचनात्मक लेखन"
    APATHIT_BODH = "अपठित बोध"

    # General sections
    GENERAL = "general"


@dataclass
class CBSEWordLimits:
    """CBSE-specific word limits"""
    min_words: int
    max_words: int
    target_words: int
    description: str
    is_exact: bool = False  # For MCQs, fill-blanks etc.


@dataclass
class RefinementContext:
    """Enhanced context for CBSE answer refinement"""
    original_answer: str
    question: str
    marks: int
    subject: str
    board: str
    chapter_name: str
    concept_title: str
    keywords: List[str]
    question_type: QuestionType
    subject_section: SubjectSection
    sub_question_marks: Optional[List[int]] = None  # For 4-mark questions


@dataclass
class RefinementResult:
    """Result of CBSE answer refinement"""
    refined_answer: str
    original_word_count: int
    refined_word_count: int
    meets_cbse_standards: bool
    word_limit_info: CBSEWordLimits
    quality_score: float
    preserved_elements: List[str]
    refinement_strategy: str
    cbse_compliance_notes: str


class AnswerRefinementAgent:
    """
    CBSE-Compliant Answer Refinement Agent

    Transforms answers to meet exact CBSE word count requirements
    while preserving educational value and board-specific formatting.
    """

    # CBSE Word Count Standards - Exact as per official guidelines
    CBSE_WORD_LIMITS = {
        # Science & Social Science
        "Science": {
            1: {QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True)},
            2: {QuestionType.SUBJECTIVE: CBSEWordLimits(30, 50, 40, "30-50 words")},
            3: {QuestionType.SUBJECTIVE: CBSEWordLimits(50, 80, 65, "50-80 words")},
            4: {
                QuestionType.SUB_QUESTION: {
                    1: CBSEWordLimits(20, 30, 25, "1-mark sub: 20-30 words"),
                    2: CBSEWordLimits(30, 50, 40, "2-mark sub: 30-50 words")
                }
            },
            5: {QuestionType.SUBJECTIVE: CBSEWordLimits(80, 120, 100, "80-120 words")}
        },

        "Social Science": {
            1: {QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True)},
            2: {QuestionType.SUBJECTIVE: CBSEWordLimits(30, 50, 40, "30-50 words")},
            3: {QuestionType.SUBJECTIVE: CBSEWordLimits(50, 80, 65, "50-80 words")},
            4: {
                QuestionType.SUB_QUESTION: {
                    1: CBSEWordLimits(20, 30, 25, "1-mark sub: 20-30 words"),
                    2: CBSEWordLimits(30, 50, 40, "2-mark sub: 30-50 words")
                }
            },
            5: {QuestionType.SUBJECTIVE: CBSEWordLimits(80, 120, 100, "80-120 words")}
        },

        # Mathematics - Focus on steps, not word count
        "Mathematics": {
            1: {QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True)},
            2: {QuestionType.SUBJECTIVE: CBSEWordLimits(0, 999, 0, "Show required steps")},
            3: {QuestionType.SUBJECTIVE: CBSEWordLimits(0, 999, 0, "Show required steps")},
            4: {QuestionType.SUBJECTIVE: CBSEWordLimits(0, 999, 0, "Show required steps")},
            5: {QuestionType.SUBJECTIVE: CBSEWordLimits(0, 999, 0, "Show required steps")}
        },

        # English with sections
        "English": {
            (SubjectSection.LITERATURE, 3): CBSEWordLimits(40, 50, 45, "Literature 3-marks: 40-50 words"),
            (SubjectSection.LITERATURE, 5): {
                QuestionType.FILL_BLANK: CBSEWordLimits(1, 2, 1, "1-2 words", True),
                QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True),
                QuestionType.TRUE_FALSE: CBSEWordLimits(1, 1, 1, "True/False only", True),
                (QuestionType.SUBJECTIVE, 1): CBSEWordLimits(15, 25, 20, "1-mark subjective: 20 words"),
                (QuestionType.SUBJECTIVE, 2): CBSEWordLimits(35, 45, 40, "2-mark subjective: 40 words")
            },
            (SubjectSection.LITERATURE, 6): CBSEWordLimits(100, 120, 110, "6-marks: 100-120 words"),
            (SubjectSection.GRAMMAR, 1): {
                QuestionType.FILL_BLANK: CBSEWordLimits(1, 1, 1, "1 word", True),
                QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True),
                QuestionType.ERROR_CORRECTION: CBSEWordLimits(2, 2, 2, "2 words", True),
                QuestionType.REPORTING: CBSEWordLimits(0, 999, 0, "As required by question")
            },
            (SubjectSection.WRITING, 5): CBSEWordLimits(115, 125, 120, "Writing 5-marks: 120 words"),
            (SubjectSection.READING, 10): {
                QuestionType.FILL_BLANK: CBSEWordLimits(1, 2, 1, "1-2 words", True),
                QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True),
                (QuestionType.SUBJECTIVE, 1): CBSEWordLimits(10, 20, 15, "1-mark subjective: 10-20 words"),
                (QuestionType.SUBJECTIVE, 2): CBSEWordLimits(10, 30, 20, "2-mark subjective: 10-30 words")
            }
        },

        # Hindi with sections
        "Hindi": {
            (SubjectSection.SAHITYA, 1): CBSEWordLimits(40, 50, 45, "साहित्य 1-mark: 40-50 words"),
            (SubjectSection.SAHITYA, 2): CBSEWordLimits(25, 30, 27, "साहित्य 2-marks: 25-30 words"),
            (SubjectSection.SAHITYA, 4): CBSEWordLimits(50, 60, 55, "साहित्य 4-marks: 50-60 words"),
            (SubjectSection.SAHITYA, 5): {QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True)},
            (SubjectSection.VYAKARAN, 1): {
                QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True),
                QuestionType.SUBJECTIVE: CBSEWordLimits(0, 999, 0, "As required by question")
            },
            (SubjectSection.RACHNATMAK_LEKHAN, 4): CBSEWordLimits(35, 45, 40, "रचनात्मक लेखन 4-marks: 40 words"),
            (SubjectSection.RACHNATMAK_LEKHAN, 5): CBSEWordLimits(80, 100, 90, "रचनात्मक लेखन 5-marks: 80-100 words"),
            (SubjectSection.RACHNATMAK_LEKHAN, 6): CBSEWordLimits(115, 125, 120, "रचनात्मक लेखन 6-marks: 120 words"),
            (SubjectSection.APATHIT_BODH, 7): {
                QuestionType.MCQ: CBSEWordLimits(0, 0, 0, "Only correct option", True),
                (QuestionType.SUBJECTIVE, 2): CBSEWordLimits(20, 30, 25, "2-mark subjective: 20-30 words")
            }
        }
    }

    def __init__(self):
        """Initialize the CBSE refinement agent"""
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=0.15,  # Very low for consistent refinement
            max_tokens=1500,  # Sufficient for refinement tasks
            request_timeout=45
        )

        self.system_message = SystemMessage(content=self._get_cbse_system_prompt())

        # Initialize quality checkers
        self._setup_quality_checkers()

    def _get_cbse_system_prompt(self) -> str:
        """Get the CBSE-specific system prompt"""
        return """
        You are a CBSE Board Expert Answer Refinement Specialist with 15+ years of experience.

        🎯 YOUR MISSION:
        Transform student answers to meet EXACT CBSE word count requirements while maintaining:
        • Generating an ACCURATE and RELEVANT answer to the given Question, STRICTLY FOLLOWING THE CBSE WORD COUNT RULES 
        • Complete accuracy of facts and formulas
        • Logical flow and structure
        • All essential educational content
        • Proper CBSE answer format

        📏 CBSE WORD COUNT RULES (STRICTLY FOLLOW):
        • Science/Social Science: 2M(30-50), 3M(50-80), 4M(sub-parts), 5M(80-120)
        • Mathematics: Focus on STEPS, not word count
        • English: Section-specific limits (Literature, Grammar, Writing, Reading)
        • Hindi: साहित्य, व्याकरण, रचनात्मक लेखन specific limits
        • MCQs: ONLY the correct option, NO explanation

        🔧 REFINEMENT STRATEGIES:
        1. COMPRESSION: Remove redundant phrases, combine sentences
        2. PRECISION: Use exact terminology, eliminate vague words
        3. STRUCTURE: Maintain Given→Formula→Solution→Answer format
        4. PRESERVATION: Keep all mathematical steps, formulas, units

        ⚠️ NEVER REMOVE:
        • Formulas, equations, calculations
        • Units, final answers
        • Key definitions, theorems
        • Diagram labels, data
        • Logical reasoning steps

        🎨 CBSE FORMATTING:
        • Use "Given:", "To find:", "Formula:", "Solution:", "Therefore:"
        • Underline key terms
        • Box final answers
        • Number all steps clearly
        • Include proper units

        Your refined answer must be examination-ready and scoring maximum marks.
        """

    def _setup_quality_checkers(self):
        """Setup quality checking mechanisms"""
        self.essential_patterns = {
            'formula': r'(?:=|→|∴|∵|\+|\-|\*|\/|\^)',
            'units': r'\b(?:m|cm|mm|km|kg|g|mg|s|min|hr|°C|K|mol|A|V|Ω|J|W|N|Pa|Rs|₹)\b',
            'given_data': r'(?:Given|given)[:;]',
            'final_answer': r'(?:Therefore|Hence|Thus|Answer|∴)',
            'steps': r'(?:Step|step)\s*\d+|(?:First|Second|Third|Finally)',
            'definitions': r'(?:is defined as|means|refers to|definition)'
        }

    def refine_answer(self, context: RefinementContext) -> RefinementResult:
        """
        Main method to refine answer according to CBSE standards

        Args:
            context: Refinement context with all necessary information

        Returns:
            RefinementResult with refined answer and metadata
        """
        try:
            logger.info(f"Starting refinement for {context.subject} {context.marks}-mark question")

            # Step 1: Detect question type and get word limits
            question_type = self._detect_question_type(context.question, context.original_answer)
            word_limits = self._get_word_limits(context.subject, context.marks,
                                                context.subject_section, question_type)

            if not word_limits:
                logger.warning(f"No word limits found for {context.subject} {context.marks}-mark")
                return self._create_fallback_result(context)

                # Step 2: Analyze current answer
                original_word_count = self._count_words(context.original_answer)
                essential_elements = self._extract_essential_elements(context.original_answer)

                # Step 3: Determine refinement strategy
                refinement_strategy = self._determine_strategy(
                    original_word_count, word_limits, context.subject, question_type
                )

                # Step 4: Execute refinement
                refined_answer = self._execute_refinement(
                    context, word_limits, refinement_strategy, essential_elements
                )

                # Step 5: Validate and post-process
                refined_word_count = self._count_words(refined_answer)
                quality_score = self._calculate_quality_score(
                    context.original_answer, refined_answer, essential_elements
                )

                # Step 6: Ensure CBSE compliance
                cbse_compliant_answer = self._ensure_cbse_compliance(
                    refined_answer, context, word_limits
                )

                final_word_count = self._count_words(cbse_compliant_answer)
                meets_standards = self._check_cbse_standards(
                    cbse_compliant_answer, word_limits, context.subject
                )

                # Create result
                result = RefinementResult(
                    refined_answer=cbse_compliant_answer,
                    original_word_count=original_word_count,
                    refined_word_count=final_word_count,
                    meets_cbse_standards=meets_standards,
                    word_limit_info=word_limits,
                    quality_score=quality_score,
                    preserved_elements=essential_elements,
                    refinement_strategy=refinement_strategy,
                    cbse_compliance_notes=self._generate_compliance_notes(
                        word_limits, final_word_count, meets_standards
                    )
                )

                logger.info(f"Refinement complete: {original_word_count}→{final_word_count} words")
                return result

        except Exception as e:
            logger.error(f"Error in answer refinement: {str(e)}")
            return self._create_fallback_result(context, str(e))

    def _detect_question_type(self, question: str, answer: str) -> QuestionType:
        """Detect the type of question based on content"""
        question_lower = question.lower()
        answer_lower = answer.lower()

        # MCQ detection patterns
        mcq_patterns = [
            r'$a$|$b$|$c$|$d$',
            r'choose the correct',
            r'select the right',
            r'which of the following',
            r'the correct option'
        ]

        if any(re.search(pattern, question_lower) for pattern in mcq_patterns):
            return QuestionType.MCQ

        # Fill in the blank detection
        if 'fill in the blank' in question_lower or '______' in question:
            return QuestionType.FILL_BLANK

        # True/False detection
        if 'true or false' in question_lower or answer_lower in ['true', 'false']:
            return QuestionType.TRUE_FALSE

        # Error correction detection
        if 'error' in question_lower and 'correct' in question_lower:
            return QuestionType.ERROR_CORRECTION

        # Reporting questions
        if any(word in question_lower for word in ['report', 'indirect speech', 'narration']):
            return QuestionType.REPORTING

        # Default to subjective
        return QuestionType.SUBJECTIVE

    def _get_word_limits(self, subject: str, marks: int, section: SubjectSection,
                         question_type: QuestionType) -> Optional[CBSEWordLimits]:
        """Get CBSE word limits for specific question parameters"""
        try:
            subject_limits = self.CBSE_WORD_LIMITS.get(subject, {})

            # Handle English and Hindi with sections
            if subject in ["English", "Hindi"]:
                section_key = (section, marks)
                section_limits = subject_limits.get(section_key)

                if isinstance(section_limits, dict):
                    # Multiple question types in this section
                    if question_type in section_limits:
                        return section_limits[question_type]
                    elif (question_type, marks) in section_limits:
                        return section_limits[(question_type, marks)]
                elif isinstance(section_limits, CBSEWordLimits):
                    return section_limits

            # Handle regular subjects (Science, Social Science, Mathematics)
            else:
                mark_limits = subject_limits.get(marks, {})

                if isinstance(mark_limits, dict):
                    if question_type == QuestionType.SUB_QUESTION:
                        # Handle 4-mark sub-questions
                        sub_limits = mark_limits.get(QuestionType.SUB_QUESTION, {})
                        return sub_limits.get(1, None)  # Default to 1-mark sub
                    else:
                        return mark_limits.get(question_type)
                elif isinstance(mark_limits, CBSEWordLimits):
                    return mark_limits

            logger.warning(f"No word limits found for {subject}, {marks} marks, {section}, {question_type}")
            return None

        except Exception as e:
            logger.error(f"Error getting word limits: {str(e)}")
            return None

    def _count_words(self, text: str) -> int:
        """Count words in text, excluding mathematical symbols"""
        if not text or not text.strip():
            return 0

        # Remove mathematical expressions and symbols for word counting
        clean_text = re.sub(r'[=+\-*/÷×∴∵→←↑↓≥≤≠≈∞∑∏∫∆∇]', ' ', text)
        clean_text = re.sub(r'\d+\.?\d*', ' ', clean_text)  # Remove numbers
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # Remove punctuation

        words = clean_text.split()
        return len([word for word in words if len(word) > 1])  # Exclude single characters

    def _extract_essential_elements(self, answer: str) -> List[str]:
        """Extract essential elements that must be preserved"""
        essential_elements = []

        for element_type, pattern in self.essential_patterns.items():
            if re.search(pattern, answer, re.IGNORECASE):
                essential_elements.append(element_type)

        # Check for specific content types
        if re.search(r'(?:diagram|figure|graph)', answer, re.IGNORECASE):
            essential_elements.append('diagram_reference')

        if re.search(r'(?:example|for instance|such as)', answer, re.IGNORECASE):
            essential_elements.append('examples')

        return essential_elements

    def _determine_strategy(self, current_count: int, limits: CBSEWordLimits,
                            subject: str, question_type: QuestionType) -> str:
        """Determine the best refinement strategy"""

        # MCQs and exact answers
        if limits.is_exact or question_type == QuestionType.MCQ:
            return "extract_exact"

        # Mathematics - focus on steps
        if subject == "Mathematics":
            return "optimize_steps"

        # Word count based strategies
        if current_count > limits.max_words:
            excess_ratio = (current_count - limits.max_words) / limits.max_words
            if excess_ratio > 0.5:
                return "heavy_compression"
            else:
                return "moderate_compression"

        elif current_count < limits.min_words:
            deficit_ratio = (limits.min_words - current_count) / limits.min_words
            if deficit_ratio > 0.3:
                return "expand_content"
            else:
                return "minor_expansion"

        else:
            return "optimize_precision"

    def _execute_refinement(self, context: RefinementContext, limits: CBSEWordLimits,
                            strategy: str, essential_elements: List[str]) -> str:
        """Execute the refinement based on strategy"""

        # Create refinement prompt
        refinement_prompt = self._create_refinement_prompt(
            context, limits, strategy, essential_elements
        )

        try:
            # Generate refined answer
            human_message = HumanMessage(content=refinement_prompt)
            messages = [self.system_message, human_message]
            result = self.llm(messages)

            return result.content.strip()

        except Exception as e:
            logger.error(f"Error in LLM refinement: {str(e)}")
            # Fallback to rule-based refinement
            return self._rule_based_refinement(context.original_answer, limits, strategy)

    def _create_refinement_prompt(self, context: RefinementContext, limits: CBSEWordLimits,
                                  strategy: str, essential_elements: List[str]) -> str:
        """Create a detailed refinement prompt"""

        essential_str = ", ".join(essential_elements) if essential_elements else "None identified"

        prompt = f"""
                🎯 CBSE ANSWER REFINEMENT TASK

                ORIGINAL QUESTION: {context.question}
                SUBJECT: {context.subject} | MARKS: {context.marks} | CHAPTER: {context.chapter_name}

                STUDENT'S ORIGINAL ANSWER:
                {context.original_answer}

                📏 CBSE REQUIREMENTS:
                • Word Limit: {limits.min_words}-{limits.max_words} words (Target: {limits.target_words})
                • Description: {limits.description}
                • Current Word Count: {self._count_words(context.original_answer)} words

                🔧 REFINEMENT STRATEGY: {strategy}

                🎯 ESSENTIAL ELEMENTS TO PRESERVE: {essential_str}

                📋 SPECIFIC INSTRUCTIONS:

                {self._get_strategy_instructions(strategy, context.subject, limits)}

                ✅ CBSE FORMAT REQUIREMENTS:
                • Use "Given:", "To find:", "Formula:", "Solution:", "Therefore:"
                • Include all mathematical steps for Science/Maths
                • Underline key terms
                • Box final answer
                • Include proper units
                • Maintain logical flow

                🎯 OUTPUT REQUIREMENTS:
                1. Provide ONLY the refined answer
                2. Ensure exact word count compliance
                3. Maintain educational accuracy
                4. Follow CBSE answer format
                5. Preserve all essential content

                REFINED ANSWER:
                """

        return prompt

    def _get_strategy_instructions(self, strategy: str, subject: str, limits: CBSEWordLimits) -> str:
        """Get specific instructions based on refinement strategy"""

        instructions = {
            "extract_exact": f"""
                    • Extract ONLY the required answer (MCQ option/fill-in-blank word)
                    • NO explanation or justification needed
                    • Word count: Exactly {limits.target_words} words
                    """,

            "heavy_compression": f"""
                    • Remove all redundant phrases and unnecessary words
                    • Combine related sentences
                    • Keep only core concepts, formulas, and final answer
                    • Compress to {limits.min_words}-{limits.max_words} words
                    • Maintain all mathematical steps and formulas
                    """,

            "moderate_compression": f"""
                    • Remove verbose explanations
                    • Use concise terminology
                    • Eliminate repetitive content
                    • Target: {limits.target_words} words
                    • Preserve examples and key points
                    """,

            "optimize_steps": f"""
                    • Focus on showing all mathematical/logical steps clearly
                    • Use bullet points or numbered steps
                    • Include Given→Formula→Solution→Answer format
                    • Word count is secondary to step clarity
                    """,

            "expand_content": f"""
                    • Add relevant examples or applications
                    • Include more detailed explanations
                    • Add proper introduction and conclusion
                    • Target: {limits.target_words} words
                    • Maintain CBSE answer structure
                    """,

            "optimize_precision": f"""
                    • Use precise CBSE terminology
                    • Improve clarity and structure
                    • Fine-tune word choice
                    • Maintain current length around {limits.target_words} words
                    """
        }

        base_instruction = instructions.get(strategy, instructions["optimize_precision"])

        # Add subject-specific instructions
        if subject == "Mathematics":
            base_instruction += "\n• Show every calculation step\n• Include proper mathematical notation"
        elif subject in ["Science", "Social Science"]:
            base_instruction += "\n• Include definitions where relevant\n• Add real-world connections if space allows"

        return base_instruction

    def _rule_based_refinement(self, answer: str, limits: CBSEWordLimits, strategy: str) -> str:
        """Fallback rule-based refinement"""

        if limits.is_exact:
            # Extract only essential part for MCQs etc.
            if "option" in answer.lower():
                match = re.search(r'$([a-d])$', answer)
                return match.group(0) if match else answer.split()[0]
            return answer.split('.')[0] if '.' in answer else answer

        # Basic compression rules
        refined = answer

        # Remove redundant phrases
        redundant_phrases = [
            r'\bas we all know\b', r'\bit is clear that\b', r'\bobviously\b',
            r'\bas mentioned above\b', r'\bfrom the above discussion\b',
            r'\bhence we get\b', r'\btherefore it can be concluded\b'
        ]

        for phrase in redundant_phrases:
            refined = re.sub(phrase, '', refined, flags=re.IGNORECASE)

        # Clean up extra spaces and punctuation
        refined = re.sub(r'\s+', ' ', refined).strip()

        return refined

    def _ensure_cbse_compliance(self, answer: str, context: RefinementContext,
                                limits: CBSEWordLimits) -> str:
        """Ensure final answer meets CBSE compliance standards"""

        compliant_answer = answer

        # Add CBSE formatting if missing
        if context.subject == "Mathematics" and "Given:" not in answer:
            if re.search(r'\d+', context.question):  # Numerical question
                compliant_answer = f"Given: [From question]\nSolution:\n{answer}"

        # Ensure proper conclusion
        if not re.search(r'(?:Therefore|Hence|Thus|∴)', compliant_answer, re.IGNORECASE):
            if not limits.is_exact:
                compliant_answer += "\n∴ " + self._extract_final_answer(compliant_answer)

        # Check word count and adjust if necessary
        current_count = self._count_words(compliant_answer)

        if not limits.is_exact and current_count > limits.max_words:
            # Emergency compression
            sentences = compliant_answer.split('.')
            essential_sentences = [s for s in sentences if any(
                keyword in s.lower() for keyword in ['given', 'formula', 'therefore', '=']
            )]
            compliant_answer = '. '.join(essential_sentences[:3]) + '.'

        return compliant_answer.strip()

    def _extract_final_answer(self, answer: str) -> str:
        """Extract the final answer from the solution"""

        # Look for final numerical value with units
        number_with_unit = re.findall(r'\d+(?:\.\d+)?\s*(?:m|cm|mm|km|kg|g|mg|s|min|hr|°C|K|mol|A|V|Ω|J|W|N|Pa|Rs|₹)',
                                      answer)
        if number_with_unit:
            return f"Final answer = {number_with_unit[-1]}"

        # Look for final numerical value
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        if numbers:
            return f"Final answer = {numbers[-1]}"

        # Look for conclusion statements
        conclusion_patterns = [
            r'(?:therefore|hence|thus|∴)\s*(.+?)(?:\.|$)',
            r'(?:answer|result)\s*[:=]\s*(.+?)(?:\.|$)',
            r'(?:final answer|conclusion)\s*[:=]\s*(.+?)(?:\.|$)'
        ]

        for pattern in conclusion_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return last sentence
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        return sentences[-1] if sentences else "Answer completed"

    def _calculate_quality_score(self, original: str, refined: str,
                                 essential_elements: List[str]) -> float:
        """Calculate quality score of refinement (0-1)"""

        score = 0.0
        max_score = 5.0  # Total possible score

        # 1. Essential elements preservation (2 points)
        preserved_count = 0
        for element in essential_elements:
            pattern = self.essential_patterns.get(element, element)
            if re.search(pattern, refined, re.IGNORECASE):
                preserved_count += 1

        if essential_elements:
            score += 2.0 * (preserved_count / len(essential_elements))
        else:
            score += 2.0  # No essential elements to preserve

        # 2. Clarity improvement (1 point)
        # Check for clear structure
        structure_indicators = ['given:', 'formula:', 'solution:', 'therefore:']
        structure_score = sum(1 for indicator in structure_indicators
                              if indicator in refined.lower()) / len(structure_indicators)
        score += 1.0 * structure_score

        # 3. Conciseness (1 point)
        # Reward appropriate length reduction without losing content
        original_words = self._count_words(original)
        refined_words = self._count_words(refined)

        if original_words > 0:
            compression_ratio = refined_words / original_words
            if 0.5 <= compression_ratio <= 0.9:  # Good compression
                score += 1.0
            elif 0.3 <= compression_ratio < 0.5:  # Heavy but acceptable
                score += 0.7
            elif compression_ratio > 0.9:  # Minimal compression
                score += 0.5
            else:  # Over-compression
                score += 0.2
        else:
            score += 0.5

        # 4. CBSE formatting (1 point)
        cbse_format_indicators = [
            r'given\s*:', r'to find\s*:', r'formula\s*:', r'solution\s*:',
            r'therefore\s*:', r'∴', r'answer\s*[:=]'
        ]
        format_score = sum(1 for indicator in cbse_format_indicators
                           if re.search(indicator, refined, re.IGNORECASE))
        score += min(1.0, format_score / 3)  # Max 1 point for good formatting

        return min(1.0, score / max_score)

    def _check_cbse_standards(self, answer: str, limits: CBSEWordLimits, subject: str) -> bool:
        """Check if answer meets CBSE standards"""

        word_count = self._count_words(answer)

        # Check word count compliance
        if limits.is_exact:
            # For MCQs, fill-blanks etc., check exact requirements
            if limits.max_words == 0:  # MCQ - should be just option
                return re.match(r'^[a-d]$|^$[a-d]$$', answer.strip()) is not None
            elif limits.max_words <= 2:  # Fill blanks, True/False
                return word_count <= limits.max_words
        else:
            # For subjective questions, check range
            if not (limits.min_words <= word_count <= limits.max_words):
                return False

        # Subject-specific checks
        if subject == "Mathematics":
            # Must show steps for non-MCQ questions
            if not limits.is_exact:
                has_steps = any(indicator in answer.lower()
                                for indicator in ['step', 'given', 'formula', 'solution'])
                if not has_steps:
                    return False

        elif subject in ["Science", "Social Science"]:
            # Must have proper structure for longer answers
            if word_count > 50:
                has_structure = any(indicator in answer.lower()
                                    for indicator in ['given', 'definition', 'formula', 'therefore'])
                if not has_structure:
                    return False

        # Check for essential CBSE elements
        essential_checks = [
            ('has_conclusion', r'(?:therefore|hence|thus|∴|answer)'),
            ('proper_units', r'\b(?:m|cm|kg|g|s|°C|K|Rs|₹)\b') if subject == "Science" else (None, None),
            ('mathematical_notation', r'[=+\-*/]') if subject in ["Mathematics", "Science"] else (None, None)
        ]

        for check_name, pattern in essential_checks:
            if check_name and pattern and word_count > 20:  # Only for substantial answers
                if subject == "Science" and check_name == 'proper_units':
                    # Units required for numerical problems
                    if re.search(r'\d+', answer) and not re.search(pattern, answer):
                        return False
                elif subject in ["Mathematics", "Science"] and check_name == 'mathematical_notation':
                    # Math notation required for calculation problems
                    if 'calculate' in answer.lower() and not re.search(pattern, answer):
                        return False

        return True

    def _generate_compliance_notes(self, limits: CBSEWordLimits, word_count: int,
                                   meets_standards: bool) -> str:
        """Generate compliance notes for the refinement"""

        notes = []

        # Word count compliance
        if limits.is_exact:
            if meets_standards:
                notes.append(f"✅ Exact format requirements met: {limits.description}")
            else:
                notes.append(f"❌ Does not meet exact format: {limits.description}")
        else:
            if limits.min_words <= word_count <= limits.max_words:
                notes.append(f"✅ Word count compliant: {word_count} words ({limits.description})")
            elif word_count < limits.min_words:
                notes.append(f"⚠️ Below minimum: {word_count}/{limits.min_words} words")
            else:
                notes.append(f"⚠️ Exceeds maximum: {word_count}/{limits.max_words} words")

        # Overall compliance
        if meets_standards:
            notes.append("✅ Meets CBSE board standards")
            notes.append("✅ Ready for board examination")
        else:
            notes.append("❌ Requires further refinement")
            notes.append("💡 Review CBSE answer format requirements")

        return " | ".join(notes)

    def _create_fallback_result(self, context: RefinementContext, error: str = None) -> RefinementResult:
        """Create a fallback result when refinement fails"""

        original_count = self._count_words(context.original_answer)

        # Create basic word limits
        fallback_limits = CBSEWordLimits(
            min_words=max(1, context.marks * 20),
            max_words=context.marks * 50,
            target_words=context.marks * 35,
            description=f"Fallback limits for {context.marks}-mark question"
        )

        return RefinementResult(
            refined_answer=context.original_answer,  # Return original if refinement fails
            original_word_count=original_count,
            refined_word_count=original_count,
            meets_cbse_standards=False,
            word_limit_info=fallback_limits,
            quality_score=0.5,  # Neutral score
            preserved_elements=[],
            refinement_strategy="fallback",
            cbse_compliance_notes=f"⚠️ Refinement failed: {error or 'Unknown error'} | Original answer returned"
        )

    def get_cbse_guidelines(self, subject: str, marks: int, section: SubjectSection = None) -> Dict[str, Any]:
        """Get CBSE guidelines for specific question parameters"""

        try:
            # Get word limits
            question_type = QuestionType.SUBJECTIVE  # Default
            word_limits = self._get_word_limits(subject, marks, section or SubjectSection.GENERAL, question_type)

            # Get subject-specific guidelines
            subject_guidelines = {
                "Mathematics": {
                    "focus": "Step-by-step solutions",
                    "format": "Given → Formula → Solution → Answer",
                    "requirements": ["Show all steps", "Include formulas", "Proper notation", "Final answer boxed"],
                    "avoid": ["Skipping steps", "Missing units", "Unclear working"]
                },
                "Science": {
                    "focus": "Concept clarity with applications",
                    "format": "Definition → Explanation → Example → Conclusion",
                    "requirements": ["Scientific terminology", "Proper units", "Labeled diagrams", "Real examples"],
                    "avoid": ["Vague explanations", "Missing units", "Incorrect terms"]
                },
                "Social Science": {
                    "focus": "Comprehensive understanding",
                    "format": "Introduction → Main points → Examples → Conclusion",
                    "requirements": ["Factual accuracy", "Current examples", "Maps/diagrams", "Chronological order"],
                    "avoid": ["Factual errors", "Outdated examples", "Poor organization"]
                },
                "English": {
                    "focus": "Language proficiency and analysis",
                    "format": "Context → Analysis → Examples → Conclusion",
                    "requirements": ["Proper grammar", "Literary devices", "Text references",
                                     "Personal interpretation"],
                    "avoid": ["Grammar errors", "Off-topic content", "Missing quotes"]
                },
                "Hindi": {
                    "focus": "भाषा प्रवाहता और विश्लेषण",
                    "format": "प्रसंग → विश्लेषण → उदाहरण → निष्कर्ष",
                    "requirements": ["व्याकरण शुद्धता", "साहित्यिक तत्व", "पाठ संदर्भ", "व्यक्तिगत मत"],
                    "avoid": ["व्याकरण त्रुटियां", "विषय से भटकाव", "अपूर्ण उत्तर"]
                }
            }

            guidelines = subject_guidelines.get(subject, subject_guidelines["Science"])

            return {
                "word_limits": word_limits.__dict__ if word_limits else None,
                "subject_guidelines": guidelines,
                "cbse_format": {
                    "structure": guidelines["format"],
                    "requirements": guidelines["requirements"],
                    "common_mistakes": guidelines["avoid"]
                },
                "marking_emphasis": {
                    1: "Accuracy only",
                    2: "Concept + Application",
                    3: "Detailed explanation + Example",
                    4: "Comprehensive analysis (sub-parts)",
                    5: "Complete mastery demonstration"
                }.get(marks, "Complete understanding")
            }

        except Exception as e:
            logger.error(f"Error getting CBSE guidelines: {str(e)}")
            return {"error": str(e)}

    def batch_refine_answers(self, contexts: List[RefinementContext]) -> List[RefinementResult]:
        """Refine multiple answers in batch for efficiency"""

        results = []

        for i, context in enumerate(contexts):
            try:
                logger.info(f"Processing answer {i + 1}/{len(contexts)}")
                result = self.refine_answer(context)
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing answer {i + 1}: {str(e)}")
                results.append(self._create_fallback_result(context, str(e)))

        return results

    def validate_cbse_compliance(self, answer: str, subject: str, marks: int,
                                     question_type: QuestionType = QuestionType.SUBJECTIVE) -> Dict[str, Any]:
            """Validate if an answer meets CBSE compliance standards"""

            word_count = self._count_words(answer)
            section = SubjectSection.GENERAL

            # Get word limits
            word_limits = self._get_word_limits(subject, marks, section, question_type)

            if not word_limits:
                return {
                    "compliant": False,
                    "reason": "No word limits found for given parameters",
                    "suggestions": ["Check subject and marks combination"],
                    "word_count": word_count,
                    "expected_range": "Unknown"
                }

            compliance_issues = []
            suggestions = []

            # Check word count compliance
            if word_limits.is_exact:
                if word_limits.max_words == 0:  # MCQ
                    if not re.match(r'^[a-d]$|^$[a-d]$$', answer.strip()):
                        compliance_issues.append("MCQ format incorrect - should be option letter only")
                        suggestions.append("Provide only the correct option (a), (b), (c), or (d)")
                elif word_count != word_limits.target_words:
                    compliance_issues.append(f"Exact word count not met: {word_count} vs {word_limits.target_words}")
                    suggestions.append(f"Adjust to exactly {word_limits.target_words} words")
            else:
                if word_count < word_limits.min_words:
                    compliance_issues.append(f"Below minimum word count: {word_count}/{word_limits.min_words}")
                    suggestions.append(f"Add more detail to reach at least {word_limits.min_words} words")
                elif word_count > word_limits.max_words:
                    compliance_issues.append(f"Exceeds maximum word count: {word_count}/{word_limits.max_words}")
                    suggestions.append(f"Reduce content to stay within {word_limits.max_words} words")

            # Subject-specific compliance checks
            if subject == "Mathematics" and not word_limits.is_exact:
                required_elements = ["Given:", "Formula:", "Solution:", "Therefore:"]
                missing_elements = [elem for elem in required_elements
                                    if elem.lower() not in answer.lower()]
                if missing_elements:
                    compliance_issues.append(f"Missing CBSE format elements: {missing_elements}")
                    suggestions.append("Include proper mathematical format with Given, Formula, Solution, Answer")

            elif subject in ["Science", "Social Science"] and word_count > 30:
                # Check for proper conclusion
                if not re.search(r'(?:therefore|hence|thus|∴|conclusion)', answer, re.IGNORECASE):
                    compliance_issues.append("Missing proper conclusion")
                    suggestions.append("Add a clear conclusion using 'Therefore' or 'Hence'")

                # Check for units in numerical answers
                if subject == "Science" and re.search(r'\d+', answer):
                    if not re.search(r'\b(?:m|cm|mm|km|kg|g|mg|s|min|hr|°C|K|mol|A|V|Ω|J|W|N|Pa)\b', answer):
                        compliance_issues.append("Missing units in numerical answer")
                        suggestions.append("Include appropriate SI units with numerical values")

            # Format compliance checks
            format_issues = []
            if marks >= 3 and not word_limits.is_exact:
                # Check for proper structure
                if not re.search(r'(?:given|definition|explanation)', answer, re.IGNORECASE):
                    format_issues.append("Missing clear introduction/definition")

                if not re.search(r'(?:example|application|instance)', answer,
                                 re.IGNORECASE) and subject != "Mathematics":
                    format_issues.append("Missing examples or applications")

            if format_issues:
                compliance_issues.extend(format_issues)
                suggestions.append("Follow CBSE answer structure with introduction, explanation, and examples")

            # Overall compliance assessment
            is_compliant = len(compliance_issues) == 0

            return {
                "compliant": is_compliant,
                "word_count": word_count,
                "expected_range": f"{word_limits.min_words}-{word_limits.max_words}" if not word_limits.is_exact else str(
                    word_limits.target_words),
                "word_limit_description": word_limits.description,
                "compliance_issues": compliance_issues,
                "suggestions": suggestions,
                "quality_indicators": {
                    "has_proper_format": "given:" in answer.lower() or "definition" in answer.lower(),
                    "has_conclusion": bool(re.search(r'(?:therefore|hence|thus|∴)', answer, re.IGNORECASE)),
                    "has_examples": bool(re.search(r'(?:example|for instance|such as)', answer, re.IGNORECASE)),
                    "has_units": bool(re.search(r'\b(?:m|cm|kg|g|s|°C|K|Rs|₹)\b', answer)),
                    "proper_length": word_limits.min_words <= word_count <= word_limits.max_words if not word_limits.is_exact else word_count == word_limits.target_words
                }
            }

    # Helper functions for integration with existing system

def create_refinement_context(question: str, answer: str, marks: int, subject: str,
                              board: str = "CBSE", chapter_name: str = "",
                              concept_title: str = "", keywords: List[str] = None,
                              subject_section: str = "general") -> RefinementContext:
    """Helper function to create refinement context"""

    # Map subject sections
    section_mapping = {
        "literature": SubjectSection.LITERATURE,
        "grammar": SubjectSection.GRAMMAR,
        "writing": SubjectSection.WRITING,
        "reading": SubjectSection.READING,
        "साहित्य": SubjectSection.SAHITYA,
        "व्याकरण": SubjectSection.VYAKARAN,
        "रचनात्मक लेखन": SubjectSection.RACHNATMAK_LEKHAN,
        "अपठित बोध": SubjectSection.APATHIT_BODH,
        "general": SubjectSection.GENERAL
    }

    section = section_mapping.get(subject_section.lower(), SubjectSection.GENERAL)

    return RefinementContext(
        original_answer=answer,
        question=question,
        marks=marks,
        subject=subject,
        board=board,
        chapter_name=chapter_name,
        concept_title=concept_title,
        keywords=keywords or [],
        question_type=QuestionType.SUBJECTIVE,  # Will be detected automatically
        subject_section=section
    )

def get_cbse_word_limits_summary() -> Dict[str, Any]:
    """Get a summary of all CBSE word limits for reference"""

    return {
        "Science": {
            "1_mark_mcq": "Only correct option",
            "2_marks": "30-50 words",
            "3_marks": "50-80 words",
            "4_marks": "Sub-questions: 1M(20-30), 2M(30-50)",
            "5_marks": "80-120 words"
        },
        "Social_Science": {
            "1_mark_mcq": "Only correct option",
            "2_marks": "30-50 words",
            "3_marks": "50-80 words",
            "4_marks": "Sub-questions: 1M(20-30), 2M(30-50)",
            "5_marks": "80-120 words"
        },
        "Mathematics": {
            "1_mark_mcq": "Only correct option",
            "2_to_5_marks": "Show required steps (word count not specified)"
        },
        "English": {
            "Literature": {
                "3_marks": "40-50 words",
                "5_marks": "Mixed format with sub-parts",
                "6_marks": "100-120 words"
            },
            "Grammar": {
                "1_mark": "Exact words needed (1-2 words typically)"
            },
            "Writing": {
                "5_marks": "120 words"
            },
            "Reading": {
                "10_marks": "Mixed format with sub-parts"
            }
        },
        "Hindi": {
            "साहित्य": {
                "1_marks": "40-50 words",
                "2_marks": "25-30 words",
                "4_marks": "50-60 words",
                "5_marks_mcq": "Only correct option"
            },
            "व्याकरण": {
                "1_mark": "As required by question"
            },
            "रचनात्मक_लेखन": {
                "4_marks": "40 words",
                "5_marks": "80-100 words",
                "6_marks": "120 words"
            },
            "अपठित_बोध": {
                "7_marks": "Mixed format with sub-parts"
            }
        }
    }

# Example usage and testing functions

def test_refinement_agent():
    """Test function for the refinement agent"""

    agent = AnswerRefinementAgent()

    # Test case 1: Science 3-mark question (needs compression)
    context1 = create_refinement_context(
        question="Explain the process of photosynthesis in plants.",
        answer="""
        Photosynthesis is a very important biological process that occurs in plants. 
        As we all know, it is the process by which plants make their own food using sunlight, 
        carbon dioxide, and water. The process takes place in the chloroplasts of plant cells, 
        specifically in the green parts like leaves. The chlorophyll pigment captures sunlight 
        energy. From the above discussion, we can see that carbon dioxide enters through stomata, 
        water is absorbed by roots. The overall reaction is 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. 
        Therefore, we can conclude that photosynthesis produces glucose and oxygen as products.
        """,
        marks=3,
        subject="Science",
        chapter_name="Life Processes"
    )

    result1 = agent.refine_answer(context1)
    print(f"Test 1 - Original: {result1.original_word_count} words")
    print(f"Test 1 - Refined: {result1.refined_word_count} words")
    print(f"Test 1 - Compliant: {result1.meets_cbse_standards}")
    print(f"Test 1 - Refined Answer:\n{result1.refined_answer}\n")

    # Test case 2: Mathematics 2-mark question
    context2 = create_refinement_context(
        question="Find the value of x if 2x + 5 = 15",
        answer="We need to solve 2x + 5 = 15. Subtracting 5: 2x = 10. Dividing by 2: x = 5",
        marks=2,
        subject="Mathematics"
    )

    result2 = agent.refine_answer(context2)
    print(f"Test 2 - Refined Answer:\n{result2.refined_answer}\n")

    return [result1, result2]

# Export all classes and functions
__all__ = [
    'AnswerRefinementAgent',
    'RefinementContext',
    'RefinementResult',
    'CBSEWordLimits',
    'QuestionType',
    'SubjectSection',
    'create_refinement_context',
    'get_cbse_word_limits_summary',
    'test_refinement_agent'
]