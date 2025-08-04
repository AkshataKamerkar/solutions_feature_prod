# app.py

import streamlit as st
import logging
from typing import Optional
import time
import json

# Updated imports for enhanced pipeline
from src.pipeline.qa_pipeline import QAPipeline
from src.config import config
from src.evaluation.metrics import CBSEEvaluationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced CBSE Solutions Feature",
    page_icon="📚",
    layout="wide"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .context-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0066cc;
    }
    
    .answer-section {
    background-color: #f7f7f9;
    border-left: 4px solid #007acc;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    width: 100%;
    max-width: 100%;
    }
    .cbse-compliant {
        background-color: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cbse-non-compliant {
        background-color: #ffeaa7;
        border-left: 4px solid #fdcb6e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 0.5rem;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .evaluation-criteria {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .score-high {
        background-color: #4CAF50;
        color: white;
    }
    .score-medium {
        background-color: #FFC107;
        color: black;
    }
    .score-low {
        background-color: #F44336;
        color: white;
    }
    .refinement-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    .word-count-info {
        background-color: #f3e5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9em;
    }
    /* Additional fix: make tabs content expand to full width */
    .stTabs .answer-section {
        width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Remove default horizontal padding of tab contents */
    .stTabs [data-testid="stVerticalBlock"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Expand any text container inside answer-section */
    .answer-section p,
    .answer-section div,
    .answer-section pre,
    .answer-section code {
        width: 100% !important;
        max-width: 100% !important;
        word-break: break-word;
    }
    
    /* Force full-width layout for entire app */
        div.block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            width: 100% !important;
        }
        
        /* Expand content inside all tabs fully */
        .stTabs [data-testid="stVerticalBlock"] {
            max-width: 100% !important;
            width: 100% !important;
            flex-grow: 1 !important;
        }
        
        /* Also apply full width to markdown and columns inside tabs */
        .stTabs [data-testid="stMarkdownContainer"],
        .stTabs .element-container,
        .stTabs [data-testid="column"] {
            max-width: 100% !important;
            width: 100% !important;
        }
        
        /* Force full width for answer-section inside any layout */
        .answer-section {
            width: 100% !important;
            max-width: 100% !important;
            display: block;
            flex-grow: 1;
            padding-left: 1rem;
            padding-right: 1rem;
            box-sizing: border-box;
        }
        
        /* Expand Streamlit's default layout container */
        div.block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            width: 100% !important;
        }
        
        /* Expand the vertical block inside tabs */
        .stTabs [data-testid="stVerticalBlock"] {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        /* Ensure markdown inside answer-section is not restricted */
        .answer-section p,
        .answer-section div,
        .answer-section pre,
        .answer-section code,
        .answer-section .element-container,
        .answer-section [data-testid="stMarkdownContainer"],
        .answer-section [data-testid="column"] {
            width: 100% !important;
            max-width: 100% !important;
            flex-grow: 1 !important;
            box-sizing: border-box;
        }
        html, body {s
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: hidden;
        }


</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    try:
        with st.spinner("🚀 Initializing Enhanced CBSE Pipeline..."):
            st.session_state.pipeline = QAPipeline()
        st.success("✅ Enhanced Pipeline initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []


def display_cbse_criteria(marks: int):
    """Display CBSE evaluation criteria for the selected marks"""
    try:
        metrics = CBSEEvaluationMetrics.get_evaluation_metrics(marks)

        with st.expander(f"📋 CBSE Evaluation Criteria for {marks}-mark Questions", expanded=False):
            st.markdown(f"**General Instructions:** {metrics['general_instructions']}")

            for criterion in metrics['evaluation_criteria']:
                if criterion.get('internal_use_only', False):
                    continue

                st.markdown(f"### {criterion['criteria_name']}")
                st.markdown(f"*{criterion['description']}*")
                st.markdown(f"**Max Marks:** {criterion['max_marks']}")

                # Display marking scheme
                if 'marking_scheme' in criterion:
                    st.markdown("**Marking Scheme:**")
                    for condition, marks_value in criterion['marking_scheme'].items():
                        st.write(f"• {condition.replace('_', ' ').title()}: {marks_value}")

                st.markdown("---")

    except Exception as e:
        logger.error(f"Error displaying CBSE criteria: {str(e)}")


def display_cbse_compliance_info(result: dict):
    """Display CBSE compliance information"""

    compliance_class = "cbse-compliant" if result.get('cbse_compliant', False) else "cbse-non-compliant"
    compliance_icon = "✅" if result.get('cbse_compliant', False) else "⚠️"
    compliance_text = "CBSE Compliant" if result.get('cbse_compliant', False) else "Needs CBSE Refinement"

    st.markdown(f'''
    <div class="{compliance_class}">
        <h4>{compliance_icon} {compliance_text}</h4>
        <p><strong>Ready for Board Exam:</strong> {"Yes" if result.get('ready_for_exam', False) else "Needs improvement"}</p>
    </div>
    ''', unsafe_allow_html=True)


def display_refinement_info(refinement_info: dict):
    """Display word count and refinement information"""

    if not refinement_info or not refinement_info.get('applied'):
        return

    st.markdown('<div class="refinement-info">', unsafe_allow_html=True)
    st.markdown("### 🔄 CBSE Refinement Applied")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Words", refinement_info.get('original_word_count', 0))

    with col2:
        st.metric("Refined Words", refinement_info.get('refined_word_count', 0))

    with col3:
        word_change = refinement_info.get('refined_word_count', 0) - refinement_info.get('original_word_count', 0)
        st.metric("Word Change", f"{word_change:+d}")

    with col4:
        st.metric("Quality Score", f"{refinement_info.get('quality_score', 0):.2f}")

    st.markdown(f"**Strategy Used:** {refinement_info.get('strategy_used', 'none')}")
    st.markdown(f"**Compliance Notes:** {refinement_info.get('compliance_notes', 'Not applied')}")

    st.markdown('</div>', unsafe_allow_html=True)


def display_enhanced_answer_comparison(result: dict):
    """Enhanced display function for the new pipeline results"""

    if not result.get('success', False):
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        if result.get('answers', {}).get('direct_answer'):
            st.subheader("📝 Direct LLM Answer")
            st.write(result['answers']['direct_answer'])
        return

    # Display CBSE Compliance Status
    display_cbse_compliance_info(result)

    # Display context information
    st.markdown("### 📋 Context Information")
    with st.container():
        st.markdown('<div class="context-box">', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"**📚 Subject:** {result.get('subject', 'Unknown')}")
            st.markdown(f"**🎓 Board:** {result.get('board', 'Unknown')}")

        with col2:
            st.markdown(f"**📖 Chapter:** {result.get('chapter', 'Unknown')}")
            st.markdown(f"**⭐ Marks:** {result.get('marks', 0)}")

        with col3:
            context = result.get('context', {})
            st.markdown(f"**📄 Section:** {context.get('concept_title', 'N/A')}")
            st.markdown(f"**🎯 Similarity:** {context.get('similarity_score', 0):.2f}")

        with col4:
            st.markdown(f"**📊 CBSE Compliant:** {'✅' if result.get('cbse_compliant') else '❌'}")
            st.markdown(f"**🎯 Exam Ready:** {'✅' if result.get('ready_for_exam') else '❌'}")

        # Display keywords and summary
        context = result.get('context', {})
        if context.get('keywords'):
            st.markdown(f"**🔑 Keywords:** {context['keywords'][:100]}...")

        with st.expander("📝 Reference Content"):
            st.write(context.get('summary_text', 'No summary available'))

        st.markdown('</div>', unsafe_allow_html=True)

    # Display CBSE evaluation criteria
    if result.get('board') == 'CBSE':
        display_cbse_criteria(result.get('marks', 1))

    # Display refinement information
    if result.get('refinement_info'):
        display_refinement_info(result['refinement_info'])

    # Enhanced answer display with new answer types
    st.markdown("### 📝 Enhanced Answer Comparison")

    # Determine number of tabs based on available answers
    answers = result.get('answers', {})
    tabs_config = []

    if answers.get('direct_answer'):
        tabs_config.append("🤖 Direct Answer")

    if answers.get('agent_answer'):
        tabs_config.append("📚 Context-Aware Answer")

    if answers.get('cbse_refined_answer'):
        tabs_config.append("✅ CBSE Refined Answer")

    if answers.get('final_answer'):
        tabs_config.append("⭐ Final Answer")

    tabs_config.append("📊 Evaluation Details")

    # Create tabs
    tabs = st.tabs(tabs_config)
    tab_index = 0

    # Direct Answer Tab
    if answers.get('direct_answer'):
        with tabs[tab_index]:
            st.markdown('<div class="answer-section">', unsafe_allow_html=True)
            st.markdown("#### 🤖 Direct LLM Response")
            st.info("Generated without database context - baseline response")

            # Word count info
            word_count = len(answers['direct_answer'].split())
            st.markdown(f'<div class="word-count-info">📊 Word Count: {word_count}</div>', unsafe_allow_html=True)

            st.write(answers['direct_answer'])
            st.markdown('</div>', unsafe_allow_html=True)
        tab_index += 1

    # Context-Aware Answer Tab
    if answers.get('agent_answer'):
        with tabs[tab_index]:
            st.markdown('<div class="answer-section">', unsafe_allow_html=True)
            st.markdown("#### 📚 Context-Aware Agent Response")
            context = result.get('context', {})
            st.info(
                f"Generated using content from: **{context.get('chapter', 'Unknown')}** - *{context.get('concept_title', 'Unknown')}*")

            # Word count info
            word_count = len(answers['agent_answer'].split())
            st.markdown(f'<div class="word-count-info">📊 Word Count: {word_count}</div>', unsafe_allow_html=True)

            st.write(answers['agent_answer'])
            st.markdown('</div>', unsafe_allow_html=True)
        tab_index += 1

    # CBSE Refined Answer Tab
    if answers.get('cbse_refined_answer'):
        with tabs[tab_index]:
            st.markdown('<div class="answer-section">', unsafe_allow_html=True)
            st.markdown("#### ✅ CBSE Refined Answer")
            st.success("Refined to meet exact CBSE word count and formatting standards")

            # Word count and compliance info
            refinement_info = result.get('refinement_info', {})
            if refinement_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f'<div class="word-count-info">📊 Word Count: {refinement_info.get("refined_word_count", "N/A")}</div>',
                        unsafe_allow_html=True)
                with col2:
                    st.markdown(
                        f'<div class="word-count-info">🎯 Quality Score: {refinement_info.get("quality_score", 0):.2f}</div>',
                        unsafe_allow_html=True)
                with col3:
                    st.markdown(
                        f'<div class="word-count-info">🔄 Strategy: {refinement_info.get("strategy_used", "N/A")}</div>',
                        unsafe_allow_html=True)

            st.write(answers['cbse_refined_answer'])
            st.markdown('</div>', unsafe_allow_html=True)
        tab_index += 1

    # Final Answer Tab
    if answers.get('final_answer'):
        with tabs[tab_index]:
            st.markdown('<div class="answer-section">', unsafe_allow_html=True)
            st.markdown("#### ⭐ Final Evaluated Answer")
            st.success("This is the final answer after CBSE evaluation and improvements")

            # Display evaluation score with enhanced visualization
            evaluation = result.get('evaluation', {})
            if evaluation.get('total_score') is not None:
                score = evaluation['total_score']
                max_score = evaluation.get('max_possible', result.get('marks', 1))
                percentage = evaluation.get('percentage', 0)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Enhanced score display
                    if percentage >= 80:
                        emoji = "🟢"
                        grade = "Excellent"
                    elif percentage >= 70:
                        emoji = "🟡"
                        grade = "Good"
                    elif percentage >= 50:
                        emoji = "🟠"
                        grade = "Average"
                    else:
                        emoji = "🔴"
                        grade = "Needs Improvement"

                    st.markdown(f'''
                    <div class="metric-card">
                        <h2>{emoji} {score}/{max_score}</h2>
                        <p><strong>CBSE Score</strong></p>
                        <p>{percentage:.1f}% - {grade}</p>
                    </div>
                    ''', unsafe_allow_html=True)

                with col2:
                    if evaluation.get('strengths'):
                        st.markdown("**✅ Strengths:**")
                        for strength in evaluation['strengths'][:3]:
                            st.write(f"• {strength}")

                with col3:
                    if evaluation.get('improvements'):
                        st.markdown("**📈 Improvements:**")
                        for improvement in evaluation['improvements'][:3]:
                            st.write(f"• {improvement}")

                with col4:
                    # Additional metrics
                    st.markdown("**📊 Metrics:**")
                    st.write(f"• Criterion Scores: {len(evaluation.get('criterion_scores', {}))}")
                    st.write(f"• CBSE Compliant: {'Yes' if result.get('cbse_compliant') else 'No'}")
                    st.write(f"• Exam Ready: {'Yes' if result.get('ready_for_exam') else 'No'}")

            # Display final answer
            st.markdown("---")
            st.markdown("**📝 Final Answer:**")
            word_count = len(answers['final_answer'].split())
            st.markdown(f'<div class="word-count-info">📊 Word Count: {word_count}</div>', unsafe_allow_html=True)
            st.write(answers['final_answer'])
            st.markdown('</div>', unsafe_allow_html=True)
        tab_index += 1

    # Evaluation Details Tab
    with tabs[tab_index]:
        st.markdown("### 📊 Comprehensive Evaluation Details")

        evaluation = result.get('evaluation', {})

        # Score overview
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Score Overview")
            if evaluation.get('total_score') is not None:
                st.metric("Total Score",
                          f"{evaluation['total_score']}/{evaluation.get('max_possible', result.get('marks', 1))}")
                st.metric("Percentage", f"{evaluation.get('percentage', 0):.1f}%")

                # Progress bar for score
                score_percentage = evaluation.get('percentage', 0) / 100
                st.progress(score_percentage)

        with col2:
            st.markdown("#### 📈 Performance Analysis")
            if result.get('cbse_compliant'):
                st.success("✅ Meets CBSE Standards")
            else:
                st.warning("⚠️ Needs CBSE Refinement")

            if result.get('ready_for_exam'):
                st.success("🎯 Ready for Board Exam")
            else:
                st.info("📚 Needs More Practice")

        # Criterion-wise scores
        if evaluation.get('criterion_scores'):
            st.markdown("#### 📋 Criterion-wise Analysis")

            for criterion, score in evaluation['criterion_scores'].items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{criterion}**")
                with col2:
                    st.write(f"**{score}**")

        # Detailed feedback sections
        col1, col2 = st.columns(2)

        with col1:
            if evaluation.get('strengths'):
                st.markdown("#### ✅ Identified Strengths")
                for i, strength in enumerate(evaluation['strengths'], 1):
                    st.write(f"{i}. {strength}")

        with col2:
            if evaluation.get('improvements'):
                st.markdown("#### 📈 Areas for Improvement")
                for i, improvement in enumerate(evaluation['improvements'], 1):
                    st.write(f"{i}. {improvement}")

        # Complete evaluation report
        if evaluation.get('detailed_feedback'):
            with st.expander("📝 Complete Evaluation Report"):
                st.text(evaluation['detailed_feedback'])

        # Processing metadata
        metadata = result.get('metadata', {})
        if metadata:
            with st.expander("⚙️ Processing Information"):
                st.json(metadata)

def display_word_count_guidelines(board: str, subject: str, marks: int):
    """Display CBSE word count guidelines"""

    if board != "CBSE":
        return

    # CBSE word count guidelines
    word_guidelines = {
        "Science": {
            1: "Only correct option (MCQ)",
            2: "30-50 words",
            3: "50-80 words",
            4: "Sub-questions: 1M(20-30), 2M(30-50)",
            5: "80-120 words"
        },
        "Social Science": {
            1: "Only correct option (MCQ)",
            2: "30-50 words",
            3: "50-80 words",
            4: "Sub-questions: 1M(20-30), 2M(30-50)",
            5: "80-120 words"
        },
        "Mathematics": {
            1: "Only correct option (MCQ)",
            2: "Show required steps",
            3: "Show required steps",
            4: "Show required steps",
            5: "Show required steps"
        },
        "English": {
            1: "Varies by section",
            2: "30-50 words (subjective)",
            3: "40-50 words (Literature)",
            4: "Mixed format",
            5: "120 words (Writing)"
        },
        "Hindi": {
            1: "40-50 words (साहित्य)",
            2: "25-30 words (साहित्य)",
            3: "Varies by section",
            4: "40 words (रचनात्मक लेखन)",
            5: "80-100 words (रचनात्मक लेखन)"
        }
    }

    guidelines = word_guidelines.get(subject, word_guidelines.get("Science", {}))
    guideline = guidelines.get(marks, "Standard CBSE format")

    st.info(f"📏 **CBSE Word Limit for {subject} ({marks} marks):** {guideline}")

def create_question_examples():
    """Create example questions for different subjects and boards"""

    return {
        "CBSE": {
            "Physics": [
                {"q": "State the laws of reflection of light.", "marks": 2},
                {"q": "Derive the mirror formula and explain its significance.", "marks": 5},
                {"q": "What is meant by power of a lens?", "marks": 1}
            ],
            "Biology": [
                {"q": "What is photosynthesis? Explain the process.", "marks": 3},
                {"q": "Describe the structure and function of nephron.", "marks": 5},
                {"q": "Name the respiratory pigment in human blood.", "marks": 1}
            ],
            "Mathematics": [
                {"q": "Find the discriminant of 2x² + 3x + 1 = 0", "marks": 2},
                {"q": "Prove that √2 is an irrational number.", "marks": 3},
                {"q": "Solve: 3x + 2y = 11 and 2x + 3y = 4", "marks": 3}
            ],
            "Chemistry": [
                {"q": "What are acids and bases? Give examples.", "marks": 2},
                {"q": "Explain the process of electrolysis of water.", "marks": 3},
                {"q": "Write the chemical formula of washing soda.", "marks": 1}
            ],
            "History": [
                {"q": "Explain the causes of the First World War.", "marks": 5},
                {"q": "What was the impact of the Revolt of 1857?", "marks": 3},
                {"q": "Who was the founder of the Mauryan Empire?", "marks": 1}
            ],
            "Geography": [
                {"q": "Explain the water cycle with a diagram.", "marks": 3},
                {"q": "What are the major landforms of India?", "marks": 5},
                {"q": "Name the longest river in India.", "marks": 1}
            ]
        }
    }

def export_session_data():
    """Export session data for analysis"""

    if not st.session_state.history:
        return None

    export_data = {
        "session_info": {
            "total_questions": len(st.session_state.history),
            "timestamp": time.time(),
            "session_duration": time.time() - st.session_state.history[0][
                'timestamp'] if st.session_state.history else 0
        },
        "questions": st.session_state.history
    }

    return json.dumps(export_data, indent=2)

def main():
    """Enhanced main Streamlit application with CBSE compliance features"""

    # Header with enhanced branding
    st.title("📚 Enhanced CBSE Solutions Feature")
    st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4>🎯 Features:</h4>
            <ul>
                <li><strong>Direct LLM Generation:</strong> Basic AI-powered answers</li>
                <li><strong>Context-Aware Responses:</strong> Answers using textbook content from your database</li>
                <li><strong>CBSE Compliance:</strong> Answers refined to meet exact CBSE word count and format standards</li>
                <li><strong>Comprehensive Evaluation:</strong> Detailed scoring using CBSE marking schemes</li>
                <li><strong>Multi-Board Support:</strong> CBSE, ICSE, State Boards, and more</li>
            </ul>
        </div>

        """, unsafe_allow_html=True)



    # Enhanced sidebar with more options
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Board selection with descriptions
        board_options = {
            "CBSE": "Central Board of Secondary Education",
            "ICSE": "Indian Certificate of Secondary Education",
            "STATE_BOARD": "State Education Boards",
            "IB": "International Baccalaureate",
            "CAMBRIDGE": "Cambridge International"
        }

        board_filter = st.selectbox(
            "🎓 Select Educational Board",
            list(board_options.keys()),
            index=0,
            help="Choose the educational board for board-specific answers and evaluation"
        )
        st.caption(board_options[board_filter])

        # Subject filter with board-specific subjects
        if board_filter == "CBSE":
            subject_options = ["All", "Mathematics", "Science", "Physics", "Chemistry", "Biology",
                               "English", "Hindi", "Social Science", "History", "Geography",
                               "Political Science", "Economics"]
        else:
            subject_options = ["All", "Mathematics", "Science", "English", "History", "Geography"]

        subject_filter = st.selectbox(
            "📚 Filter by Subject",
            subject_options,
            index=0,
            help="Filter content by subject for more relevant answers"
        )
        subject_filter = None if subject_filter == "All" else subject_filter

        # CBSE-specific options
        if board_filter == "CBSE":
            st.markdown("---")
            st.markdown("### 🎯 CBSE Specific Options")

            include_refinement = st.checkbox(
                "Apply CBSE Refinement",
                value=True,
                help="Refine answers to meet exact CBSE word count and format standards"
            )

            question_section = st.selectbox(
                "Question Section (for English/Hindi)",
                ["general", "literature", "grammar", "writing", "reading"],
                index=0,
                help="Specify section for subject-specific word limits"
            )
        else:
            include_refinement = False
            question_section = "general"

        # CBSE marking information
        if board_filter == "CBSE":
            st.markdown("---")
            st.markdown("### 📊 CBSE Question Pattern")
            cbse_info = {
                "1 mark": "MCQ/Objective • Very Short Answer",
                "2 marks": "Short Answer I • 30-50 words",
                "3 marks": "Short Answer II • 50-80 words",
                "4 marks": "Long Answer I • Case Studies",
                "5 marks": "Long Answer II • 80-120 words"
            }

            for marks, description in cbse_info.items():
                st.write(f"**{marks}:** {description}")

        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Example Questions")

        examples = create_question_examples()
        if board_filter in examples and subject_filter in examples[board_filter]:
            example_questions = examples[board_filter][subject_filter]

            for i, example in enumerate(example_questions[:3]):
                if st.button(
                        f"📝 {example['marks']}M: {example['q'][:25]}...",
                        key=f"example_{i}",
                        use_container_width=True
                ):
                    st.session_state.example_question = example['q']
                    st.session_state.example_marks = example['marks']
                    st.rerun()

        # History section
        st.markdown("---")
        st.header("📜 Recent Questions")
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                timestamp = time.strftime('%H:%M', time.localtime(item['timestamp']))
                success_icon = "✅" if item.get('success', False) else "❌"
                cbse_icon = "🎯" if item.get('cbse_compliant', False) else ""

                if st.button(
                        f"{success_icon}{cbse_icon} {timestamp} | {item['marks']}M\n{item['question'][:30]}...",
                        key=f"hist_{i}",
                        use_container_width=True
                ):
                    st.session_state.selected_question = item['question']
                    st.session_state.selected_marks = item['marks']
                    st.rerun()

        # System information
        st.markdown("---")
        st.markdown("### ⚙️ System Info")
        with st.expander("Technical Details"):
            st.text(f"Model: {config.LLM_MODEL}")
            st.text(f"Embeddings: {config.EMBEDDING_MODEL}")
            st.text(f"Index: {config.PINECONE_INDEX_NAME}")
            st.text(f"Dimension: {config.PINECONE_DIMENSION}")
            st.text(f"Top K: {config.TOP_K_RESULTS}")
            st.text(f"Threshold: {config.SIMILARITY_THRESHOLD}")

        # Export functionality
        if st.button("📥 Export Session Data") and st.session_state.history:
            export_data = export_session_data()
            if export_data:
                st.download_button(
                    label="💾 Download Session Data",
                    data=export_data,
                    file_name=f"session_data_{int(time.time())}.json",
                    mime="application/json"
                )

    # Main input area with enhanced layout
    st.markdown("### 📝 Enter Your Question")

    # Check for example or history selection
    default_question = st.session_state.get('example_question',
                                            st.session_state.get('selected_question', ''))
    default_marks = st.session_state.get('example_marks',
                                         st.session_state.get('selected_marks', 3))

    col1, col2 = st.columns([1, 3])

    with col1:
        question = st.text_area(
            "Enter your educational question:",
            value=default_question,
            height=120,
            placeholder="e.g., Explain the process of photosynthesis with a neat labeled diagram.",
            help="Enter any educational question. The system will find relevant content and provide comprehensive answers."
        )

        # Clear selections after use
        for key in ['example_question', 'selected_question', 'example_marks', 'selected_marks']:
            if key in st.session_state:
                del st.session_state[key]

    with col2:
        # Enhanced marks selection
        if board_filter == "CBSE":
            marks = st.selectbox(
                "📊 Question Marks:",
                options=[1, 2, 3, 4, 5],
                index=[1, 2, 3, 4, 5].index(default_marks) if default_marks in [1, 2, 3, 4, 5] else 2,
                help="Select marks according to CBSE question pattern"
            )
            # Display question type info
            question_types = {
                1: "MCQ/Objective",
                2: "Short Answer I",
                3: "Short Answer II",
                4: "Long Answer I",
                5: "Long Answer II"
            }
            st.caption(f"**Type:** {question_types.get(marks, 'Unknown')}")

        else:
            marks = st.number_input(
                "📊 Question Marks:",
                min_value=1,
                max_value=20,
                value=default_marks,
                step=1,
                help="Marks determine answer length and complexity"
            )

            # Advanced options
        with st.expander("⚙️ Advanced Options"):
            top_k = st.slider(
                "Search Results",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of relevant content pieces to consider"
            )

            if board_filter == "CBSE":
                show_word_guidelines = st.checkbox(
                    "Show Word Count Guidelines",
                    value=True,
                    help="Display CBSE word count requirements"
                )
            else:
                show_word_guidelines = False

        # Display word count guidelines for CBSE
        if board_filter == "CBSE" and subject_filter and show_word_guidelines:
            display_word_count_guidelines(board_filter, subject_filter, marks)

        # Enhanced submit section
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            submit_button = st.button(
                "🚀 Generate Enhanced Answer",
                type="primary",
                use_container_width=True,
                help="Process question through the complete enhanced pipeline"
            )

        with col2:
            if st.button("🔍 Search Only", use_container_width=True):
                if question:
                    with st.spinner("🔍 Searching database..."):
                        try:
                            results = st.session_state.pipeline._search_relevant_content(
                                question, board_filter, subject_filter
                            )

                            if results:
                                st.success(f"Found {len(results)} relevant results!")
                                for i, result in enumerate(results[:3]):
                                    with st.expander(f"Result {i + 1} (Score: {result.get('score', 0):.3f})"):
                                        metadata = result.get('metadata', {})
                                        st.write(f"**Chapter:** {metadata.get('chapter', 'Unknown')}")
                                        st.write(f"**Section:** {metadata.get('section_type', 'Unknown')}")
                                        st.write(f"**Summary:** {metadata.get('summary', 'No summary')[:200]}...")
                            else:
                                st.warning("No relevant content found in database")

                        except Exception as e:
                            st.error(f"Search error: {str(e)}")
                else:
                    st.error("Please enter a question first")

        with col3:
            if st.button("📋 Guidelines", use_container_width=True):
                if board_filter == "CBSE":
                    st.info("CBSE guidelines displayed above and in sidebar")
                    display_cbse_criteria(marks)
                else:
                    st.info(f"General guidelines for {board_filter} board")

        # Main processing
        if submit_button:
            if not question:
                st.error("❌ Please enter a question!")
                return

            # Enhanced progress tracking
            progress_container = st.container()

            try:
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Stage 1: Initialization
                    status_text.text("🚀 Initializing enhanced pipeline...")
                    progress_bar.progress(20)

                    # Stage 2: Processing
                    status_text.text("📝 Processing question through pipeline...")
                    progress_bar.progress(50)

                    # Process the question
                    result = st.session_state.pipeline.process_question(
                        question=question,
                        marks=marks,
                        subject_filter=subject_filter,
                        board_filter=board_filter,
                        include_cbse_refinement=include_refinement if board_filter == "CBSE" else False,
                        question_section=question_section if board_filter == "CBSE" else "general"
                    )

                    # Stage 3: Complete
                    status_text.text("✅ Processing complete!")
                    progress_bar.progress(100)

                    # Clear progress indicators
                    progress_container.empty()

                # Add to history with enhanced metadata
                history_item = {
                    'question': question,
                    'marks': marks,
                    'timestamp': time.time(),
                    'success': result.get('success', False),
                    'cbse_compliant': result.get('cbse_compliant', False),
                    'board': board_filter,
                    'subject': subject_filter,
                    'ready_for_exam': result.get('ready_for_exam', False)
                }
                st.session_state.history.append(history_item)

                # Display comprehensive results
                display_enhanced_answer_comparison(result)

                # Enhanced success/status messages
                if result.get('success', False):
                    success_metrics = []

                    if result.get('cbse_compliant'):
                        success_metrics.append("✅ CBSE Compliant")

                    if result.get('ready_for_exam'):
                        success_metrics.append("🎯 Exam Ready")

                    evaluation = result.get('evaluation', {})
                    if evaluation.get('percentage'):
                        success_metrics.append(f"📊 Score: {evaluation['percentage']:.1f}%")

                    if success_metrics:
                        st.success(f"🎉 Successfully generated enhanced answer! {' • '.join(success_metrics)}")
                    else:
                        st.success(f"✅ Successfully generated answers using {board_filter} curriculum!")

                    # Additional insights
                    insights = []

                    refinement_info = result.get('refinement_info')
                    if refinement_info and refinement_info.get('applied'):
                        word_change = refinement_info.get('refined_word_count', 0) - refinement_info.get(
                            'original_word_count', 0)
                        if word_change != 0:
                            insights.append(f"📝 Word count optimized ({word_change:+d} words)")

                    if result.get('context', {}).get('similarity_score', 0) > 0.8:
                        insights.append("🎯 High relevance content found")

                    if insights:
                        st.info(" • ".join(insights))

                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

                    # Show partial results if available
                    if result.get('answers', {}).get('direct_answer'):
                        st.markdown("### 📝 Partial Result Available")
                        st.write(result['answers']['direct_answer'])

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                logger.error(f"Error processing question: {str(e)}", exc_info=True)

                # Show troubleshooting tips
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                        **Common solutions:**
                        1. Check your internet connection
                        2. Verify API keys are configured correctly
                        3. Try a simpler question
                        4. Select a different subject or board
                        5. Reduce the number of search results (in Advanced Options)

                        **If the problem persists:**
                        - Check the system logs
                        - Verify Pinecone database connectivity
                        - Ensure OpenAI API quota is available
                        """)

        # Enhanced statistics and analytics section
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 📊 Session Analytics")

            # Calculate statistics
            total_questions = len(st.session_state.history)
            successful_questions = len([q for q in st.session_state.history if q.get('success', False)])
            cbse_compliant_questions = len([q for q in st.session_state.history if q.get('cbse_compliant', False)])
            exam_ready_questions = len([q for q in st.session_state.history if q.get('ready_for_exam', False)])

            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Questions", total_questions)

            with col2:
                success_rate = (successful_questions / total_questions * 100) if total_questions > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")

            with col3:
                if successful_questions > 0:
                    cbse_rate = (cbse_compliant_questions / successful_questions * 100)
                    st.metric("CBSE Compliant", f"{cbse_rate:.1f}%")
                else:
                    st.metric("CBSE Compliant", "N/A")

            with col4:
                if successful_questions > 0:
                    exam_ready_rate = (exam_ready_questions / successful_questions * 100)
                    st.metric("Exam Ready", f"{exam_ready_rate:.1f}%")
                else:
                    st.metric("Exam Ready", "N/A")

            with col5:
                # Average marks
                marks_list = [q.get('marks', 0) for q in st.session_state.history if q.get('success', False)]
                avg_marks = sum(marks_list) / len(marks_list) if marks_list else 0
                st.metric("Avg Marks", f"{avg_marks:.1f}")

            # Subject and board distribution
            if total_questions > 5:  # Only show if we have enough data
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📚 Subject Distribution")
                    subjects = {}
                    for q in st.session_state.history:
                        subject = q.get('subject', 'Unknown')
                        subjects[subject] = subjects.get(subject, 0) + 1

                    for subject, count in sorted(subjects.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total_questions * 100)
                        st.write(f"• {subject}: {count} ({percentage:.1f}%)")

                with col2:
                    st.markdown("#### 🎓 Board Distribution")
                    boards = {}
                    for q in st.session_state.history:
                        board = q.get('board', 'Unknown')
                        boards[board] = boards.get(board, 0) + 1

                    for board, count in sorted(boards.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total_questions * 100)
                        st.write(f"• {board}: {count} ({percentage:.1f}%)")

            # Clear history option
            if st.button("🗑️ Clear Session History", help="Clear all questions from this session"):
                st.session_state.history = []
                st.success("Session history cleared!")
                st.rerun()

        # Footer with enhanced information
        st.markdown("---")

        # Feature highlights
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Key Features")
            st.markdown("""
                - **Multi-Stage Processing**: Direct → Context → Refinement → Evaluation
                - **CBSE Compliance**: Exact word count and format standards
                - **Comprehensive Evaluation**: Detailed scoring with improvement suggestions
                - **Multi-Board Support**: CBSE, ICSE, State Boards, International
                - **Content Search**: AI-powered search through textbook database
                - **Session Analytics**: Track your learning progress
                """)

        with col2:
            st.markdown("### 💡 Tips for Best Results")
            st.markdown("""
                - **Be Specific**: Use clear, specific questions
                - **Include Context**: Mention chapter/topic if known
                - **Choose Right Marks**: Select appropriate marks for question complexity
                - **Use Filters**: Apply subject filters for focused results
                - **Review Guidelines**: Check CBSE word count requirements
                - **Practice Regularly**: Use the system to improve answer writing
                """)

        # System status indicator
        with st.expander("🔧 System Status"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🤖 AI Models**")
                st.write(f"• LLM: {config.LLM_MODEL}")
                st.write(f"• Embeddings: {config.EMBEDDING_MODEL}")
                st.write("• Status: ✅ Active")

            with col2:
                st.markdown("**🗄️ Database**")
                st.write(f"• Index: {config.PINECONE_INDEX_NAME}")
                st.write(f"• Dimension: {config.PINECONE_DIMENSION}")
                st.write("• Status: ✅ Connected")

            with col3:
                st.markdown("**⚡ Performance**")
                st.write(f"• Search Results: {config.TOP_K_RESULTS}")
                st.write(f"• Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
                st.write(f"• Session Questions: {len(st.session_state.history)}")

        # Version and credits
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
            "Enhanced CBSE Solutions Feature v2.0 | "
            "Powered by OpenAI GPT & Pinecone Vector Search | "
            "Built with Streamlit"
            "</div>",
            unsafe_allow_html=True
        )

# Run the enhanced application
if __name__ == "__main__":
    main()