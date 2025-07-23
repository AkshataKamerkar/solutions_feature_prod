import streamlit as st
import logging
from typing import Optional
from src.pipeline.qa_pipeline import QAPipeline
from src.config import config
from src.evaluation.metrics import CBSEEvaluationMetrics
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Solutions Feature",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .context-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .answer-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        text-align: center;
        padding: 0.5rem;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    try:
        st.session_state.pipeline = QAPipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
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
                
                # Display marking scheme
                if 'marking_scheme' in criterion:
                    cols = st.columns(len(criterion['marking_scheme']))
                    for i, (condition, marks_value) in enumerate(criterion['marking_scheme'].items()):
                        with cols[i]:
                            st.metric(
                                condition.replace('_', ' ').title(),
                                f"{marks_value} marks" if not isinstance(marks_value, str) else marks_value
                            )
    except Exception as e:
        logger.error(f"Error displaying CBSE criteria: {str(e)}")

def display_answer_comparison(result: dict):
    """Display the three answers in a comparison format with CBSE evaluation"""
    
    if not result['success']:
        st.error(f"Error: {result['error']}")
        if result.get('direct_answer'):
            st.subheader("Direct LLM Answer")
            st.write(result['direct_answer'])
        return
    
    # Display context information with all Pinecone fields
    st.markdown("### 📋 Context Information")
    with st.container():
        st.markdown('<div class="context-box">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**📚 Subject:** {result['context']['subject']}")
            st.markdown(f"**🎓 Board:** {result['context']['board']}")
        
        with col2:
            st.markdown(f"**📖 Chapter:** {result['context']['chapter']}")
            st.markdown(f"**📄 Concept:** {result['context']['concept_title']}")
        
        with col3:
            st.markdown(f"**🔑 Keywords:** {result['context']['keywords'][:100]}...")
        
        # Display the summary text in an expandable section
        with st.expander("📝 Reference Content"):
            st.write(result['context']['summary_text'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display CBSE evaluation criteria
    if result['context']['board'] == 'CBSE':
        display_cbse_criteria(result['marks'])
    
    # Display answers
    st.markdown("### 📝 Answer Comparison")
    
    # Create tabs for different answers
    tab1, tab2, tab3, tab4 = st.tabs([
        "Direct Answer", 
        "Context-Aware Answer", 
        "Final Answer (CBSE Evaluated) ⭐",
        "Evaluation Details"
    ])
    
    with tab1:
        st.markdown('<div class="answer-section">', unsafe_allow_html=True)
        st.markdown("#### Answer 1: Direct LLM Response")
        st.info("This answer is generated without any context from the database.")
        st.write(result['answers']['direct_answer'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="answer-section">', unsafe_allow_html=True)
        st.markdown("#### Answer 2: Context-Aware Agent Response")
        st.info(f"This answer is generated using content from: **{result['context']['chapter']}** - *{result['context']['concept_title']}*")
        st.write(result['answers']['agent_answer'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="answer-section">', unsafe_allow_html=True)
        st.markdown("#### Answer 3: CBSE-Evaluated Model Answer")
        st.success("This is the final answer evaluated and improved according to CBSE standards.")
        
        # Display evaluation score with color coding
        if result['evaluation']['score'] is not None:
            score = result['evaluation']['score']
            max_score = result['marks']
            percentage = (score / max_score) * 100
            
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                # Color code based on percentage
                if percentage >= 80:
                    score_class = "score-high"
                    emoji = "🟢"
                elif percentage >= 60:
                    score_class = "score-medium"
                    emoji = "🟡"
                else:
                    score_class = "score-low"
                    emoji = "🔴"
                
                st.markdown(
                    f'<div class="metric-card"><h2>{emoji} {score}/{max_score}</h2>'
                    f'<p>CBSE Score</p><p>({percentage:.0f}%)</p></div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                if result['evaluation']['strengths']:
                    st.markdown("**✅ Strengths:**")
                    for strength in result['evaluation']['strengths']:
                        st.write(f"• {strength}")
            
            with col3:
                if result['evaluation']['improvements']:
                    st.markdown("**📈 Areas Improved:**")
                    for improvement in result['evaluation']['improvements']:
                        st.write(f"• {improvement}")
        
        # Display final answer
        st.markdown("---")
        st.markdown("**CBSE Model Answer:**")
        st.write(result['answers']['final_answer'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### 📊 Detailed CBSE Evaluation")
        
        # Display criterion-wise scores
        if result['evaluation'].get('criterion_scores'):
            st.markdown("#### Criterion-wise Scores:")
            
            for criterion, score in result['evaluation']['criterion_scores'].items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{criterion}**")
                with col2:
                    st.write(f"Score: {score}")
        
        # Display detailed feedback
        if result['evaluation'].get('detailed_feedback'):
            with st.expander("📝 Complete Evaluation Report"):
                st.text(result['evaluation']['detailed_feedback'])

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📚 Solutions Feature")
    st.markdown("""
    This system provides comprehensive answers to educational questions using:
    - Direct LLM generation
    - Context-aware responses from textbook content
    - CBSE board-specific evaluation and model answers
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Board selection
        board_filter = st.selectbox(
            "🎓 Select Board",
            ["CBSE", "ICSE", "STATE_BOARD", "IB", "CAMBRIDGE"],
            index=0,
            help="Select the educational board for context-specific answers"
        )
        
        # Subject filter
        subject_filter = st.selectbox(
            "📚 Filter by Subject (Optional)",
            ["All", "Mathematics", "Science", "Physics", "Chemistry", "Biology", 
             "English", "History", "Geography"],
            index=0
        )
        subject_filter = None if subject_filter == "All" else subject_filter
        
        # CBSE-specific information
        if board_filter == "CBSE":
            st.markdown("---")
            st.markdown("### 📊 CBSE Marking Scheme")
            st.info("""
            **CBSE Question Types:**
            - 1 mark: Objective/MCQ
            - 2 marks: Short Answer I
            - 3 marks: Short Answer II
            - 4 marks: Long Answer I
            - 5 marks: Long Answer II
            """)
        
        # Display supported boards
        st.markdown("---")
        st.markdown("**Supported Boards:**")
        for board in config.SUPPORTED_BOARDS:
            st.write(f"• {board}")
        
        # History
        st.markdown("---")
        st.header("📜 Recent Questions")
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                timestamp = time.strftime('%H:%M', time.localtime(item['timestamp']))
                if st.button(f"🕐 {timestamp} | {item['marks']}m | {item['question'][:30]}...", 
                           key=f"hist_{i}",
                           use_container_width=True):
                    st.session_state.selected_question = item['question']
                    st.session_state.selected_marks = item['marks']
                    st.rerun()
    
    # Main input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Check for selected question from history
        default_question = st.session_state.get('selected_question', '')
        
        question = st.text_area(
            "Enter your question:",
            value=default_question,
            height=100,
            placeholder="e.g., Explain the process of photosynthesis with a diagram"
        )
        
        # Clear selection after use
        if 'selected_question' in st.session_state:
            del st.session_state.selected_question
    
    with col2:
        # Check for selected marks from history
        default_marks = st.session_state.get('selected_marks', 3)
        
        # CBSE-specific marks selection
        if board_filter == "CBSE":
            marks = st.selectbox(
                "Marks:",
                options=[1, 2, 3, 4, 5],
                index=[1, 2, 3, 4, 5].index(default_marks) if default_marks in [1, 2, 3, 4, 5] else 2,
                help="Select marks according to CBSE pattern"
            )
        else:
            marks = st.number_input(
                "Marks:",
                min_value=1,
                max_value=20,
                value=default_marks,
                step=1,
                help="Marks determine answer length and detail"
            )
        
        # Clear selection after use
        if 'selected_marks' in st.session_state:
            del st.session_state.selected_marks
        
        # Quick mark buttons for CBSE
        if board_filter == "CBSE":
            st.markdown("**Question Type:**")
            mark_descriptions = {
                1: "MCQ/Objective",
                2: "Short Answer I",
                3: "Short Answer II",
                4: "Long Answer I",
                5: "Long Answer II"
            }
            st.caption(mark_descriptions.get(marks, ""))
    
    # Show CBSE evaluation criteria preview
    if board_filter == "CBSE":
        display_cbse_criteria(marks)
    
    # Submit button
    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not question:
            st.error("❌ Please enter a question!")
            return
        
        # Add to history
        st.session_state.history.append({
            'question': question,
            'marks': marks,
            'timestamp': time.time()
        })
        
        # Process question
        with st.spinner("🔄 Processing your question..."):
            try:
                # Create progress container
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Stage 1: Direct Answer
                    status_text.text("📝 Generating direct answer...")
                    progress_bar.progress(20)
                    
                    # Process question
                    result = st.session_state.pipeline.process_question(
                        question=question,
                        marks=marks,
                        subject_filter=subject_filter,
                        board_filter=board_filter
                    )
                    
                    # Stage 2: Context Search
                    status_text.text(f"🔍 Searching {board_filter} curriculum content...")
                    progress_bar.progress(40)
                    time.sleep(0.5)
                    
                    # Stage 3: Agent Processing
                    status_text.text("🤖 Generating context-aware answer...")
                    progress_bar.progress(60)
                    time.sleep(0.5)
                    
                    # Stage 4: CBSE Evaluation
                    if board_filter == "CBSE":
                        status_text.text(f"✅ Evaluating using CBSE {marks}-mark criteria...")
                    else:
                        status_text.text("✅ Evaluating and improving answer...")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    # Clear progress
                    progress_container.empty()
                
                # Display results
                display_answer_comparison(result)
                
                # Show success message if all went well
                if result['success']:
                    if board_filter == "CBSE":
                        st.success(f"✅ Successfully generated answers using CBSE {marks}-mark evaluation criteria!")
                    else:
                        st.success(f"✅ Successfully generated answers using {board_filter} board curriculum!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                logger.error(f"Error processing question: {str(e)}", exc_info=True)
    
    # Footer with information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 System Info**")
        st.text(f"Model: {config.LLM_MODEL}")
        st.text(f"Embeddings: {config.EMBEDDING_MODEL}")
    with col2:
        st.markdown("**🗄️ Database**")
        st.text(f"Index: {config.PINECONE_INDEX_NAME}")
        st.text(f"Dimension: {config.PINECONE_DIMENSION}")
    with col3:
        st.markdown("**⚡ Performance**")
        st.text(f"Top K: {config.TOP_K_RESULTS}")
        st.text(f"Threshold: {config.SIMILARITY_THRESHOLD}")

if __name__ == "__main__":
    main()