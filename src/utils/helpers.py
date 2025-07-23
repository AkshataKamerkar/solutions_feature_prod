import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class QueryCache:
    """Simple in-memory cache for query results"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def _generate_key(self, question: str, marks: int) -> str:
        """Generate cache key from question and marks"""
        content = f"{question}_{marks}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, question: str, marks: int) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        key = self._generate_key(question, marks)
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, question: str, marks: int, result: Dict[str, Any]) -> None:
        """Cache a result"""
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove least recently accessed item
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        key = self._generate_key(question, marks)
        self.cache[key] = result
        self.access_times[key] = datetime.now()

def format_answer_for_display(answer: str, marks: int) -> str:
    """Format answer based on marks allocation"""
    if marks <= 2:
        # Short answer format
        return answer.strip()
    elif marks <= 5:
        # Medium answer format - ensure proper paragraphs
        paragraphs = answer.strip().split('\n\n')
        return '\n\n'.join(p.strip() for p in paragraphs if p.strip())
    else:
        # Long answer format - ensure structure
        return format_long_answer(answer)

def format_long_answer(answer: str) -> str:
    """Format long answers with proper structure"""
    lines = answer.strip().split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
        elif line.endswith(':'):
            # Section headers
            formatted_lines.append(f"\n**{line}**")
        elif line.startswith(('•', '-', '*')):
            # Bullet points
            formatted_lines.append(f"  {line}")
        elif line[0].isdigit() and line[1] in '.):':
            # Numbered points
            formatted_lines.append(f"  {line}")
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def validate_question(question: str) -> tuple[bool, str]:
    """Validate user question"""
    if not question or len(question.strip()) < 10:
        return False, "Question must be at least 10 characters long"
    
    if len(question) > 1000:
        return False, "Question must be less than 1000 characters"
    
    # Check for minimum word count
    word_count = len(question.split())
    if word_count < 3:
        return False, "Question must contain at least 3 words"
    
    return True, ""

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text for better search"""
    # Simple keyword extraction - can be enhanced with NLP
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'
    }
    
    words = text.lower().split()
    keywords = [w for w in words if len(w) > 3 and w not in stop_words]
    
    return list(set(keywords))[:10]  # Return top 10 unique keywords

def calculate_answer_statistics(answer: str) -> Dict[str, Any]:
    """Calculate statistics for an answer"""
    words = answer.split()
    sentences = answer.split('.')
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'average_sentence_length': len(words) / max(len(sentences), 1),
        'paragraph_count': len([p for p in answer.split('\n\n') if p.strip()])
    }

def save_qa_session(result: Dict[str, Any], filename: str = None) -> str:
    """Save Q&A session to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_session_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Session saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving session: {str(e)}")
        raise

def load_qa_session(filename: str) -> Dict[str, Any]:
    """Load Q&A session from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading session: {str(e)}")
        raise