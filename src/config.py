# src/config/config.py

import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Enhanced configuration for CBSE Educational Service"""

    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cbse-grade10-content")
    PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

    # Answer Refinement Configuration
    REFINEMENT_TEMPERATURE = float(os.getenv("REFINEMENT_TEMPERATURE", "0.15"))
    REFINEMENT_MAX_TOKENS = int(os.getenv("REFINEMENT_MAX_TOKENS", "1500"))
    REFINEMENT_REQUEST_TIMEOUT = int(os.getenv("REFINEMENT_REQUEST_TIMEOUT", "45"))

    # Search Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # Board and Subject Configuration
    SUPPORTED_BOARDS = ["CBSE", "ICSE", "SSC", "MAHARASHTRA", "KARNATAKA", "TAMILNADU", "WESTBENGAL"]
    SUPPORTED_SUBJECTS = [
        "Physics", "Chemistry", "Biology", "Science", "Mathematics",
        "History", "Geography", "Political Science", "Economics", "Social Science",
        "English", "Hindi"
    ]

    # CBSE Grade Configuration
    SUPPORTED_GRADES = list(range(1, 13))  # Grades 1-12
    DEFAULT_GRADE = int(os.getenv("DEFAULT_GRADE", "10"))

    # CBSE-specific settings
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_WORD_COUNT = int(os.getenv("MAX_WORD_COUNT", "350"))  # For 5-mark questions
    MIN_WORD_COUNT = int(os.getenv("MIN_WORD_COUNT", "15"))  # For 1-mark questions

    # Quality thresholds
    MIN_CBSE_COMPLIANCE_SCORE = float(os.getenv("MIN_CBSE_COMPLIANCE_SCORE", "0.8"))
    MIN_EVALUATION_SCORE = float(os.getenv("MIN_EVALUATION_SCORE", "0.7"))
    MIN_QUALITY_SCORE = float(os.getenv("MIN_QUALITY_SCORE", "0.6"))

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE = os.getenv("LOG_FILE", "cbse_educational_service.log")

    # Database Configuration (if needed)
    DATABASE_URL = os.getenv("DATABASE_URL")
    DATABASE_MAX_CONNECTIONS = int(os.getenv("DATABASE_MAX_CONNECTIONS", "10"))

    # Cache Configuration
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default

    # Rate Limiting Configuration
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

    # Feature Flags
    ENABLE_ANSWER_REFINEMENT = os.getenv("ENABLE_ANSWER_REFINEMENT", "true").lower() == "true"
    ENABLE_QUALITY_EVALUATION = os.getenv("ENABLE_QUALITY_EVALUATION", "true").lower() == "true"
    ENABLE_CBSE_COMPLIANCE_CHECK = os.getenv("ENABLE_CBSE_COMPLIANCE_CHECK", "true").lower() == "true"
    ENABLE_BATCH_PROCESSING = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"

    # Development/Production Settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    VERSION = os.getenv("VERSION", "1.0.0")

    @classmethod
    def validate(cls) -> Dict[str, bool]:
        """Validate configuration"""
        validation = {
            "pinecone_api_key": bool(cls.PINECONE_API_KEY),
            "openai_api_key": bool(cls.OPENAI_API_KEY),
            "pinecone_environment": bool(cls.PINECONE_ENVIRONMENT),
            "pinecone_index": bool(cls.PINECONE_INDEX_NAME)
        }

        missing_configs = [key for key, value in validation.items() if not value]
        if missing_configs:
            raise ValueError(f"Missing required configuration: {', '.join(missing_configs)}")

        return validation

    @classmethod
    def get_word_limits(cls, marks: int) -> Dict[str, int]:
        """Get word limits for marks (Basic limits - for detailed CBSE limits use AnswerRefinementAgent)"""
        limits = {
            1: {"min": 15, "max": 40, "target": 25},
            2: {"min": 30, "max": 50, "target": 40},
            3: {"min": 50, "max": 80, "target": 65},
            4: {"min": 150, "max": 250, "target": 200},
            5: {"min": 250, "max": 350, "target": 300}
        }
        return limits.get(marks, limits[1])

    @classmethod
    def get_subject_sections(cls, subject: str) -> List[str]:
        """Get sections for a subject"""
        sections = {
            "English": ["literature", "grammar", "writing", "reading"],
            "Hindi": ["साहित्य", "व्याकरण", "रचनात्मक लेखन", "अपठित बोध"],
            "Science": ["physics", "chemistry", "biology"],
            "Social Science": ["history", "geography", "political_science", "economics"],
            "Mathematics": ["general"],
            "Physics": ["general"],
            "Chemistry": ["general"],
            "Biology": ["general"],
            "History": ["general"],
            "Geography": ["general"],
            "Political Science": ["general"],
            "Economics": ["general"]
        }
        return sections.get(subject, ["general"])

    @classmethod
    def get_question_types(cls) -> List[str]:
        """Get supported question types"""
        return [
            "mcq",
            "subjective",
            "fill_blank",
            "true_false",
            "error_correction",
            "reporting",
            "sub_question"
        ]

    @classmethod
    def get_refinement_strategies(cls) -> List[str]:
        """Get available refinement strategies"""
        return [
            "extract_exact",
            "heavy_compression",
            "moderate_compression",
            "optimize_steps",
            "expand_content",
            "minor_expansion",
            "optimize_precision"
        ]

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return cls.ENVIRONMENT.lower() == "production"

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment"""
        return cls.ENVIRONMENT.lower() == "development"

    @classmethod
    def get_config_summary(cls) -> Dict[str, any]:
        """Get a summary of current configuration (excluding sensitive data)"""
        return {
            "environment": cls.ENVIRONMENT,
            "version": cls.VERSION,
            "debug": cls.DEBUG,
            "llm_model": cls.LLM_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "supported_boards": cls.SUPPORTED_BOARDS,
            "supported_subjects": cls.SUPPORTED_SUBJECTS,
            "supported_grades": cls.SUPPORTED_GRADES,
            "default_grade": cls.DEFAULT_GRADE,
            "features": {
                "answer_refinement": cls.ENABLE_ANSWER_REFINEMENT,
                "quality_evaluation": cls.ENABLE_QUALITY_EVALUATION,
                "cbse_compliance": cls.ENABLE_CBSE_COMPLIANCE_CHECK,
                "batch_processing": cls.ENABLE_BATCH_PROCESSING,
                "caching": cls.CACHE_ENABLED,
                "rate_limiting": cls.RATE_LIMIT_ENABLED
            },
            "thresholds": {
                "cbse_compliance_score": cls.MIN_CBSE_COMPLIANCE_SCORE,
                "evaluation_score": cls.MIN_EVALUATION_SCORE,
                "quality_score": cls.MIN_QUALITY_SCORE,
                "similarity_threshold": cls.SIMILARITY_THRESHOLD
            }
        }

    @classmethod
    def validate_subject_and_grade(cls, subject: str, grade: int) -> bool:
        """Validate if subject and grade combination is supported"""
        if subject not in cls.SUPPORTED_SUBJECTS:
            return False
        if grade not in cls.SUPPORTED_GRADES:
            return False
        return True

    @classmethod
    def get_default_context(cls, subject: str, grade: int = None) -> Dict[str, any]:
        """Get default context for a subject and grade"""
        grade = grade or cls.DEFAULT_GRADE

        return {
            "board": "CBSE",
            "subject": subject,
            "grade": grade,
            "sections": cls.get_subject_sections(subject),
            "question_types": cls.get_question_types(),
            "word_limits": {
                1: cls.get_word_limits(1),
                2: cls.get_word_limits(2),
                3: cls.get_word_limits(3),
                4: cls.get_word_limits(4),
                5: cls.get_word_limits(5)
            }
        }


# Create a global config instance for backward compatibility
config = Config()

# Validate configuration on import (optional - can be commented out for production)
if config.DEBUG:
    try:
        config.validate()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"⚠️ Configuration validation warning: {e}")