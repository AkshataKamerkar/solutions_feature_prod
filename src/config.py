import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "textbook-unified-index")
    PINECONE_DIMENSION: int = 1536  # OpenAI text-embedding-3-small dimension
    
    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Updated to match your implementation
    LLM_MODEL: str = "gpt-4"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
    
    # Search Configuration
    TOP_K_RESULTS: int = 2
    SIMILARITY_THRESHOLD: float = 0.3
    
    # Supported boards
    SUPPORTED_BOARDS = ["CBSE", "ICSE", "STATE_BOARD", "IB", "CAMBRIDGE"]
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        required_vars = [
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_ENVIRONMENT"
        ]
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

config = Config()