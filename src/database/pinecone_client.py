from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any, Optional
import logging
import json
from src.config import config

logger = logging.getLogger(__name__)

class PineconeClient:
    """Client for interacting with Pinecone vector database"""
    
    def __init__(self):
        """Initialize Pinecone client"""
        config.validate()
        
        # Initialize Pinecone client
        self.pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Ensure the index exists
        if config.PINECONE_INDEX_NAME not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=config.PINECONE_ENVIRONMENT
                )
            )
        
        # Connect to index
        self.index = self.pinecone_client.Index(config.PINECONE_INDEX_NAME)
    
    def get_board_namespace(self, board: str) -> str:
        """Get namespace for a board"""
        # Normalize board name to create namespace
        namespace = board.upper().replace(" ", "_")
        
        if namespace not in config.SUPPORTED_BOARDS:
            logger.warning(f"Board '{board}' not in supported list. Using as-is.")
        
        return namespace
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI's text-embedding-3-small"""
        try:
            response = self.openai_client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def search_similar_content(
        self, 
        query: str, 
        board: Optional[str] = None,
        top_k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content in Pinecone using board as namespace
        
        Args:
            query: The search query
            board: Board name to use as namespace (e.g., 'CBSE', 'ICSE')
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Use board as namespace if provided
            namespace = self.get_board_namespace(board) if board else None
            
            # Prepare filters
            pinecone_filter = None
            if filter_dict and 'subject' in filter_dict and filter_dict['subject']:
                pinecone_filter = {'subject': filter_dict['subject']}
                logger.info(f"Using filter: {pinecone_filter}")
            
            # Search in Pinecone
            logger.info(f"Querying Pinecone in namespace: {namespace}")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k or config.TOP_K_RESULTS,
                include_metadata=True,
                namespace=namespace,
                filter=pinecone_filter
            )
            
            logger.info(f"Got {len(results.get('matches', []))} results from Pinecone")
            
            # Process results - your data has all the required fields
            processed_results = []
            
            for match in results.get('matches', []):
                score = match.get('score', 0)
                
                # Log scores for debugging
                logger.debug(f"Result score: {score}, threshold: {config.SIMILARITY_THRESHOLD}")
                
                # Apply score threshold
                if score >= config.SIMILARITY_THRESHOLD:
                    metadata = match.get('metadata', {})
                    
                    result = {
                        'id': match.get('id', 'unknown'),
                        'score': score,
                        'metadata': metadata  # Use metadata as-is since it has all required fields
                    }
                    
                    processed_results.append(result)
                else:
                    logger.debug(f"Skipping result with score {score} < {config.SIMILARITY_THRESHOLD}")
            
            logger.info(f"Returning {len(processed_results)} results after filtering")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise