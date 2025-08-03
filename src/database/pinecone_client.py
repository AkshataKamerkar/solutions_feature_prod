# src/database/pinecone_client.py

from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any, Optional
import logging
import json
import os

# Updated config import - works with both Config and CBSEConfig classes
try:
    from src.config import config
except ImportError:
    # Fallback for CBSEConfig structure
    from src.config import config

logger = logging.getLogger(__name__)


class PineconeClient:
    """Enhanced client for interacting with Pinecone vector database"""

    def __init__(self):
        """Initialize Pinecone client with enhanced error handling"""
        try:
            # Validate configuration
            self._validate_config()

            # Initialize Pinecone client
            self.pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)

            # Initialize OpenAI client
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

            # Get dimension from config or default
            dimension = getattr(config, 'PINECONE_DIMENSION', 1536)

            # Ensure the index exists
            existing_indexes = self.pinecone_client.list_indexes().names()
            if config.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {config.PINECONE_INDEX_NAME}")
                self.pinecone_client.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=config.PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"Successfully created index: {config.PINECONE_INDEX_NAME}")
            else:
                logger.info(f"Using existing Pinecone index: {config.PINECONE_INDEX_NAME}")

            # Connect to index
            self.index = self.pinecone_client.Index(config.PINECONE_INDEX_NAME)

            # Get embedding model from config
            self.embedding_model = getattr(config, 'EMBEDDING_MODEL', 'text-embedding-ada-002')

            logger.info("PineconeClient initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PineconeClient: {str(e)}")
            raise Exception(f"PineconeClient initialization failed: {str(e)}")

    def _validate_config(self):
        """Validate required configuration parameters"""
        required_configs = {
            'PINECONE_API_KEY': config.PINECONE_API_KEY,
            'OPENAI_API_KEY': config.OPENAI_API_KEY,
            'PINECONE_ENVIRONMENT': config.PINECONE_ENVIRONMENT,
            'PINECONE_INDEX_NAME': config.PINECONE_INDEX_NAME
        }

        missing_configs = [key for key, value in required_configs.items() if not value]
        if missing_configs:
            raise ValueError(f"Missing required configuration parameters: {', '.join(missing_configs)}")

        logger.info("Configuration validation successful")

    def get_board_namespace(self, board: str) -> str:
        """Get namespace for a board with enhanced validation"""
        if not board:
            return None

        # Normalize board name to create namespace
        namespace = board.upper().replace(" ", "_").replace("-", "_")

        # Get supported boards from config
        supported_boards = getattr(config, 'SUPPORTED_BOARDS', [
            'CBSE', 'ICSE', 'SSC', 'MAHARASHTRA', 'KARNATAKA', 'TAMILNADU', 'WESTBENGAL'
        ])

        if namespace not in supported_boards:
            logger.warning(f"Board '{board}' not in supported list: {supported_boards}. Using as-is.")

        return namespace

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI's embedding model with error handling"""
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")

            # Clean and prepare text
            cleaned_text = text.strip()
            if len(cleaned_text) > 8000:  # OpenAI token limit consideration
                cleaned_text = cleaned_text[:8000]
                logger.warning("Text truncated to 8000 characters for embedding")

            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text,
                encoding_format="float"
            )

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")

    def search_similar_content(
            self,
            query: str,
            board: Optional[str] = None,
            top_k: int = None,
            filter_dict: Optional[Dict[str, Any]] = None,
            include_metadata: bool = True,
            include_values: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search for similar content in Pinecone with comprehensive error handling

        Args:
            query: The search query
            board: Board name to use as namespace (e.g., 'CBSE', 'ICSE')
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results

        Returns:
            List of matching documents with metadata and scores
        """
        try:
            # Input validation
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            # Get configuration values
            top_k = top_k or getattr(config, 'TOP_K_RESULTS', 5)
            similarity_threshold = getattr(config, 'SIMILARITY_THRESHOLD', 0.3)

            # Validate top_k
            if top_k <= 0 or top_k > 100:
                raise ValueError("top_k must be between 1 and 100")

            logger.info(f"Searching for query: '{query[:50]}...' with top_k={top_k}")

            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Use board as namespace if provided
            namespace = self.get_board_namespace(board) if board else None

            # Enhanced filter preparation
            pinecone_filter = None
            if filter_dict:
                # Clean and validate filters
                cleaned_filters = {}
                for key, value in filter_dict.items():
                    if value is not None and str(value).strip():
                        cleaned_filters[key] = value

                if cleaned_filters:
                    pinecone_filter = cleaned_filters
                    logger.info(f"Using filters: {pinecone_filter}")

            # Search in Pinecone with enhanced parameters
            logger.info(f"Querying Pinecone - Namespace: {namespace}, Filters: {pinecone_filter}")

            search_params = {
                'vector': query_embedding,
                'top_k': top_k,
                'include_metadata': include_metadata,
                'include_values': include_values
            }

            if namespace:
                search_params['namespace'] = namespace

            if pinecone_filter:
                search_params['filter'] = pinecone_filter

            results = self.index.query(**search_params)

            raw_matches = results.get('matches', [])
            logger.info(f"Pinecone returned {len(raw_matches)} raw results")

            # Enhanced result processing
            processed_results = []

            for match in raw_matches:
                try:
                    score = match.get('score', 0.0)

                    # Log scores for debugging
                    logger.debug(f"Processing match with score: {score}, threshold: {similarity_threshold}")

                    # Apply score threshold
                    if score >= similarity_threshold:
                        metadata = match.get('metadata', {})

                        # Enhanced metadata validation and cleaning
                        cleaned_metadata = self._clean_metadata(metadata)

                        result = {
                            'id': match.get('id', 'unknown'),
                            'score': score,
                            'metadata': cleaned_metadata
                        }

                        # Include values if requested
                        if include_values and 'values' in match:
                            result['values'] = match['values']

                        processed_results.append(result)
                        logger.debug(f"Added result: {result['id']} (score: {score:.3f})")
                    else:
                        logger.debug(f"Filtered out result with score {score:.3f} < {similarity_threshold}")

                except Exception as e:
                    logger.warning(f"Error processing match: {str(e)}")
                    continue

            # Sort by score (highest first)
            processed_results.sort(key=lambda x: x['score'], reverse=True)

            logger.info(f"Returning {len(processed_results)} results after processing and filtering")

            # Log top results for debugging
            for i, result in enumerate(processed_results[:3]):
                metadata = result['metadata']
                chapter = metadata.get('chapter', metadata.get('chapter_name', 'Unknown'))
                section = metadata.get('section_type', metadata.get('concept_title', 'Unknown'))
                logger.debug(f"Top result {i + 1}: {chapter} - {section} (score: {result['score']:.3f})")

            return processed_results

        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise Exception(f"Pinecone search failed: {str(e)}")

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize metadata from Pinecone results"""
        if not metadata:
            return {}

        cleaned = {}

        # Handle different metadata field variations
        field_mappings = {
            'chapter': ['chapter', 'chapter_name'],
            'subject': ['subject', 'subject_name'],
            'board': ['board', 'board_name'],
            'type': ['type', 'content_type'],
            'section_type': ['section_type', 'concept_title'],
            'indexed': ['indexed', 'created_at', 'timestamp'],
            'summary': ['summary', 'summary_text', 'content'],
            'keywords': ['keywords', 'tags']
        }

        for standard_field, possible_fields in field_mappings.items():
            for field in possible_fields:
                if field in metadata and metadata[field]:
                    cleaned[standard_field] = metadata[field]
                    break

        # Handle keywords specially (ensure it's a string)
        if 'keywords' in cleaned:
            keywords = cleaned['keywords']
            if isinstance(keywords, list):
                cleaned['keywords'] = ', '.join(str(k) for k in keywords)
            else:
                cleaned['keywords'] = str(keywords)

        # Ensure required fields have defaults
        defaults = {
            'chapter': 'Unknown Chapter',
            'subject': 'Unknown Subject',
            'board': 'Unknown Board',
            'section_type': 'Unknown Section',
            'summary': 'No summary available',
            'keywords': ''
        }

        for field, default_value in defaults.items():
            if field not in cleaned or not cleaned[field]:
                cleaned[field] = default_value

        return cleaned

    def upsert_content(self, vectors: List[Dict[str, Any]], namespace: Optional[str] = None) -> bool:
        """
        Upsert content vectors to Pinecone

        Args:
            vectors: List of vector dictionaries with id, values, and metadata
            namespace: Optional namespace for the vectors

        Returns:
            Boolean indicating success
        """
        try:
            if not vectors:
                raise ValueError("Vectors list cannot be empty")

            logger.info(f"Upserting {len(vectors)} vectors to namespace: {namespace}")

            # Validate vector format
            for i, vector in enumerate(vectors):
                if 'id' not in vector or 'values' not in vector:
                    raise ValueError(f"Vector {i} missing required 'id' or 'values' field")

            # Upsert in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]

                upsert_params = {'vectors': batch}
                if namespace:
                    upsert_params['namespace'] = namespace

                self.index.upsert(**upsert_params)
                logger.debug(f"Upserted batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True

        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False

    def delete_content(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """
        Delete content from Pinecone

        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace

        Returns:
            Boolean indicating success
        """
        try:
            if not ids:
                raise ValueError("IDs list cannot be empty")

            logger.info(f"Deleting {len(ids)} vectors from namespace: {namespace}")

            delete_params = {'ids': ids}
            if namespace:
                delete_params['namespace'] = namespace

            self.index.delete(**delete_params)

            logger.info(f"Successfully deleted {len(ids)} vectors")
            return True

        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False

    def get_index_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index

        Args:
            namespace: Optional namespace to get stats for

        Returns:
            Dictionary with index statistics
        """
        try:
            stats_params = {}
            if namespace:
                stats_params['filter'] = {}  # Empty filter to get namespace stats

            stats = self.index.describe_index_stats()

            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0.0),
                'namespaces': stats.get('namespaces', {}),
                'namespace': namespace,
                'namespace_vector_count': stats.get('namespaces', {}).get(namespace, {}).get('vector_count',
                                                                                             0) if namespace else None
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {'error': str(e)}

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Pinecone and OpenAI services

        Returns:
            Dictionary with connection test results
        """
        results = {
            'pinecone_connection': False,
            'openai_connection': False,
            'index_accessible': False,
            'embedding_generation': False,
            'overall_status': 'failed'
        }

        try:
            # Test Pinecone connection
            try:
                indexes = self.pinecone_client.list_indexes()
                results['pinecone_connection'] = True
                logger.info("Pinecone connection: OK")
            except Exception as e:
                logger.error(f"Pinecone connection failed: {str(e)}")

            # Test index access
            try:
                stats = self.index.describe_index_stats()
                results['index_accessible'] = True
                logger.info("Index access: OK")
            except Exception as e:
                logger.error(f"Index access failed: {str(e)}")

            # Test OpenAI connection and embedding generation
            try:
                test_embedding = self.generate_embedding("test connection")
                if test_embedding and len(test_embedding) > 0:
                    results['openai_connection'] = True
                    results['embedding_generation'] = True
                    logger.info("OpenAI connection and embedding generation: OK")
            except Exception as e:
                logger.error(f"OpenAI connection/embedding failed: {str(e)}")

            # Overall status
            if all([results['pinecone_connection'], results['openai_connection'],
                    results['index_accessible'], results['embedding_generation']]):
                results['overall_status'] = 'healthy'
                logger.info("Overall connection test: PASSED")
            else:
                results['overall_status'] = 'degraded'
                logger.warning("Overall connection test: PARTIAL")

        except Exception as e:
            logger.error(f"Connection test error: {str(e)}")
            results['error'] = str(e)

        return results

    def search_by_metadata(self, filters: Dict[str, Any],
                           board: Optional[str] = None,
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search content by metadata filters only (without vector similarity)

        Args:
            filters: Metadata filters to apply
            board: Optional board namespace
            top_k: Number of results to return

        Returns:
            List of matching documents
        """
        try:
            # Use a zero vector for metadata-only search
            zero_vector = [0.0] * getattr(config, 'PINECONE_DIMENSION', 1536)

            namespace = self.get_board_namespace(board) if board else None

            search_params = {
                'vector': zero_vector,
                'top_k': top_k,
                'include_metadata': True,
                'filter': filters
            }

            if namespace:
                search_params['namespace'] = namespace

            results = self.index.query(**search_params)

            processed_results = []
            for match in results.get('matches', []):
                result = {
                    'id': match.get('id', 'unknown'),
                    'metadata': self._clean_metadata(match.get('metadata', {}))
                }
                processed_results.append(result)

            logger.info(f"Metadata search returned {len(processed_results)} results")
            return processed_results

        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            return []

    def __del__(self):
        """Cleanup when the client is destroyed"""
        try:
            # Clean up connections if needed
            pass
        except:
            pass


# Utility functions for easier usage

def create_pinecone_client() -> PineconeClient:
    """Factory function to create a PineconeClient instance"""
    return PineconeClient()


def test_pinecone_setup() -> bool:
    """Test if Pinecone is properly set up"""
    try:
        client = PineconeClient()
        test_results = client.test_connection()
        return test_results['overall_status'] == 'healthy'
    except Exception as e:
        logger.error(f"Pinecone setup test failed: {str(e)}")
        return False


# Export main components
# Export main components
__all__ = [
    'PineconeClient',
    'create_pinecone_client',
    'test_pinecone_setup'
]

# Example usage and testing
if __name__ == "__main__":
    """
    Example usage and testing of the PineconeClient
    """
    import time


    def run_pinecone_tests():
        """Run comprehensive tests of the PineconeClient"""

        print("🚀 Starting PineconeClient Tests")
        print("=" * 50)

        try:
            # Test 1: Initialize client
            print("Test 1: Initializing PineconeClient...")
            client = PineconeClient()
            print("✅ PineconeClient initialized successfully")

            # Test 2: Connection test
            print("\nTest 2: Testing connections...")
            connection_results = client.test_connection()
            print(f"Connection Status: {connection_results['overall_status']}")
            for service, status in connection_results.items():
                if service != 'overall_status':
                    status_icon = "✅" if status else "❌"
                    print(f"  {status_icon} {service}: {status}")

            # Test 3: Index statistics
            print("\nTest 3: Getting index statistics...")
            stats = client.get_index_stats()
            if 'error' not in stats:
                print(f"✅ Index Stats:")
                print(f"  • Total vectors: {stats.get('total_vector_count', 0)}")
                print(f"  • Dimension: {stats.get('dimension', 0)}")
                print(f"  • Index fullness: {stats.get('index_fullness', 0):.2%}")
                print(f"  • Namespaces: {len(stats.get('namespaces', {}))}")
            else:
                print(f"❌ Failed to get stats: {stats['error']}")

            # Test 4: Embedding generation
            print("\nTest 4: Testing embedding generation...")
            test_text = "This is a test for embedding generation in the educational system."
            embedding = client.generate_embedding(test_text)
            print(f"✅ Generated embedding with dimension: {len(embedding)}")

            # Test 5: Search functionality
            print("\nTest 5: Testing search functionality...")
            search_query = "laws of reflection of light"
            search_results = client.search_similar_content(
                query=search_query,
                board="CBSE",
                top_k=3
            )
            print(f"✅ Search returned {len(search_results)} results for: '{search_query}'")

            # Display top search results
            for i, result in enumerate(search_results[:2], 1):
                metadata = result['metadata']
                print(f"  Result {i}:")
                print(f"    • Score: {result['score']:.3f}")
                print(f"    • Chapter: {metadata.get('chapter', 'Unknown')}")
                print(f"    • Subject: {metadata.get('subject', 'Unknown')}")
                print(f"    • Section: {metadata.get('section_type', 'Unknown')}")

            # Test 6: Filter search
            print("\nTest 6: Testing filtered search...")
            filtered_results = client.search_similar_content(
                query="photosynthesis",
                board="CBSE",
                filter_dict={"subject": "Biology"},
                top_k=2
            )
            print(f"✅ Filtered search returned {len(filtered_results)} Biology results")

            # Test 7: Namespace functionality
            print("\nTest 7: Testing namespace functionality...")
            test_boards = ["CBSE", "ICSE", "STATE_BOARD"]
            for board in test_boards:
                namespace = client.get_board_namespace(board)
                print(f"  • {board} → {namespace}")

            print(f"\n🎉 All tests completed successfully!")
            print("=" * 50)
            return True

        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            print("=" * 50)
            return False


    def demo_educational_search():
        """Demonstrate educational content search"""

        print("\n📚 Educational Content Search Demo")
        print("=" * 40)

        try:
            client = PineconeClient()

            # Sample educational queries
            demo_queries = [
                {
                    "query": "What is photosynthesis?",
                    "board": "CBSE",
                    "subject": "Biology",
                    "marks": 3
                },
                {
                    "query": "Laws of reflection of light",
                    "board": "CBSE",
                    "subject": "Physics",
                    "marks": 2
                },
                {
                    "query": "Quadratic equations solving methods",
                    "board": "CBSE",
                    "subject": "Mathematics",
                    "marks": 5
                }
            ]

            for i, demo in enumerate(demo_queries, 1):
                print(f"\nDemo Query {i}: {demo['query']}")
                print(f"Board: {demo['board']} | Subject: {demo['subject']} | Marks: {demo['marks']}")
                print("-" * 40)

                # Search for content
                results = client.search_similar_content(
                    query=demo['query'],
                    board=demo['board'],
                    filter_dict={"subject": demo['subject']} if demo['subject'] else None,
                    top_k=3
                )

                if results:
                    print(f"Found {len(results)} relevant results:")
                    for j, result in enumerate(results, 1):
                        metadata = result['metadata']
                        print(f"  {j}. {metadata.get('chapter', 'Unknown Chapter')}")
                        print(f"     Section: {metadata.get('section_type', 'Unknown')}")
                        print(f"     Score: {result['score']:.3f}")
                        summary = metadata.get('summary', 'No summary')
                        print(f"     Summary: {summary[:100]}...")
                        print()
                else:
                    print("  No results found")

                time.sleep(1)  # Small delay between demos

            print("🎉 Demo completed!")

        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")


    def benchmark_search_performance():
        """Benchmark search performance"""

        print("\n⚡ Search Performance Benchmark")
        print("=" * 35)

        try:
            client = PineconeClient()

            test_queries = [
                "photosynthesis process in plants",
                "Newton's laws of motion",
                "quadratic equation formula",
                "water cycle explanation",
                "periodic table elements"
            ]

            total_time = 0
            successful_searches = 0

            for i, query in enumerate(test_queries, 1):
                print(f"Search {i}/5: {query[:30]}...")

                start_time = time.time()
                try:
                    results = client.search_similar_content(
                        query=query,
                        board="CBSE",
                        top_k=5
                    )
                    end_time = time.time()

                    search_time = end_time - start_time
                    total_time += search_time
                    successful_searches += 1

                    print(f"  ✅ {len(results)} results in {search_time:.2f}s")

                except Exception as e:
                    print(f"  ❌ Failed: {str(e)}")

                time.sleep(0.5)  # Small delay between searches

            if successful_searches > 0:
                avg_time = total_time / successful_searches
                print(f"\n📊 Performance Summary:")
                print(f"  • Successful searches: {successful_searches}/{len(test_queries)}")
                print(f"  • Average search time: {avg_time:.2f}s")
                print(f"  • Total time: {total_time:.2f}s")

                if avg_time < 2.0:
                    print("  🚀 Excellent performance!")
                elif avg_time < 5.0:
                    print("  ✅ Good performance")
                else:
                    print("  ⚠️  Consider optimization")

        except Exception as e:
            print(f"❌ Benchmark failed: {str(e)}")


    # Run all tests and demos
    print("🔧 PineconeClient Comprehensive Testing Suite")
    print("=" * 60)

    # Check if we should run tests (based on environment or user input)
    run_tests = os.getenv('RUN_PINECONE_TESTS', 'false').lower() == 'true'

    if run_tests:
        # Run basic functionality tests
        test_success = run_pinecone_tests()

        if test_success:
            # Run educational content demo
            demo_educational_search()

            # Run performance benchmark
            benchmark_search_performance()

        print("\n✅ Testing suite completed!")
    else:
        print("💡 Set environment variable RUN_PINECONE_TESTS=true to run tests")
        print("   Example: export RUN_PINECONE_TESTS=true && python pinecone_client.py")

        # Show configuration info instead
        print(f"\n📋 Current Configuration:")
        print(f"  • Index Name: {getattr(config, 'PINECONE_INDEX_NAME', 'Not set')}")
        print(f"  • Environment: {getattr(config, 'PINECONE_ENVIRONMENT', 'Not set')}")
        print(f"  • Embedding Model: {getattr(config, 'EMBEDDING_MODEL', 'text-embedding-ada-002')}")
        print(f"  • Top K Results: {getattr(config, 'TOP_K_RESULTS', 5)}")
        print(f"  • Similarity Threshold: {getattr(config, 'SIMILARITY_THRESHOLD', 0.7)}")

        # Test basic connection
        try:
            print(f"\n🔍 Testing basic connection...")
            test_result = test_pinecone_setup()
            if test_result:
                print("✅ Basic connection test passed!")
            else:
                print("❌ Basic connection test failed!")
        except Exception as e:
            print(f"❌ Connection test error: {str(e)}")