# pinecone_dashboard.py
import streamlit as st
import pinecone
from pinecone import Pinecone
import pandas as pd
import json
import numpy as np
import os
import certifi
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Pinecone DB Viewer",
    layout="wide",
    page_icon="📌"
)

# Set the certificate bundle path
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


# Initialize Pinecone and OpenAI
def init_services():
    # Get API keys from environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not pinecone_api_key:
        st.error("PINECONE_API_KEY not found. Set it in .env file or Streamlit secrets.")
        st.stop()

    try:
        pc = Pinecone(api_key=pinecone_api_key)
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        st.stop()

    # Initialize OpenAI if key is available
    openai_client = None
    if openai_api_key:
        try:
            # Clean initialization without proxy arguments
            openai_client = OpenAI(
                api_key=openai_api_key,
                # Remove any proxy-related arguments
            )
        except Exception as e:

            print(f'Method 1 - is not Working {str(e)}')
            # Try alternative initialization if needed
            try:
                # If you need to handle proxies, do it at the HTTP level
                import httpx

                # Create custom HTTP client if needed
                http_client = httpx.Client()

                openai_client = OpenAI(
                    api_key=openai_api_key,
                    http_client=http_client
                )
            except:
                st.warning("OpenAI initialization failed. Text search will not be available.")
    else:
        st.warning("OpenAI API key not found. Text search will not be available.")

    return pc, openai_client


def generate_embedding(text: str, openai_client):
    """Generate embedding using OpenAI"""
    if not openai_client:
        return None

    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None


def format_metadata_display(metadata):
    """Format metadata for display"""
    # Chapter/Section information
    chunk_type = metadata.get('chunk_type', 'unknown')

    if chunk_type == 'section':
        return {
            'Type': 'Section',
            'Concept': metadata.get('concept_title', 'N/A'),
            'Chapter': metadata.get('chapter_name', 'N/A'),
            'Subject': metadata.get('subject', 'N/A'),
            'Board': metadata.get('board', 'N/A'),
            'Section Type': metadata.get('section_type', 'N/A'),
            'Summary': metadata.get('summary_text', 'N/A')[:200] + '...' if metadata.get('summary_text', '') else 'N/A',
            'Keywords': metadata.get('keywords', 'N/A'),
            'Indexed': metadata.get('indexed_at', 'N/A')[:19] if metadata.get('indexed_at') else 'N/A'
        }
    else:
        return {
            'Type': 'Chapter Summary',
            'Chapter': metadata.get('chapter_name', 'N/A'),
            'Subject': metadata.get('subject', 'N/A'),
            'Board': metadata.get('board', 'N/A'),
            'Summary': metadata.get('summary_text', 'N/A')[:200] + '...' if metadata.get('summary_text', '') else 'N/A',
            'Keywords': metadata.get('keywords', 'N/A'),
            'Indexed': metadata.get('indexed_at', 'N/A')[:19] if metadata.get('indexed_at') else 'N/A'
        }


def stats_to_dict(stats_response):
    """Convert Pinecone stats response to dictionary"""
    stats_dict = {
        'total_vector_count': 0,
        'dimension': 0,
        'index_fullness': 0,
        'namespaces': {}
    }

    # Safely extract values
    if hasattr(stats_response, 'total_vector_count'):
        stats_dict['total_vector_count'] = stats_response.total_vector_count or 0
    if hasattr(stats_response, 'dimension'):
        stats_dict['dimension'] = stats_response.dimension or 0
    if hasattr(stats_response, 'index_fullness'):
        stats_dict['index_fullness'] = stats_response.index_fullness or 0

    # Handle namespaces
    if hasattr(stats_response, 'namespaces') and stats_response.namespaces:
        for ns_name, ns_info in stats_response.namespaces.items():
            stats_dict['namespaces'][ns_name] = {
                'vector_count': ns_info.vector_count if hasattr(ns_info, 'vector_count') else 0
            }

    return stats_dict


# Custom CSS
st.markdown("""
<style>
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.title("📌 Pinecone Database Viewer")
st.markdown("### Explore and manage your Pinecone vector database")

pc, openai_client = init_services()

# Sidebar for index and namespace selection
with st.sidebar:
    st.header("⚙ Configuration")

    # List all indexes
    try:
        indexes = pc.list_indexes().names()
        if not indexes:
            st.error("No indexes found in Pinecone")
            st.stop()

        selected_index = st.selectbox("🗂 Select Index", indexes)
    except Exception as e:
        st.error(f"Error listing indexes: {str(e)}")
        st.stop()

    if selected_index:
        try:
            index = pc.Index(selected_index)
            stats_response = index.describe_index_stats()
            stats = stats_to_dict(stats_response)
        except Exception as e:
            st.error(f"Error connecting to index: {str(e)}")
            st.stop()

        st.markdown("### 📊 Index Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Vectors", f"{stats.get('total_vector_count', 0):,}")
        with col2:
            st.metric("Dimensions", stats.get('dimension', 0))

        # Namespace selection
        namespaces = ["No namespace (default)"]
        if stats.get('namespaces'):
            namespaces.extend(sorted(list(stats['namespaces'].keys())))

        selected_namespace = st.selectbox("📁 Select Namespace", namespaces)

        # Convert selection to actual namespace value
        namespace_value = None if selected_namespace == "No namespace (default)" else selected_namespace

        # Show namespace statistics
        if stats.get('namespaces'):
            st.markdown("### 📈 Namespace Statistics")
            for ns, info in stats['namespaces'].items():
                st.info(f"{ns}: {info.get('vector_count', 0):,} vectors")

# Main content
if selected_index:
    index = pc.Index(selected_index)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Search", "📄 Fetch by ID", "📊 Browse", "📈 Statistics", "🗑 Management"])

    with tab1:
        st.header("🔍 Search Vectors")

        if openai_client:
            search_method = st.radio(
                "Search Method",
                ["Text Search", "Random Vector (Demo)"],
                horizontal=True
            )
        else:
            search_method = "Random Vector (Demo)"
            st.info("Text search is disabled. OpenAI API key not configured.")

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            if search_method == "Text Search":
                query_text = st.text_input("🔎 Search Query",
                                           placeholder="e.g., Ohm's Law, photosynthesis, quadratic equations")
            else:
                query_text = st.text_input("🔎 Search Query (ignored for random)", placeholder="Any text")
        with col2:
            top_k = st.number_input("Top K Results", min_value=1, max_value=100, value=10)
        with col3:
            include_summary = st.checkbox("Show summaries", value=True)

        # Add filters
        with st.expander("🔧 Advanced Filters"):
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                filter_subject = st.text_input("Filter by Subject (optional)", placeholder="e.g., Physics")
            with filter_col2:
                filter_chunk_type = st.selectbox("Chunk Type", ["All", "section", "summary"])

        if st.button("🔍 Search", type="primary", use_container_width=True):
            if search_method == "Text Search" and openai_client:
                # Generate embedding from text
                with st.spinner("Generating embedding..."):
                    query_vector = generate_embedding(query_text, openai_client)
                    if not query_vector:
                        st.error("Failed to generate embedding. Check your OpenAI API key.")
                        st.stop()
            else:
                # Use random vector for demo
                query_vector = np.random.rand(stats.get('dimension', 1536)).tolist()

            # Build filter
            filter_dict = {}
            if filter_subject:
                filter_dict['subject'] = filter_subject
            if filter_chunk_type != "All":
                filter_dict['chunk_type'] = filter_chunk_type

            # Search with namespace
            with st.spinner("Searching..."):
                try:
                    results = index.query(
                        vector=query_vector,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=namespace_value,
                        filter=filter_dict if filter_dict else None
                    )
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    st.stop()

            st.subheader(f"Search Results in '{selected_namespace}'")

            if results['matches']:
                st.success(f"Found {len(results['matches'])} results")

                for i, match in enumerate(results['matches']):
                    metadata = match.get('metadata', {})
                    formatted_meta = format_metadata_display(metadata)

                    score_percentage = match['score'] * 100
                    score_color = "🟢" if score_percentage > 70 else "🟡" if score_percentage > 50 else "🔴"

                    with st.expander(
                            f"{score_color} Result {i + 1} - {formatted_meta['Type']}: {formatted_meta.get('Concept', formatted_meta.get('Chapter', 'Unknown'))} (Score: {score_percentage:.1f}%)"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("*Chapter:*", formatted_meta['Chapter'])
                            st.write("*Subject:*", formatted_meta['Subject'])
                            st.write("*Board:*", formatted_meta['Board'])

                        with col2:
                            st.write("*Type:*", formatted_meta['Type'])
                            if formatted_meta.get('Section Type'):
                                st.write("*Section Type:*", formatted_meta['Section Type'])
                            st.write("*Indexed:*", formatted_meta['Indexed'])

                        if include_summary and formatted_meta.get('Summary') != 'N/A':
                            st.markdown("*Summary:*")
                            st.info(formatted_meta['Summary'])

                        if formatted_meta.get('Keywords') != 'N/A':
                            st.markdown("*Keywords:*")
                            st.code(formatted_meta['Keywords'])

                        st.divider()

                        # Show full metadata in a collapsible section (not nested expander)
                        if st.checkbox(f"View Full Metadata for Result {i + 1}", key=f"meta_{i}"):
                            st.json(metadata)

                        st.write("*Vector ID:*", match['id'])
            else:
                st.warning("No results found. Try different search terms or check the namespace.")

    with tab2:
        st.header("📄 Fetch Vector by ID")

        vector_id = st.text_input("🆔 Vector ID", placeholder="e.g., 595373d8631f2c12_section_0")

        col1, col2 = st.columns(2)
        with col1:
            fetch_namespace = st.selectbox(
                "📁 Namespace for Fetch",
                namespaces,
                key="fetch_namespace"
            )
        with col2:
            auto_search = st.checkbox("Auto-search all namespaces if not found", value=True)

        # Convert selection to actual namespace value
        fetch_namespace_value = None if fetch_namespace == "No namespace (default)" else fetch_namespace

        if st.button("🔎 Fetch Vector", type="primary") and vector_id:
            try:
                with st.spinner(f"Fetching from namespace '{fetch_namespace}'..."):
                    result = index.fetch(ids=[vector_id], namespace=fetch_namespace_value)

                if result['vectors'] and vector_id in result['vectors']:
                    st.success(f"✅ Vector found in namespace '{fetch_namespace}'!")

                    vector_data = result['vectors'][vector_id]
                    metadata = vector_data.get('metadata', {})
                    formatted_meta = format_metadata_display(metadata)

                    # Display formatted metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("*Chapter:*", formatted_meta.get('Chapter', 'N/A'))
                        st.write("*Subject:*", formatted_meta.get('Subject', 'N/A'))
                        st.write("*Type:*", formatted_meta.get('Type', 'N/A'))
                    with col2:
                        st.write("*Board:*", formatted_meta.get('Board', 'N/A'))
                        if formatted_meta.get('Concept'):
                            st.write("*Concept:*", formatted_meta['Concept'])
                        st.write("*Indexed:*", formatted_meta.get('Indexed', 'N/A'))

                    if formatted_meta.get('Summary') != 'N/A':
                        st.markdown("*Summary:*")
                        st.info(formatted_meta['Summary'])

                    if formatted_meta.get('Keywords') != 'N/A':
                        st.markdown("*Keywords:*")
                        st.code(formatted_meta['Keywords'])

                    # Show full metadata
                    with st.expander("🔍 View Full Metadata"):
                        st.json(metadata)

                    # Show vector values
                    with st.expander("📊 View Vector Values"):
                        st.write(f"Vector dimension: {len(vector_data.get('values', []))}")
                        st.code(str(vector_data.get('values', [])[:10]) + "... (showing first 10 values)")
                else:
                    st.warning(f"❌ Vector not found in namespace '{fetch_namespace}'")

                    # Try to find in other namespaces
                    if auto_search and len(namespaces) > 1:
                        with st.spinner("Searching in other namespaces..."):
                            found_in = []
                            for ns in namespaces:
                                if ns == fetch_namespace or ns == "No namespace (default)":
                                    continue
                                try:
                                    ns_value = None if ns == "No namespace (default)" else ns
                                    ns_result = index.fetch(ids=[vector_id], namespace=ns_value)
                                    if ns_result['vectors'] and vector_id in ns_result['vectors']:
                                        found_in.append(ns)
                                except:
                                    pass

                            if found_in:
                                st.info(f"🔍 Vector found in namespaces: {', '.join(found_in)}")
                                st.write("Select one of these namespaces and fetch again to see the details.")
                            else:
                                st.error("Vector not found in any namespace")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab3:
        st.header("📊 Browse Vectors")

        st.info("Browse sample vectors from the selected namespace to understand the data structure.")

        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Number of samples", min_value=5, max_value=100, value=20)
        with col2:
            browse_filter = st.selectbox("Filter by type", ["All", "section", "summary"])

        if st.button("📥 Load Sample Vectors", type="primary"):
            # Build filter
            filter_dict = {}
            if browse_filter != "All":
                filter_dict['chunk_type'] = browse_filter

            # Use a random vector to get samples
            with st.spinner("Loading samples..."):
                try:
                    sample_results = index.query(
                        vector=np.random.rand(stats.get('dimension', 1536)).tolist(),
                        top_k=sample_size,
                        include_metadata=True,
                        namespace=namespace_value,
                        filter=filter_dict if filter_dict else None
                    )
                except Exception as e:
                    st.error(f"Error loading samples: {str(e)}")
                    st.stop()

            if sample_results['matches']:
                data = []
                for match in sample_results['matches']:
                    metadata = match.get('metadata', {})

                    # Determine display based on chunk type
                    if metadata.get('chunk_type') == 'section':
                        display_name = metadata.get('concept_title', 'Unknown Section')
                    else:
                        display_name = metadata.get('chapter_name', 'Unknown Chapter')

                    data.append({
                        'ID': match['id'][:20] + '...' if len(match['id']) > 20 else match['id'],
                        'Score': f"{match['score']:.4f}",
                        'Type': metadata.get('chunk_type', 'unknown'),
                        'Name': display_name[:50] + '...' if len(display_name) > 50 else display_name,
                        'Chapter': metadata.get('chapter_name', 'N/A')[:30] + '...' if len(
                            metadata.get('chapter_name', '')) > 30 else metadata.get('chapter_name', 'N/A'),
                        'Subject': metadata.get('subject', 'N/A'),
                        'Board': metadata.get('board', 'N/A'),
                        'Indexed': metadata.get('indexed_at', 'N/A')[:10] if metadata.get('indexed_at') else 'N/A'
                    })

                df = pd.DataFrame(data)

                # Display summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(data))
                with col2:
                    section_count = len([d for d in data if d['Type'] == 'section'])
                    st.metric("Sections", section_count)
                with col3:
                    summary_count = len([d for d in data if d['Type'] == 'summary'])
                    st.metric("Summaries", summary_count)

                # Display dataframe
                st.dataframe(df, use_container_width=True, height=400)

                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"pinecone_samples_{selected_namespace}{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"No vectors found in namespace '{selected_namespace}' with the selected filter.")

    with tab4:
        st.header("📈 Index Statistics")

        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Vectors", f"{stats.get('total_vector_count', 0):,}")
        with col2:
            st.metric("Dimensions", stats.get('dimension', 0))
        with col3:
            st.metric("Namespaces", len(stats.get('namespaces', {})))
        with col4:
            st.metric("Index Fill", f"{(stats.get('total_vector_count', 0) / 1000000) * 100:.1f}%")

        # Namespace breakdown
        if stats.get('namespaces'):
            st.subheader("📊 Vectors by Namespace")

            # Create a bar chart
            ns_data = pd.DataFrame([
                {'Namespace': ns, 'Vectors': info.get('vector_count', 0)}
                for ns, info in stats['namespaces'].items()
            ]).sort_values('Vectors', ascending=False)

            st.bar_chart(ns_data.set_index('Namespace'))

            # Show as table with percentages
            st.subheader("📋 Detailed Breakdown")
            total_vectors = stats.get('total_vector_count', 1)
            ns_data['Percentage'] = (ns_data['Vectors'] / total_vectors * 100).round(2)
            ns_data['Percentage'] = ns_data['Percentage'].astype(str) + '%'

            st.dataframe(ns_data, use_container_width=True)

            # Export stats
            stats_json = json.dumps(stats, indent=2)
            st.download_button(
                label="📥 Download Full Statistics",
                data=stats_json,
                file_name=f"pinecone_stats_{selected_index}{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                mime="application/json"
            )

        # Raw stats
        with st.expander("🔍 View Raw Statistics"):
            st.json(stats)

    with tab5:
        st.header("🗑 Vector Management")
        st.warning("⚠ Be careful with delete operations. They cannot be undone!")

        # Delete by ID
        st.subheader("Delete Single Vector")
        del_vector_id = st.text_input("Vector ID to delete", key="delete_id")
        del_namespace = st.selectbox("Namespace", namespaces, key="delete_namespace")
        del_namespace_value = None if del_namespace == "No namespace (default)" else del_namespace

        col1, col2 = st.columns([1, 3])
        with col1:
            delete_btn = st.button("🗑 Delete Vector", type="secondary")
        with col2:
            if delete_btn:
                st.session_state.show_delete_confirm = True

        if hasattr(st.session_state, 'show_delete_confirm') and st.session_state.show_delete_confirm:
            if del_vector_id:
                confirm = st.checkbox("I confirm I want to delete this vector", key="confirm_single_delete")
                if confirm:
                    try:
                        index.delete(ids=[del_vector_id], namespace=del_namespace_value)
                        st.success(f"✅ Vector {del_vector_id} deleted from namespace '{del_namespace}'")
                        st.session_state.show_delete_confirm = False
                    except Exception as e:
                        st.error(f"Error deleting vector: {str(e)}")
            else:
                st.error("Please enter a vector ID")
                st.session_state.show_delete_confirm = False

        st.divider()

        # Delete by filter
        st.subheader("Delete by Filter")
        st.info("Delete all vectors matching specific criteria")

        col1, col2 = st.columns(2)
        with col1:
            del_filter_subject = st.text_input("Filter by Subject", key="del_filter_subject")
        with col2:
            del_filter_chapter = st.text_input("Filter by Chapter Name", key="del_filter_chapter")

        if st.button("🔍 Preview Vectors to Delete"):
            filter_dict = {}
            if del_filter_subject:
                filter_dict['subject'] = del_filter_subject
            if del_filter_chapter:
                filter_dict['chapter_name'] = del_filter_chapter

            if filter_dict:
                with st.spinner("Finding matching vectors..."):
                    try:
                        # Query to find matching vectors
                        preview_results = index.query(
                            vector=np.random.rand(stats.get('dimension', 1536)).tolist(),
                            top_k=100,
                            include_metadata=True,
                            namespace=del_namespace_value,
                            filter=filter_dict
                        )

                        if preview_results['matches']:
                            st.warning(f"⚠ Found {len(preview_results['matches'])} vectors matching the criteria")

                            # Show preview
                            preview_data = []
                            for match in preview_results['matches'][:10]:  # Show first 10
                                metadata = match.get('metadata', {})
                                preview_data.append({
                                    'ID': match['id'],
                                    'Chapter': metadata.get('chapter_name', 'N/A'),
                                    'Subject': metadata.get('subject', 'N/A'),
                                    'Type': metadata.get('chunk_type', 'N/A')
                                })

                            st.dataframe(pd.DataFrame(preview_data))

                            if len(preview_results['matches']) > 10:
                                st.info(f"Showing first 10 of {len(preview_results['matches'])} matches")

                            # Store IDs in session state for deletion
                            st.session_state.vectors_to_delete = [m['id'] for m in preview_results['matches']]
                            st.session_state.show_bulk_delete = True
                        else:
                            st.info("No vectors found matching the criteria")
                    except Exception as e:
                        st.error(f"Error searching for vectors: {str(e)}")
            else:
                st.warning("Please enter at least one filter criterion")

        # Show bulk delete button only after preview
        if hasattr(st.session_state, 'show_bulk_delete') and st.session_state.show_bulk_delete:
            col1, col2 = st.columns([1, 3])
            with col1:
                bulk_delete_btn = st.button("🗑 Delete All Matching", type="secondary")
            with col2:
                if bulk_delete_btn:
                    st.session_state.confirm_bulk_delete = True

            if hasattr(st.session_state, 'confirm_bulk_delete') and st.session_state.confirm_bulk_delete:
                confirm_del = st.checkbox("I understand this will delete ALL matching vectors", key="confirm_bulk")
                if confirm_del:
                    try:
                        index.delete(ids=st.session_state.vectors_to_delete, namespace=del_namespace_value)
                        st.success(f"✅ Deleted {len(st.session_state.vectors_to_delete)} vectors")
                        del st.session_state.vectors_to_delete
                        st.session_state.show_bulk_delete = False
                        st.session_state.confirm_bulk_delete = False
                    except Exception as e:
                        st.error(f"Error deleting vectors: {str(e)}")

        st.divider()

        # Clear namespace
        st.subheader("Clear Entire Namespace")
        st.error("⚠ This will delete ALL vectors in the selected namespace!")

        clear_namespace = st.selectbox("Select namespace to clear",
                                       namespaces[1:] if len(namespaces) > 1 else ["No namespaces available"],
                                       key="clear_namespace")

        if clear_namespace != "No namespaces available":
            col1, col2 = st.columns([1, 3])
            with col1:
                clear_btn = st.button("🗑 Clear Namespace", type="secondary")
            with col2:
                if clear_btn:
                    st.session_state.show_clear_confirm = True

            if hasattr(st.session_state, 'show_clear_confirm') and st.session_state.show_clear_confirm:
                confirm_clear = st.checkbox(f"I confirm I want to delete ALL vectors in namespace '{clear_namespace}'",
                                            key="confirm_clear")
                if confirm_clear:
                    confirm_text = st.text_input(f"Type '{clear_namespace}' to confirm", key="confirm_text")
                    if confirm_text == clear_namespace:
                        try:
                            index.delete(delete_all=True, namespace=clear_namespace)
                            st.success(f"✅ Cleared all vectors from namespace '{clear_namespace}'")
                            st.balloons()
                            st.session_state.show_clear_confirm = False
                        except Exception as e:
                            st.error(f"Error clearing namespace: {str(e)}")
                    elif confirm_text:
                        st.warning("Please type the namespace name exactly to confirm")

# Footer
st.markdown("---")

# Info section
with st.expander("ℹ About this Dashboard"):
    st.markdown("""
    ### Pinecone Dashboard Features:

    - *🔍 Search*: Search vectors using text queries (requires OpenAI API) or random vectors
    - *📄 Fetch by ID*: Retrieve specific vectors by their ID
    - *📊 Browse*: View sample vectors from your database
    - *📈 Statistics*: Visualize index and namespace statistics
    - *🗑 Management*: Delete vectors individually or in bulk

    ### Important Notes:

    1. *Namespaces*: Vectors are organized in board-specific namespaces (e.g., CBSE, ICSE)
    2. *Vector IDs*: Chapter sections have IDs like {chapter_id}_section_{n}
    3. *Search*: Text search requires OpenAI API key for embedding generation
    4. *Filters*: Use metadata filters to narrow down search results

    ### Environment Variables:

    - PINECONE_API_KEY: Your Pinecone API key
    - OPENAI_API_KEY: Your OpenAI API key (optional, for text search)

    ### Tips:

    - Use specific search terms for better results
    - Check the correct namespace when fetching vectors
    - Always preview before bulk deletions
    - Export data regularly for backup
    """)

# Add connection status in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔌 Connection Status")

    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Pinecone")
    with col2:
        if openai_client:
            st.success("✅ OpenAI")
        else:
            st.warning("❌ OpenAI")

    st.markdown("---")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if st.button("🔄 Refresh Page"):
        st.rerun()

# Session state management for preserving user inputs
if 'last_search_query' not in st.session_state:
    st.session_state.last_search_query = ""
if 'last_namespace' not in st.session_state:
    st.session_state.last_namespace = selected_namespace

# Custom styling
st.markdown("""
<style>
    /* Custom button styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }

    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background-color: #FF4B4B;
        color: white;
    }

    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: bold;
        font-size: 1.1em;
    }

    /* Success/Warning/Error message styling */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Error handling wrapper
def safe_operation(func, error_message="Operation failed"):
    """Wrapper for safe execution of operations"""
    try:
        return func()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        return None


# Additional utility functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def export_vectors_to_json(vectors_data):
    """Export vectors data to JSON format"""
    return json.dumps(vectors_data, indent=2)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def calculate_statistics(stats_dict):
    """Calculate additional statistics"""
    total_vectors = stats_dict.get('total_vector_count', 0)
    namespaces_count = len(stats_dict.get('namespaces', {}))

    return {
        'average_vectors_per_namespace': total_vectors / namespaces_count if namespaces_count > 0 else 0,
        'index_usage_percentage': (total_vectors / 1000000) * 100,
        'largest_namespace': max(stats_dict.get('namespaces', {}).items(),
                                 key=lambda x: x[1].get('vector_count', 0))[0] if stats_dict.get('namespaces') else None
    }


# Health check endpoint (for monitoring)
def health_check():
    """Check system health"""
    try:
        pc.list_indexes()
        return True
    except:
        return False


# Display health status
if health_check():
    st.sidebar.success("✅ System Healthy")
else:
    st.sidebar.error("❌ System Error")

# Add version info
st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0.0")
st.sidebar.caption("© 2024 Pinecone Dashboard")

# Clean up session state on page reload
if st.sidebar.button("🧹 Clear Session", help="Clear all temporary data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()