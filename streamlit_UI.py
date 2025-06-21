import streamlit as st
import os
import tempfile
import shutil
from typing import List, Dict, Any
import warnings
import uuid
warnings.filterwarnings("ignore")

# PDF Processing
import PyPDF2

# Embeddings and ML
from sentence_transformers import SentenceTransformer
import numpy as np

# Vector Database
import chromadb
from chromadb.config import Settings

# Language Model
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class PDFProcessor:
    """Handles PDF loading and text extraction."""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks for better embedding and retrieval."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
                
        return [chunk for chunk in chunks if chunk]


class EmbeddingModel:
    """Handles text embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully")
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to embedding vectors."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def encode_query(self, query: str) -> List[float]:
        """Convert a single query to embedding vector."""
        embedding = self.model.encode([query], convert_to_tensor=False)
        return embedding[0].tolist()


class VectorDatabase:
    """Handles vector storage and similarity search using ChromaDB."""
    
    def __init__(self, collection_name: str = "pdf_documents"):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = collection_name
        
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
    
    def clear_collection(self):
        """Clear all documents from the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     pdf_name: str = None):
        """Add document chunks and their embeddings to the vector database."""
        if pdf_name:
            base_name = pdf_name.replace('.pdf', '').replace(' ', '_')
            ids = [f"{base_name}_chunk_{i}" for i in range(len(texts))]
        else:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        metadatas = [{"source": pdf_name or "unknown", "chunk_index": i} 
                    for i in range(len(texts))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Added {len(texts)} document chunks from {pdf_name} to the database")
    
    def similarity_search(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        """Perform similarity search to find relevant document chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results.get('metadatas', [[]])[0]
        }
    
    def get_collection_info(self):
        """Get information about the current collection."""
        try:
            count = self.collection.count()
            print(f"Collection '{self.collection_name}' contains {count} documents")
            return count
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return 0


class LanguageModel:
    """Handles response generation using Ollama."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """Initialize the language model."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama is required but not installed. Install with: pip install ollama")
        
        self.model_name = model_name
        
        # Check if model is available
        try:
            ollama.show(model_name)
            print(f"Using model: {model_name}")
        except:
            print(f"Model {model_name} not found. Pulling model...")
            ollama.pull(model_name)
            print(f"Model {model_name} downloaded successfully")
    
    def generate_response(self, context: str, query: str) -> str:
        """Generate a response based on context and query."""
        prompt = f"""Based on the following context, please answer the question. Keep your response concise and within 50-100 words.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'max_tokens': 150,
                    'top_p': 0.9
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error generating response: {e}"


class RAGChatbot:
    """Main RAG Chatbot class that orchestrates all components."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 language_model: str = "llama3.2:3b"):
        """Initialize the RAG chatbot with all components."""
        print("Initializing RAG Chatbot...")
        
        self.pdf_processor = PDFProcessor()
        self.embedding_model = EmbeddingModel(embedding_model)
        self.vector_db = VectorDatabase()
        self.language_model = LanguageModel(language_model)
        self.conversation_history = []
        self.loaded_pdfs = []
        
        print("RAG Chatbot initialized successfully!")
    
    def load_and_process_pdf(self, pdf_path: str, clear_previous: bool = False):
        """Load and process a PDF document for the knowledge base."""
        print(f"Processing PDF: {pdf_path}")
        
        if clear_previous:
            self.vector_db.clear_collection()
            self.loaded_pdfs.clear()
            print("Cleared previous documents from database")
        
        text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Failed to extract text from PDF: {pdf_path}")
            return False
        
        print(f"Extracted {len(text)} characters from PDF")
        
        chunks = self.pdf_processor.chunk_text(text)
        print(f"Created {len(chunks)} text chunks")
        
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode_texts(chunks)
        
        pdf_name = os.path.basename(pdf_path)
        self.vector_db.add_documents(chunks, embeddings, pdf_name)
        if pdf_name not in self.loaded_pdfs:
            self.loaded_pdfs.append(pdf_name)
        
        print(f"PDF {pdf_name} processed and stored successfully!")
        return True
    
    def clear_all_documents(self):
        """Clear all documents from the database."""
        self.vector_db.clear_collection()
        self.loaded_pdfs.clear()
        self.conversation_history.clear()
    
    def query(self, user_query: str) -> str:
        """Process a user query and return a response."""
        query_embedding = self.embedding_model.encode_query(user_query)
        search_results = self.vector_db.similarity_search(query_embedding, n_results=5)
        
        context_by_source = {}
        for i, (doc, metadata) in enumerate(zip(search_results['documents'], search_results['metadatas'])):
            source = metadata.get('source', 'unknown')
            if source not in context_by_source:
                context_by_source[source] = []
            context_by_source[source].append(doc)
        
        context_parts = []
        for source, docs in context_by_source.items():
            context_parts.append(f"From {source}:\n" + "\n".join(docs[:2]))
        
        context = "\n\n".join(context_parts)
        
        memory_context = "\n".join(
            [f"User: {q}\nBot: {a}" for q, a in self.conversation_history[-3:]]
        )
        full_context = memory_context + "\n\n" + context if memory_context else context
        
        response = self.language_model.generate_response(full_context, user_query)
        self.conversation_history.append((user_query, response))
        
        return response


class StreamlitRAGChatbot:
    """Streamlit-specific wrapper for the RAG Chatbot."""
    
    def __init__(self):
        self.chatbot = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'chatbot_initialized' not in st.session_state:
            st.session_state.chatbot_initialized = False
        
        if 'chatbot_instance' not in st.session_state:
            st.session_state.chatbot_instance = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'loaded_pdfs' not in st.session_state:
            st.session_state.loaded_pdfs = []
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = ""
    
    def initialize_chatbot(self, embedding_model: str = "all-MiniLM-L6-v2", 
                          language_model: str = "llama3.2:3b"):
        """Initialize the RAG chatbot."""
        try:
            chatbot = RAGChatbot(
                embedding_model=embedding_model,
                language_model=language_model
            )
            return chatbot, None
        except Exception as e:
            return None, str(e)
    
    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file to temporary directory."""
        try:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None
    
    def process_pdfs(self, pdf_files, clear_previous=True):
        """Process uploaded PDF files."""
        if not self.chatbot:
            st.error("Chatbot not initialized!")
            return False
        
        try:
            if clear_previous:
                self.chatbot.clear_all_documents()
                st.session_state.loaded_pdfs.clear()
            
            successful_loads = 0
            temp_files = []
            
            for uploaded_file in pdf_files:
                st.session_state.processing_status = f"Processing {uploaded_file.name}..."
                
                temp_path = self.save_uploaded_file(uploaded_file)
                if temp_path:
                    temp_files.append(temp_path)
                    
                    if self.chatbot.load_and_process_pdf(temp_path, clear_previous=False):
                        successful_loads += 1
                        if uploaded_file.name not in st.session_state.loaded_pdfs:
                            st.session_state.loaded_pdfs.append(uploaded_file.name)
            
            # Cleanup temporary files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                    shutil.rmtree(os.path.dirname(temp_path))
                except:
                    pass
            
            st.session_state.processing_status = f"Successfully loaded {successful_loads}/{len(pdf_files)} PDFs"
            return successful_loads > 0
            
        except Exception as e:
            st.session_state.processing_status = f"Error: {e}"
            return False
    
    def clear_database(self):
        """Clear the ChromaDB database."""
        if self.chatbot:
            try:
                self.chatbot.clear_all_documents()
                st.session_state.loaded_pdfs.clear()
                st.session_state.chat_history.clear()
                return True
            except Exception as e:
                st.error(f"Error clearing database: {e}")
                return False
        return False
    
    def get_response(self, user_query: str):
        """Get response from the chatbot."""
        # Ensure we have the chatbot instance
        if not self.chatbot and 'chatbot_instance' in st.session_state:
            self.chatbot = st.session_state.chatbot_instance
        
        if not self.chatbot:
            return "Chatbot not initialized. Please check the setup."
        
        try:
            response = self.chatbot.query(user_query)
            return response
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 RAG Chatbot - Multi-PDF Question Answering")
    st.markdown("Upload PDF documents and ask questions about their content!")
    
    # Check if Ollama is available
    if not OLLAMA_AVAILABLE:
        st.error("⚠️ Ollama is not installed. Please install it with: `pip install ollama`")
        st.stop()
    
    # Initialize the Streamlit RAG Chatbot
    streamlit_chatbot = StreamlitRAGChatbot()
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show current models (fixed, no selection needed)
        st.info("**Embedding Model:** all-MiniLM-L6-v2")
        st.info("**Language Model:** llama3.2:3b")
        
        # Initialize chatbot button
        if st.button("🚀 Initialize Chatbot", type="primary"):
            with st.spinner("Initializing chatbot..."):
                chatbot, error = streamlit_chatbot.initialize_chatbot(
                    embedding_model="all-MiniLM-L6-v2", 
                    language_model="llama3.2:3b"
                )
                if chatbot:
                    streamlit_chatbot.chatbot = chatbot
                    st.session_state.chatbot_initialized = True
                    st.session_state.chatbot_instance = chatbot  # Store in session state
                    st.success("Chatbot initialized successfully!")
                else:
                    st.error(f"Failed to initialize chatbot: {error}")
        
        st.divider()
        
        # File upload section
        st.header("📄 Upload PDFs")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to create your knowledge base"
        )
        
        clear_previous = st.checkbox(
            "Clear previous documents", 
            value=True,
            help="Clear existing documents before loading new ones"
        )
        
        if uploaded_files:
            if st.session_state.chatbot_initialized:
                if st.button("📚 Process PDFs", type="secondary"):
                    # Ensure chatbot instance is available
                    if 'chatbot_instance' in st.session_state:
                        streamlit_chatbot.chatbot = st.session_state.chatbot_instance
                    
                    with st.spinner("Processing PDFs..."):
                        success = streamlit_chatbot.process_pdfs(uploaded_files, clear_previous)
                        if success:
                            st.success("PDFs processed successfully!")
                            st.rerun()
            else:
                st.warning("Please initialize the chatbot first!")
        
        st.divider()
        
        # Database management
        st.header("🗄️ Database Management")
        
        # Show loaded documents
        if st.session_state.loaded_pdfs:
            st.subheader("Loaded Documents:")
            for pdf_name in st.session_state.loaded_pdfs:
                st.text(f"📄 {pdf_name}")
        else:
            st.info("No documents loaded")
        
        # Clear database button
        if st.button("🗑️ Clear Database", type="secondary"):
            if streamlit_chatbot.clear_database():
                st.success("Database cleared successfully!")
                st.rerun()
        
        # Processing status
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Check if chatbot is ready
        if not st.session_state.chatbot_initialized:
            st.warning("Please initialize the chatbot first using the sidebar.")
            return
        
        if not st.session_state.loaded_pdfs:
            st.info("Upload and process PDF documents to start chatting.")
            return
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                # User message
                st.chat_message("user").write(user_msg)
                # Bot message
                st.chat_message("assistant").write(bot_msg)
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Add user message to history
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate and display bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = streamlit_chatbot.get_response(user_input)
                st.write(response)
            
            # Update chat history
            st.session_state.chat_history.append((user_input, response))
            st.rerun()
    
    with col2:
        st.header("ℹ️ Help & Tips")
        
        st.markdown("""
        **How to use:**
        1. Initialize the chatbot with your preferred models
        2. Upload one or more PDF documents
        3. Process the PDFs to create your knowledge base
        4. Start asking questions!
        
        **Tips:**
        - Use descriptive questions for better results
        - You can ask questions that span multiple documents
        - Clear the database when switching to different document sets
        
        **Commands:**
        - Ask specific questions about document content
        - Request summaries or comparisons
        - Inquire about details across multiple documents
        """)
        
        # Chat controls
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history.clear()
            st.rerun()
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.chatbot_initialized:
            st.success("✅ Chatbot Ready")
        else:
            st.error("❌ Chatbot Not Initialized")
        
        if st.session_state.loaded_pdfs:
            st.success(f"✅ {len(st.session_state.loaded_pdfs)} Documents Loaded")
        else:
            st.warning("⚠️ No Documents Loaded")


if __name__ == "__main__":
    main()