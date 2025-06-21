import os
import sys
import argparse
from typing import List, Dict, Any
import warnings
import glob
warnings.filterwarnings("ignore")

# PDF Processing
import PyPDF2

# Embeddings and ML
from sentence_transformers import SentenceTransformer
import numpy as np

# Vector Database
import chromadb
import uuid
import hashlib
from chromadb.config import Settings

# Language Model (using Ollama for open-source models)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: Ollama not installed. Install with: pip install ollama")


class PDFProcessor:
    """Handles PDF loading and text extraction."""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
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
        """
        Split text into chunks for better embedding and retrieval.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
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
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     pdf_name: str = None):
        """
        Add document chunks and their embeddings to the vector database.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            pdf_name: Name of the PDF file (for unique IDs)
        """
        # Generate unique IDs using UUID or hash-based approach
        if pdf_name:
            # Use PDF name + chunk index for more meaningful IDs
            base_name = pdf_name.replace('.pdf', '').replace(' ', '_')
            ids = [f"{base_name}_chunk_{i}" for i in range(len(texts))]
        else:
            # Use UUID for completely unique IDs
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Add metadata to track source document
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
    
    def get_documents_by_source(self, source_name: str, n_results: int = 5) -> Dict[str, Any]:
        """Get documents from a specific source PDF."""
        results = self.collection.query(
            query_embeddings=None,
            where={"source": source_name},
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results.get('metadatas', [[]])[0] if results.get('metadatas') else []
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


class EmbeddingModel:
    """Handles text embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        Using all-MiniLM-L6-v2 as it's lightweight and effective for semantic similarity.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully")
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to embedding vectors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def encode_query(self, query: str) -> List[float]:
        """
        Convert a single query to embedding vector.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode([query], convert_to_tensor=False)
        return embedding[0].tolist()


class LanguageModel:
    """Handles response generation using Ollama."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the language model.
        Using Ollama to access open-source models locally.
        llama3.2:3b - optimized for systems with limited resources.
        """
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
        """
        Generate a response based on context and query.
        
        Args:
            context: Retrieved context from documents
            query: User query
            
        Returns:
            Generated response
        """
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
        """
        Initialize the RAG chatbot with all components.
        
        Args:
            embedding_model: Name of the sentence transformer model
            language_model: Name of the Ollama model
        """
        print("Initializing RAG Chatbot...")
        
        self.pdf_processor = PDFProcessor()
        self.embedding_model = EmbeddingModel(embedding_model)
        self.vector_db = VectorDatabase()
        self.language_model = LanguageModel(language_model)
        self.conversation_history = [] # List to store conversation history (for memory)
        self.loaded_pdfs = []  # Track loaded PDF files
        
        print("RAG Chatbot initialized successfully!")
    
    def load_and_process_pdf(self, pdf_path: str, clear_previous: bool = False):
        """
        Load and process a PDF document for the knowledge base.
        
        Args:
            pdf_path: Path to the PDF file
            clear_previous: Whether to clear previous documents
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Clear previous documents if requested
        if clear_previous:
            self.vector_db.clear_collection()
            self.loaded_pdfs.clear()
            print("Cleared previous documents from database")
        
        # Extract text from PDF
        text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Failed to extract text from PDF: {pdf_path}")
            return False
        
        print(f"Extracted {len(text)} characters from PDF")
        
        # Chunk the text
        chunks = self.pdf_processor.chunk_text(text)
        print(f"Created {len(chunks)} text chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode_texts(chunks)
        
        # Store in vector database with PDF name for unique IDs
        pdf_name = os.path.basename(pdf_path)
        self.vector_db.add_documents(chunks, embeddings, pdf_name)
        self.loaded_pdfs.append(pdf_name)
        
        print(f"PDF {pdf_name} processed and stored successfully!")
        return True
    
    def load_pdfs_from_directory(self, directory_path: str, clear_previous: bool = True):
        """
        Load all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            clear_previous: Whether to clear previous documents
        """
        print(f"Loading PDFs from directory: {directory_path}")
        
        # Find all PDF files in the directory
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in directory: {directory_path}")
            return False
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        successful_loads = 0
        for i, pdf_file in enumerate(pdf_files):
            print(f"\n--- Processing {i+1}/{len(pdf_files)} ---")
            if self.load_and_process_pdf(pdf_file, clear_previous=(clear_previous and i == 0)):
                successful_loads += 1
        
        print(f"\n--- Summary ---")
        print(f"Successfully loaded {successful_loads}/{len(pdf_files)} PDF files")
        self.vector_db.get_collection_info()
        return successful_loads > 0
    
    def query(self, user_query: str) -> str:
        """
        Process a user query and return a response.
        
        Args:
            user_query: User's question
            
        Returns:
            Generated response
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode_query(user_query)
        
        # Retrieve relevant context
        search_results = self.vector_db.similarity_search(query_embedding, n_results=5)
        
        # Group results by source to ensure diverse context
        context_by_source = {}
        for i, (doc, metadata) in enumerate(zip(search_results['documents'], search_results['metadatas'])):
            source = metadata.get('source', 'unknown')
            if source not in context_by_source:
                context_by_source[source] = []
            context_by_source[source].append(doc)
        
        # Build context from multiple sources
        context_parts = []
        for source, docs in context_by_source.items():
            context_parts.append(f"From {source}:\n" + "\n".join(docs[:2]))
        
        context = "\n\n".join(context_parts)

        # Add conversation history. Limited to last 3 interactions
        memory_context = "\n".join(
            [f"User: {q}\nBot: {a}" for q, a in self.conversation_history[-3:]]
        )
        full_context = memory_context + "\n\n" + context if memory_context else context
        
        # Generate response
        response = self.language_model.generate_response(context, user_query)

        # Update memory
        self.conversation_history.append((user_query, response))
        
        return response
    
    def get_document_summary(self, pdf_name: str) -> str:
        """Get a summary of a specific document."""
        # Get first few chunks from the document
        try:
            source_docs = self.vector_db.get_documents_by_source(pdf_name, n_results=3)
            if source_docs['documents']:
                context = "\n\n".join(source_docs['documents'])
                summary_query = f"Summarize the main topics and findings in this document: {pdf_name}"
                return self.language_model.generate_response(context, summary_query)
            else:
                return f"No content found for {pdf_name}"
        except Exception as e:
            return f"Error getting summary for {pdf_name}: {e}"
    
    def show_collection_info(self):
        """Show information about loaded documents."""
        count = self.vector_db.get_collection_info()
        if self.loaded_pdfs:
            print(f"Loaded PDFs: {', '.join(self.loaded_pdfs)}")
        else:
            print("No PDFs currently loaded")
    
    def cleanup_database(self):
        """Clean up the database when shutting down."""
        try:
            self.vector_db.client.delete_collection(self.vector_db.collection_name)
            print("Database collection cleared")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def start_chat(self):
        """Start the interactive chat interface."""
        print("\n" + "="*60)
        print("RAG Chatbot Terminal Interface")
        print("="*60)
        print("Ask questions about the loaded PDF documents.")
        print("Commands:")
        print("  'info' - Show loaded documents info")
        print("  'list' - List all loaded documents")
        print("  'summary <filename>' - Get summary of specific document")
        print("  'reset' - Clear conversation memory")
        print("  'clear-db' - Clear database and quit")
        print("  'quit' or 'exit' - End conversation")
        print("="*60 + "\n")
        
        # Show initial info
        self.show_collection_info()
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == "info":
                    self.show_collection_info()
                    continue
                
                if user_input.lower() == "list":
                    print("Loaded documents:")
                    for i, pdf in enumerate(self.loaded_pdfs, 1):
                        print(f"  {i}. {pdf}")
                    print()
                    continue
                
                if user_input.lower().startswith("summary "):
                    filename = user_input[8:].strip()
                    if filename in self.loaded_pdfs:
                        print(f"Bot: Generating summary for {filename}...")
                        summary = self.get_document_summary(filename)
                        print(f"Bot: {summary}\n")
                    else:
                        print(f"Document '{filename}' not found. Available documents:")
                        for pdf in self.loaded_pdfs:
                            print(f"  - {pdf}")
                        print()
                    continue
                
                if user_input.lower() == "reset":
                    self.conversation_history.clear()
                    print("Conversation memory cleared.\n")
                    continue
                
                if user_input.lower() == "clear-db":
                    print("Clearing database and exiting...")
                    self.cleanup_database()
                    print("Goodbye!")
                    break
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    # Ask user if they want to clear the database
                    while True:
                        clear_choice = input("Clear database before exiting? (y/n): ").strip().lower()
                        if clear_choice in ['y', 'yes']:
                            self.cleanup_database()
                            break
                        elif clear_choice in ['n', 'no']:
                            print("Database preserved for next session.")
                            break
                        else:
                            print("Please enter 'y' or 'n'")
                    
                    print("Nice talking to you! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("Bot: Thinking...")
                response = self.query(user_input)
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                # Ask about clearing database
                try:
                    clear_choice = input("Clear database before exiting? (y/n): ").strip().lower()
                    if clear_choice in ['y', 'yes']:
                        self.cleanup_database()
                except KeyboardInterrupt:
                    pass
                print("Nice talking to you! Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Main function to run the RAG chatbot."""
    parser = argparse.ArgumentParser(description="RAG Chatbot Terminal Application")
    parser.add_argument("path", help="Path to PDF file or directory containing PDF files")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                       help="Sentence transformer model name")
    parser.add_argument("--language-model", default="llama3.2:3b", 
                       help="Ollama model name")
    parser.add_argument("--clear-db", action="store_true",
                       help="Clear existing database before loading new documents")
    
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' not found.")
        sys.exit(1)
    
    try:
        # Initialize chatbot
        chatbot = RAGChatbot(
            embedding_model=args.embedding_model,
            language_model=args.language_model
        )
        
        # Load documents
        success = False
        if os.path.isfile(args.path) and args.path.lower().endswith('.pdf'):
            # Single PDF file
            success = chatbot.load_and_process_pdf(args.path, clear_previous=args.clear_db)
        elif os.path.isdir(args.path):
            # Directory containing PDFs
            success = chatbot.load_pdfs_from_directory(args.path, clear_previous=args.clear_db)
        else:
            print(f"Error: '{args.path}' is not a valid PDF file or directory.")
            sys.exit(1)
        
        if not success:
            print("Failed to load any documents. Exiting.")
            sys.exit(1)
        
        # Start interactive chat
        chatbot.start_chat()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()