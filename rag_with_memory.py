
import os
import sys
import argparse
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# PDF Processing
import PyPDF2

# Embeddings and ML
from sentence_transformers import SentenceTransformer
import numpy as np

# Vector Database
import chromadb
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
            print(f"Error reading PDF: {e}")
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
        """
        Initialize ChromaDB client and collection.

        """
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        self.collection_name = collection_name
        
        
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        """
        Add document chunks and their embeddings to the vector database.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
        """
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids
        )
        print(f"Added {len(texts)} document chunks to the database")
    
    def similarity_search(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        """
        Perform similarity search to find relevant document chunks.
        
        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0]
        }


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
        
        print("RAG Chatbot initialized successfully!")
    
    def load_and_process_pdf(self, pdf_path: str):
        """
        Load and process a PDF document for the knowledge base.
        
        Args:
            pdf_path: Path to the PDF file
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        if not text:
            print("Failed to extract text from PDF")
            return
        
        print(f"Extracted {len(text)} characters from PDF")
        
        # Chunk the text
        chunks = self.pdf_processor.chunk_text(text)
        print(f"Created {len(chunks)} text chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode_texts(chunks)
        
        # Store in vector database
        self.vector_db.add_documents(chunks, embeddings)
        print("PDF processed and stored successfully!")
    
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
        search_results = self.vector_db.similarity_search(query_embedding, n_results=3)
        context = "\n\n".join(search_results['documents'])

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
    
    def start_chat(self):
        """Start the interactive chat interface."""
        print("\n" + "="*60)
        print("RAG Chatbot Terminal Interface")
        print("="*60)
        print("Ask questions about the loaded PDF document.")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == "reset":
                    self.conversation_history.clear()
                    print("Memory cleared.\n")
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Nice talking to you! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("Bot: Thinking...")
                response = self.query(user_input)
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Main function to run the RAG chatbot."""
    parser = argparse.ArgumentParser(description="RAG Chatbot Terminal Application")
    parser.add_argument("pdf_path", help="Path to the PDF document")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                       help="Sentence transformer model name")
    parser.add_argument("--language-model", default="llama3.2:3b", 
                       help="Ollama model name")
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        sys.exit(1)
    
    try:
        # Initialize chatbot
        chatbot = RAGChatbot(
            embedding_model=args.embedding_model,
            language_model=args.language_model
        )
        
        # Load and process PDF
        chatbot.load_and_process_pdf(args.pdf_path)
        
        # Start interactive chat
        chatbot.start_chat()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()