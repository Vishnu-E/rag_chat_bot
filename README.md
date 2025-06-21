# 🤖 PDF RAG Chatbot

A powerful terminal-based chatbot that uses **Retrieval-Augmented Generation (RAG)** to intelligently answer questions based on your PDF documents. Chat with your documents like never before!

## ✨ Features

- 💬 **Interactive Chat**: Ask natural language questions about your PDF content
- 📚 **Persistent Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- 🧠 **Conversational Memory**: Maintains context across conversations (optional)
- 🚀 **Local Processing**: Runs entirely on your machine with Ollama
- 🔍 **Semantic Search**: Finds relevant information even with different wording
- 🧩 **Modular Design**: Easy to extend and customize

## 🏗️ Architecture

```
📁 Project Structure
├── 📄 main.py                 # Main chatbot (stateless)
├── 📄 rag_with_memory.py      # Chatbot with memory (stateful)
├── 📁 chroma_db/              # Persistent vector database
├── 📁 documents/              # PDF storage (optional)
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md               # This file
```

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Vishnu-E/rag_chat_bot.git
cd rag_chat_bot
```

### 2. Create Virtual Environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Ollama
1. **Install Ollama**: Visit [ollama.ai](https://ollama.ai/) and download for your OS
2. **Pull the model**:
   ```bash
   ollama pull llama3.2:3b
   ```
3. **Verify installation**:
   ```bash
   ollama list
   ```

## 🚀 Usage

### Basic Chatbot (No Memory)
```bash
python main.py path/to/your/document.pdf
```

### Chatbot with Memory
```bash
python rag_with_memory.py path/to/your/document.pdf
```

### Example Session
```
📄 Processing document: research_paper.pdf
✅ Document loaded successfully!

💬 Ask me anything about your document (type 'quit' to exit):

You: What is the main conclusion of this paper?
🤖: Based on the document, the main conclusion is that...

You: Can you explain the methodology used?
🤖: The methodology section describes...
```

## 📋 Requirements

- **Python**: 3.8+
- **Ollama**: Latest version
- **Memory**: 8GB+ RAM recommended
- **Storage**: 5GB+ for model and vector database

## 🔧 Configuration

### Supported Models
- `llama3.2:3b` (default, fastest)
- `llama3.2:1b` (lighter option)
- `llama3.1:8b` (more capable, requires more resources)

### Customization Options
- **Chunk Size**: Modify document splitting parameters
- **Model Selection**: Change the Ollama model in the code
- **Memory Settings**: Adjust conversation history length
- **Vector Database**: Configure ChromaDB persistence settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

**"Ollama not found"**
- Ensure Ollama is installed and running
- Check if the model is pulled: `ollama list`

**"Out of memory"**
- Try using a smaller model like `llama3.2:1b`
- Reduce document chunk size in the code

**"ChromaDB errors"**
- Delete the `chroma_db/` folder to reset the database
- Ensure you have write permissions in the project directory

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings

---

Made with ❤️ by [Vishnu-E](https://github.com/Vishnu-E)
