# 🤖 PDF RAG Chatbot
A powerful chatbot that uses **Retrieval-Augmented Generation (RAG)** to intelligently answer questions based on your PDF documents. Now with a beautiful **Streamlit web interface** and **multi-file support**! Chat with your documents like never before!

## ✨ Features
- 🌐 **Web Interface**: Beautiful Streamlit UI for easy interaction
- 💬 **Interactive Chat**: Ask natural language questions about your PDF content
- 📚 **Multi-File Support**: Upload and query multiple PDF documents simultaneously
- 🗃️ **Persistent Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- 🧠 **Conversational Memory**: Maintains context across conversations (optional)
- 🚀 **Local Processing**: Runs entirely on your machine with Ollama
- 🔍 **Semantic Search**: Finds relevant information even with different wording
- 🧩 **Modular Design**: Easy to extend and customize
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Architecture
```
📁 Project Structure
├── 📄 app.py                  # Streamlit web application
├── 📄 main.py                 # Terminal-based chatbot (stateless)
├── 📄 rag_with_memory.py      # Terminal chatbot with memory (stateful)
├── 📁 chroma_db/              # Persistent vector database
├── 📁 documents/              # PDF storage (optional)
├── 📁 uploaded_files/         # Temporary storage for uploaded files
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

### 🌐 Web Interface (Recommended)
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

**Features of the Web Interface:**
- 📎 **Drag & Drop**: Upload multiple PDF files easily
- 💾 **Document Management**: View and manage uploaded documents
- 💬 **Chat Interface**: Clean, intuitive chat experience
- 📊 **Progress Tracking**: Real-time processing status
- 🗂️ **Session Management**: Maintain conversation history
- 🎨 **Dark/Light Mode**: Toggle between themes

### 📟 Terminal Interface

#### Basic Chatbot (No Memory)
```bash
python main.py path/to/your/document.pdf
```

#### Multi-file Terminal Usage
```bash
python main_v2.py path/to/your/folder.pdf
```

#### Chatbot with Streamlit UI
```bash
streamlit run streamlit_UI.py
```

### Example Web Session
1. **Upload PDFs**: Drag and drop multiple PDF files
2. **Processing**: Watch as documents are processed and indexed
3. **Chat**: Start asking questions about your documents
```
📄 Documents loaded: research_paper.pdf, manual.pdf, report.pdf
💬 Ask me anything about your documents:

You: What are the key findings across all documents?
🤖: Based on the uploaded documents, here are the key findings...

You: Compare the methodologies mentioned in the research paper and report
🤖: The research paper uses a quantitative approach while the report...
```

## 📋 Requirements
- **Python**: 3.8+
- **Ollama**: Latest version
- **Memory**: 8GB+ RAM recommended (12GB+ for multiple large documents)
- **Storage**: 5GB+ for model and vector database
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## 🔧 Configuration

### Supported Models
- `llama3.2:3b` (default, good balance of speed and capability)
- `llama3.2:1b` (lighter option for limited resources)
- `llama3.1:8b` (more capable, requires more resources)
- `llama3.1:70b` (most capable, requires significant resources)

### Customization Options
- **Chunk Size**: Modify document splitting parameters in the code
- **Model Selection**: Change the Ollama model in configuration
- **Memory Settings**: Adjust conversation history length
- **Vector Database**: Configure ChromaDB persistence settings
- **UI Theme**: Customize Streamlit interface colors and layout
- **File Upload Limits**: Adjust maximum file size and count limits

### Environment Variables
Create a `.env` file for custom configuration:
```env
OLLAMA_MODEL=llama3.2:3b
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## 🎯 Use Cases
- 📖 **Research**: Query academic papers and research documents
- 📋 **Documentation**: Search through technical manuals and guides
- 📊 **Reports**: Analyze business reports and financial documents
- 📚 **Education**: Study materials and textbooks
- ⚖️ **Legal**: Review contracts and legal documents
- 🏥 **Healthcare**: Medical literature and patient records

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

**"Streamlit not starting"**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 8501 is available
- Try running on a different port: `streamlit run app.py --server.port 8502`

**"Ollama not found"**
- Ensure Ollama is installed and running
- Check if the model is pulled: `ollama list`
- Verify Ollama service is running: `ollama serve`

**"Out of memory"**
- Try using a smaller model like `llama3.2:1b`
- Reduce document chunk size in the configuration
- Process fewer documents simultaneously

**"File upload errors"**
- Check file size limits (default: 50MB per file)
- Ensure PDFs are not corrupted or password-protected
- Verify write permissions in the project directory

**"ChromaDB errors"**
- Delete the `chroma_db/` folder to reset the database
- Ensure you have write permissions in the project directory
- Check disk space availability

### Performance Tips
- Use SSD storage for better vector database performance
- Increase RAM for handling larger documents
- Use GPU acceleration if available (configure in Ollama)
- Process documents in smaller batches for better stability

## 🔄 Updates & Changelog

### v2.0.0 (Latest)
- ✅ Added Streamlit web interface
- ✅ Multi-file upload and processing
- ✅ Enhanced document management
- ✅ Improved error handling
- ✅ Better user experience

### v1.0.0
- ✅ Terminal-based RAG chatbot
- ✅ ChromaDB integration
- ✅ Ollama model support
- ✅ Basic conversation memory

## 🙏 Acknowledgments
- [Streamlit](https://streamlit.io/) for the beautiful web interface
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF processing

## 🌟 Star History
If you find this project helpful, please consider giving it a star! ⭐

## 📞 Support
- 🐛 **Issues**: [GitHub Issues](https://github.com/Vishnu-E/rag_chat_bot/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Vishnu-E/rag_chat_bot/discussions)
- 📧 **Email**: Contact via GitHub profile

---
Made with ❤️ by [Vishnu-E](https://github.com/Vishnu-E)

*Transform your document interaction experience with AI-powered RAG technology!*
