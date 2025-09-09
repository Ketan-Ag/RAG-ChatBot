# PDF RAG Chatbot Setup Guide

## Prerequisites
- Python 3.8 or higher
- Groq API key (get from https://console.groq.com/)

## Installation Steps

### 1. Clone or Download the Files
Make sure you have the following files in your project directory:
- `pdf_rag_chatbot.py` (main application)
- `requirements.txt` (dependencies)

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get Groq API Key
1. Visit https://console.groq.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the API key (you'll need this in the app)

### 5. Run the Application
```bash
streamlit run pdf_rag_chatbot.py
```

The app will open in your browser at http://localhost:8501

## How to Use

### 1. Configure API Key
- Enter your Groq API key in the sidebar
- The app will confirm when the key is configured

### 2. Upload PDF
- Click "Choose a PDF file" in the sidebar
- Select your PDF document
- Click "Process PDF" to extract and index the content

### 3. Start Chatting
- Type questions about your PDF in the chat input
- The AI will answer based on the PDF content
- View source chunks used for each answer in the expandable "Sources" section

## Features

### 🔍 Intelligent Text Processing
- Extracts text from PDF using PyMuPDF
- Splits content into optimized chunks with overlap
- Creates semantic embeddings using sentence-transformers

### 🚀 Fast Similarity Search
- Uses FAISS for efficient vector similarity search
- Retrieves most relevant chunks for each question
- Cosine similarity matching for accurate results

### 🤖 Powered by Groq API
- Fast inference using Groq's LPU technology
- Uses Llama3-8B model for high-quality responses
- Contextual answers based on retrieved PDF content

### 💬 Interactive Chat Interface
- Clean Streamlit interface
- Chat history preservation
- Source attribution for transparency
- Real-time processing indicators

## Troubleshooting

### Common Issues and Solutions

**1. API Key Error**
- Ensure your Groq API key is valid and has sufficient credits
- Check if the API key is correctly entered (no extra spaces)

**2. PDF Processing Error**
- Ensure the PDF is not password-protected
- Try with a different PDF if text extraction fails
- Some PDFs with complex layouts may not extract well

**3. Memory Issues with Large PDFs**
- For very large PDFs, consider splitting them into smaller sections
- Adjust chunk size in the code if needed (default: 1000 characters)

**4. Slow Performance**
- First run may be slower due to model downloads
- Subsequent runs should be faster
- Consider using a smaller embedding model if needed

## Customization Options

### Modify Chunk Size
In `pdf_rag_chatbot.py`, line ~69:
```python
chunks = self.pdf_processor.chunk_text(text, chunk_size=1000, overlap=200)
```

### Change Embedding Model
In `pdf_rag_chatbot.py`, line ~87:
```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
```

### Adjust Number of Retrieved Chunks
In `pdf_rag_chatbot.py`, line ~248:
```python
relevant_chunks = self.vector_store.search(question, k=3)
```

## Technical Architecture

1. **PDF Processing**: PyMuPDF extracts text from uploaded PDFs
2. **Text Chunking**: Intelligent splitting with sentence boundary detection
3. **Embeddings**: sentence-transformers creates semantic vectors
4. **Vector Store**: FAISS provides fast similarity search
5. **LLM Integration**: Groq API generates contextual responses
6. **UI**: Streamlit provides interactive web interface

## Security Notes

- API keys are handled securely in session state
- Temporary files are cleaned up after processing
- No data is stored permanently on disk
- All processing happens locally except LLM calls

## Performance Tips

- Use PDFs with clear text (not scanned images)
- Smaller PDFs (< 100 pages) work best
- Ensure stable internet for Groq API calls
- Close unused browser tabs to free memory

## License and Credits

This application uses several open-source libraries:
- Streamlit for the web interface
- PyMuPDF for PDF processing
- sentence-transformers for embeddings
- FAISS for vector search
- Groq API for language model inference
