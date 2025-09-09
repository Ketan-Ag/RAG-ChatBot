import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq
import hashlib
import pickle
import re

# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PDFProcessor:
    """Handle PDF text extraction and processing"""

    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

            # Extract text using PyMuPDF
            doc = fitz.open(tmp_file_path)
            text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()

            doc.close()
            os.unlink(tmp_file_path)  # Clean up temp file

            return text

        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []

        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

            if start >= len(text):
                break

        return chunks

class VectorStore:
    """Handle vector embeddings and similarity search"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if not chunks:
            return np.array([])

        with st.spinner("Creating embeddings..."):
            embeddings = self.model.encode(chunks, show_progress_bar=False)

        return embeddings

    def build_index(self, chunks: List[str]):
        """Build FAISS index from text chunks"""
        self.chunks = chunks

        if not chunks:
            st.warning("No text chunks to process")
            return

        # Create embeddings
        self.embeddings = self.create_embeddings(chunks)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        st.success(f"Built vector index with {len(chunks)} chunks")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if self.index is None or not self.chunks:
            return []

        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })

        return results

class GroqLLM:
    """Handle Groq API interactions"""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Default model

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using Groq API"""
        # Prepare context
        context_text = "\n\n".join(context)

        # Create prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from a PDF document.

Context from PDF:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so and provide what information you can.

Answer:"""

        try:
            # Make API call
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1000
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

class RAGChatbot:
    """Main RAG chatbot class"""

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm = None
        self.pdf_processed = False

    def setup_llm(self, api_key: str):
        """Initialize Groq LLM"""
        self.llm = GroqLLM(api_key)

    def process_pdf(self, pdf_file):
        """Process uploaded PDF"""
        # Extract text
        text = self.pdf_processor.extract_text_from_pdf(pdf_file)

        if not text:
            st.error("Could not extract text from PDF")
            return False

        # Chunk text
        chunks = self.pdf_processor.chunk_text(text)

        if not chunks:
            st.error("Could not create text chunks")
            return False

        # Build vector index
        self.vector_store.build_index(chunks)
        self.pdf_processed = True

        return True

    def answer_question(self, question: str) -> str:
        """Answer question using RAG"""
        if not self.pdf_processed:
            return "Please upload and process a PDF first."

        if not self.llm:
            return "Please configure Groq API key."

        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(question, k=3)

        if not relevant_chunks:
            return "No relevant information found in the PDF."

        # Extract context
        context = [chunk['chunk'] for chunk in relevant_chunks]

        # Generate response
        response = self.llm.generate_response(question, context)

        return response, relevant_chunks

def main():
    """Main Streamlit application"""

    # Title and description
    st.title("📚 PDF RAG Chatbot")
    st.markdown("Upload a PDF and ask questions about its content!")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Groq API Key input
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key"
        )

        if api_key:
            st.session_state.chatbot.setup_llm(api_key)
            st.success("✅ API key configured")

        st.divider()

        # PDF Upload
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat about"
        )

        if uploaded_file is not None:
            st.info(f"📁 File: {uploaded_file.name}")

            if st.button("🔄 Process PDF", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    success = st.session_state.chatbot.process_pdf(uploaded_file)

                if success:
                    st.success("✅ PDF processed successfully!")
                    st.session_state.messages = []  # Clear chat history

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.header("💬 Chat with your PDF")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📋 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.text_area(
                            f"Source {i+1} (Score: {source['score']:.3f})",
                            source["chunk"],
                            height=100,
                            disabled=True
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()

        if not st.session_state.chatbot.pdf_processed:
            st.error("Please upload and process a PDF first.")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chatbot.answer_question(prompt)

                if isinstance(result, tuple):
                    response, sources = result
                    st.markdown(response)

                    # Add assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                else:
                    st.markdown(result)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result
                    })

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with Streamlit • Powered by Groq API</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()