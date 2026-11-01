# Enterprise AI Knowledge Assistant | RAG-Based PDF Q&A
An AI-powered document assistant that lets users upload PDF files and
ask questions about their content. It retrieves relevant information
from the uploaded documents and uses the Gemini API to generate
context-grounded answers, helping users find information in lengthy
documents more easily.

**Key Features**
- Upload PDF documents through a web interface.
- Extract and process text from uploaded PDFs.
- Create semantic representations of document content using Sentence
  Transformers.
- Retrieve relevant content using FAISS.
- Improve retrieval results with Cross-Encoder reranking and
  Reciprocal Rank Fusion (RRF).
- Generate answers using the Gemini API, grounded in retrieved
  document context.
- Display source and page references with answers.

**Technologies Used**
- Backend: Python, FastAPI
- Generative AI: Gemini API
- Embeddings: Sentence Transformers
- Vector Search: FAISS
- Retrieval Enhancement: Cross-Encoder, Reciprocal Rank Fusion
  (RRF)
- Frontend: HTML, JavaScript

**How It Works**
1. The user uploads a PDF through the web interface.
2. The application extracts and processes the document text.
3. Sentence Transformers convert text segments into embeddings.
4. FAISS searches for content relevant to the user's question.
5. Cross-Encoder reranking and RRF help improve the retrieved results.
6. The relevant context is passed to the Gemini API to generate an
   answer.
7. The application presents the answer with available source and page
   references.
   
**Getting Started**

**Prerequisites**
- Python installed
- A Gemini API key
- Git


## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/GauriKate/Enterprise-AI-Knowledge-Assistant.git
```

### 2. Navigate to the Project Folder

```bash
cd Enterprise-AI-Knowledge-Assistant
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Gemini API

Set your Gemini API key using the environment variable expected by the application.

**Important:** Never upload your API key or other credentials to GitHub.

## Running the Application

Start the FastAPI server using:

```bash
uvicorn main:app --reload
```

Open the local URL in your browser.

To access the FastAPI documentation, visit:

```text
http://127.0.0.1:8000/docs
```

**My Contribution**

Worked on PDF text extraction and document processing, integrating the
Gemini API, implementing semantic retrieval with the selected embedding
and search components, and developing the question-answering workflow
with source-aware responses.

**Future Improvements**
- Support additional document formats.
- Add conversation history and multi-document collections.
- Improve evaluation of retrieval relevance and answer quality.
- Add user authentication and document-level access controls.
