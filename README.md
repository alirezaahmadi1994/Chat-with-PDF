# RAG Chatbot for PDF Q\&A

A **modular Retrieval-Augmented Generation (RAG)** chatbot that enables question answering over PDF documents using LangChain, Ollama models, and ChromaDB for persistent vector storage.

## ✨ Features

* 📄 Load and chunk PDF documents with `PyPDFLoader`
* 🧠 Generate embeddings and LLM responses using [Ollama](https://ollama.com/)
* 💾 Persistent vector store with [ChromaDB](https://docs.trychroma.com/)
* 🔍 Semantic search-powered retrieval with `RetrievalQA`
* 🔁 Auto-detects PDF changes to avoid unnecessary reprocessing

---

## 🧱 Architecture

```
PDF → Chunking → Embeddings → Vector DB (Chroma) → Retriever → LLM (Ollama) → Answer
```

---

## 📦 Requirements

* Python 3.8+
* [Ollama](https://ollama.com/) installed and running locally with a supported model (e.g., `mistral`, `llama2`, etc.)


## 🚀 Getting Started

### 1. Start Ollama

Ensure Ollama is running and you’ve pulled a model:

```bash
ollama run mistral
```

### 2. Run the Chatbot

```bash
python rag_chatbot.py path/to/document.pdf
```

### 3. Ask Questions

After ingestion, you'll enter an interactive session:

```text
You: What is the main topic of this document?
Bot: The document primarily discusses...
Sources: ['document.pdf']
```

Type `exit` to quit.

---

## 🧩 Module Overview

### `PDFChunker`

* Splits PDFs into manageable chunks using recursive splitting for better semantic indexing.

### `VectorStoreHandler`

* Handles vector database creation and fingerprinting for change detection.
* Avoids redundant embedding if PDF hasn't changed.

### `RAGChatbot`

* Encapsulates ingestion, QA chain setup, and question answering.
* Uses LangChain's `RetrievalQA` to combine retriever and Ollama LLM.

---

## 🧠 Example: Embedding + QA Logic

```python
chatbot = RAGChatbot(model_name="mistral")
chatbot.ingest_pdf("sample.pdf")
answer, sources = chatbot.ask("What is the summary of this document?")
print(answer)
```

---

## 📁 Project Structure

```
rag_chatbot.py         # Main chatbot logic
README.md              # Project documentation
chroma_db/             # (Generated) Vector database storage
```

---

## 🔒 PDF Fingerprinting

To optimize performance, the system generates a SHA-256 hash and checks file size/modification time to determine if the vector store needs rebuilding.

---

## 🙋 FAQ

**Q:** What models are supported?
**A:** Any LLM and embedding model available via Ollama (e.g., `mistral`, `llama2`, `gemma`).

**Q:** Can I ingest multiple PDFs?
**A:** Not in the current version, but the system is modular and can be extended to support multi-doc indexing.

---

## 👥 Contributions

Pull requests are welcome. Feel free to open issues to suggest improvements or report bugs.
