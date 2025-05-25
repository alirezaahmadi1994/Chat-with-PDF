# rag_chatbot.py
"""
A modular RAG chatbot using PyPDF, LangChain, and ChromaDB for document-based Q&A.
"""
import hashlib
import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA


class PDFChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_and_chunk(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return self.splitter.split_documents(docs)

class VectorStoreHandler:
    def __init__(self, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        self.db = None
        self.fingerprint_path = os.path.join(self.persist_directory, "pdf_fingerprint.json")

    def _compute_fingerprint(self, pdf_path):
        stat = os.stat(pdf_path)
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return {
            "path": os.path.abspath(pdf_path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": hasher.hexdigest(),
        }

    def _load_fingerprint(self):
        if os.path.exists(self.fingerprint_path):
            with open(self.fingerprint_path, "r") as f:
                return json.load(f)
        return None

    def _save_fingerprint(self, fingerprint):
        os.makedirs(self.persist_directory, exist_ok=True)
        with open(self.fingerprint_path, "w") as f:
            json.dump(fingerprint, f)

    def needs_rebuild(self, pdf_path):
        current = self._compute_fingerprint(pdf_path)
        saved = self._load_fingerprint()
        return saved != current

    def build_store(self, docs, embedding_model, pdf_path):
        self.db = Chroma.from_documents(
            docs, embedding_model, persist_directory=self.persist_directory
        )
        self._save_fingerprint(self._compute_fingerprint(pdf_path))

    def load_store(self, embedding_model):
        self.db = Chroma(
            persist_directory=self.persist_directory, embedding_function=embedding_model
        )

    def as_retriever(self):
        return self.db.as_retriever()

class RAGChatbot:
    def __init__(self, model_name="mistral", persist_directory="chroma_db"):
        self.embedding_model = OllamaEmbeddings(model=model_name)
        self.llm = OllamaLLM(model=model_name)
        self.vector_handler = VectorStoreHandler(persist_directory)
        self.qa_chain = None

    def ingest_pdf(self, pdf_path):
        chunker = PDFChunker()
        if self.vector_handler.needs_rebuild(pdf_path):
            docs = chunker.load_and_chunk(pdf_path)
            self.vector_handler.build_store(docs, self.embedding_model, pdf_path)
        self.vector_handler.load_store(self.embedding_model)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.vector_handler.as_retriever(), return_source_documents=True
        )

    def ask(self, question):
        if not self.qa_chain:
            raise ValueError("No QA chain initialized. Ingest a PDF first.")
        result = self.qa_chain.invoke({"query": question})
        return result["result"], result["source_documents"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Chatbot for PDF Q&A")
    parser.add_argument("pdf", help="Path to PDF document")
    args = parser.parse_args()

    chatbot = RAGChatbot()
    chatbot.ingest_pdf(args.pdf)
    print("Ready for questions! Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        answer, sources = chatbot.ask(q)
        print(f"Bot: {answer}\nSources: {[s.metadata.get('source', '') for s in sources]}")
