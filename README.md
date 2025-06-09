# 🧠 RAG-PDF Chatbot with Ollama, ChromaDB, and LLaMA 3

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit**, allowing users to upload PDF documents, store them in a **ChromaDB vector store**, and ask questions using **LLaMA 3** (via Ollama). It includes **semantic search**, **document chunking**, and **re-ranking** using a **CrossEncoder**.

---

## ✨ Features

- 📄 Upload PDF files and split them into semantically meaningful chunks
- 💡 Store document embeddings in a local persistent **ChromaDB**
- 🧠 Use **`nomic-embed-text:v1.5`** for generating embeddings
- 🦙 Query responses are generated using **Ollama’s LLaMA 3 model**
- 📈 Reranking of retrieved documents using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- 🔁 Streamed, real-time responses via `ollama.chat`
- ⚡ Built using **Streamlit** for fast UI development

---

## 🧰 Tech Stack

| Layer       | Technology                                 |
|-------------|---------------------------------------------|
| Frontend    | Streamlit                                   |
| Backend     | Python + LangChain                          |
| LLM         | LLaMA 3 (via Ollama)                        |
| Embeddings  | nomic-embed-text:v1.5 (Ollama)              |
| Reranking   | sentence-transformers CrossEncoder          |
| Vector DB   | ChromaDB (local, persistent)                |
| Document Loader | PyMuPDFLoader                          |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or 3.11
- [Ollama installed](https://ollama.com/)
- Git

### 2. Clone the repository

```bash
git clone https://github.com/vikramm0907/Chat-with-your-notes
cd Chat-with-your-notes
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Pull necessary models in Ollama
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text:v1.5
```
### 5. Run the streamlit app
```bash
streamlit run app.py
```
## 📝 How It Works
### 1. Upload PDFs → The documents are loaded and split using RecursiveCharacterTextSplitter.

### 2. Embedding → Each chunk is converted into a vector using Ollama’s nomic-embed-text.

### 3. Storage → The vectors are stored in a persistent ChromaDB collection.

### 4. Querying → User questions are vector-searched against the collection.

### 5. Reranking → Top results are reranked using a CrossEncoder.

### 6. LLM Response → The top context chunks are sent to llama3.2 via ollama.chat.

## 🖼 UI Preview

<img src="ag chat app.png" width="80%" />

## 🗂 Project Structure
```
├── app.py                  # Main Streamlit app
├── demo-rag-chroma/        # ChromaDB persistent store
├── requirements.txt
├── README.md
```
## 🔐 Notes
### -> This app runs entirely locally; no API keys or cloud dependencies.
### -> For performance, try reducing chunk size or switching to faster LLMs.
### -> ChromaDB and Ollama must be running properly in the background.
