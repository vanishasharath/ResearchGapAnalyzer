# ResearchGapAnalyzer


An AI-powered system that analyzes research papers and identifies potential research gaps using **Retrieval-Augmented Generation (RAG)**, **vector search**, and **LLMs (Ollama)**.

This project enables users to upload research papers, extract meaningful insights, and discover unexplored areas in a domain.

---

## 🚀 Features

* 📄 Upload and process research papers (PDF)
* ✂️ Intelligent text chunking for better context understanding
* 🔍 Semantic search using vector embeddings
* 🤖 LLM-based analysis using **Ollama (local models)**
* 📊 Research gap identification and insights generation
* 🧠 Paper clustering to group similar research topics
* 🌐 Knowledge graph generation for concept relationships
* ⚡ Fast retrieval using vector database (ChromaDB)
* 🖥 Interactive UI using Streamlit

---

## 🛠 Tech Stack

* **Python**
* **Streamlit** (Frontend UI)
* **Ollama** (Local LLM inference)
* **ChromaDB** (Vector database)
* **NLP & Embeddings**
* **NetworkX / Graph tools** (for knowledge graph)
* **PDF Processing Libraries**

---

## 📂 Project Structure

```
research-gap-project/
├── src/
│   ├── analyzer.py              # Core RAG pipeline
│   ├── build_vector_db.py       # Vector DB creation
│   ├── chunk_text.py            # Text chunking logic
│   ├── features.py              # Feature extraction
│   ├── knowledge_graph.py       # Knowledge graph generation
│   ├── load_pdf.py              # PDF processing
│   ├── paper_clustering.py      # Clustering similar papers
│   └── retriever.py             # Retrieval logic
│
├── app.py                       # Main execution script
├── streamlit_app.py             # UI application
├── requirements.txt             # Dependencies
├── README.md
└── .gitignore
```

---

## ⚙️ How It Works

1. 📄 Upload research papers
2. ✂️ Text is split into meaningful chunks
3. 🔢 Embeddings are generated
4. 📦 Stored in vector database (ChromaDB)
5. 🔍 Relevant chunks retrieved based on query
6. 🤖 LLM (Ollama) generates insights
7. 📊 Outputs research gaps, clusters, and relationships

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/ResearchGapAnalyzer.git
cd research-gap-project
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Run the application

#### Option A: CLI

```
python app.py
```

#### Option B: Streamlit UI

```
streamlit run streamlit_app.py
```

---

## 🧠 Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic search using embeddings
* Vector databases (ChromaDB)
* LLM integration (Ollama)
* NLP preprocessing and chunking
* Clustering algorithms
* Knowledge graph construction
* Modular system design

---

