# ResearchGapAnalyzer


ResearchGapAnalyzer is an AI-powered system that analyzes research papers and identifies potential research gaps using Retrieval-Augmented Generation (RAG), vector search, and LLMs. The system enables users to upload research papers (PDFs), extract meaningful insights, and discover unexplored areas within a domain. By combining semantic search, intelligent text chunking, and LLM-based analysis, the project provides context-aware understanding of research content.
It also incorporates paper clustering and knowledge graph generation to reveal relationships between concepts and group similar research topics. With fast retrieval powered by FAISS and an interactive interface built using Streamlit, the system simplifies and accelerates the literature review process.

---

## 🌐 Live Demo
* URL:  https://researchgapanalyzer-dmrz27egtrybwnzvmhkk4j.streamlit.app/
* 
  "Note:  First load may take 1-2 mins on free tier. If the app is inactive, click Manage app → Reboot app to restart."

---

## 🎯 Objectives
* To develop a system that can upload and process research papers (PDF) for analysis.
* To implement intelligent text chunking for improved context understanding.
* To perform semantic search using vector embeddings for accurate information retrieval.
* To utilize Retrieval-Augmented Generation (RAG) with LLMs for contextual analysis.
* To identify potential research gaps and generate meaningful insights.
* To group similar research papers using clustering techniques.
* To generate a knowledge graph representing relationships between key concepts.
* To enable fast and efficient retrieval using FAISS vector search.
* To build an interactive and user-friendly interface using Streamlit.

---

## 🚀 Features

* 📄 Upload and process research papers (PDF)
* ✂️ Intelligent text chunking for better context understanding
* 🔍 Semantic search using vector embeddings (FastEmbed)
* 🤖 LLM-based analysis using Groq API (Llama 3.3 70B)
* 📊 Research gap identification and insights generation
* 📝 Automated literature review generation
* 🆚 Side-by-side paper comparison
* 📈 Method trend detection across papers
* 🧠 Paper clustering to group similar research topics
* 🌐 Knowledge graph generation for concept relationships
* ⚡ Fast retrieval using FAISS vector database
* 🖥 Interactive UI using Streamlit

---

## 🛠 Tech Stack

* Python
* Streamlit- (Frontend UI)
* Groq API (Llama 3.3 70B) — LLM inference (free, cloud-based)
* FAISS — Vector database for fast similarity search
* FastEmbed (BAAI/bge-small-en-v1.5) — Lightweight embeddings
* LangChain — RAG pipeline
* scikit-learn — Paper clustering (KMeans + PCA)
* NetworkX / Graph tools (for knowledge graph)
* PDF Processing Libraries

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
6. 🤖 Groq LLM (Llama 3.3 70B) generates insights
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

## 🐳 Docker

You can also run this app locally using Docker.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Run with Docker

```bash
# Build the image
docker build -t research-gap-analyzer .

# Run the app
docker run -p 8501:8501 -e GROQ_API_KEY=your_groq_api_key_here research-gap-analyzer
```

Then open http://localhost:8501 in your browser.

---

## 🧠 Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic search using embeddings
* Vector databases (FAISS)
* LLM integration (Groq API)
* NLP preprocessing and chunking
* Clustering algorithms (KMeans + PCA)
* Knowledge graph construction
* Modular system design
* Cloud deployment (Streamlit Community Cloud)
  
---

