from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

print("Program started")

# Check HuggingFace token
print("HF TOKEN:", hf_token)


# Import project modules
from src.load_pdf import load_papers
from src.chunk_text import chunk_papers
from src.build_vector_db import build_vector_db
from src.retreiver import retrieve_chunks
from src.analyzer import analyze_docs
from src.features import detect_method_frequency, generate_literature_review, compare_papers
from src.knowledge_graph import build_knowledge_graph, visualize_graph
from src.paper_clustering import cluster_papers


# -----------------------------
# STEP 1 — Load research papers
# -----------------------------
print("\nStep 1: Loading papers")

papers = load_papers("papers")

print("Number of papers:", len(papers))

if len(papers) == 0:
    print("No papers found in 'papers' folder")
    exit()


# -----------------------------
# STEP 2 — Chunk papers
# -----------------------------
print("\nStep 2: Chunking papers")

documents = chunk_papers(papers)

print("Chunks created:", len(documents))


# -----------------------------
# STEP 3 — Build vector database
# -----------------------------
print("\nStep 3: Building vector database")

vector_db = build_vector_db(documents)

print("Vector DB created")


# -----------------------------
# STEP 4 — Retrieve chunks
# -----------------------------
print("\nStep 4: Retrieving relevant chunks")

docs = retrieve_chunks(vector_db, "limitations and future work")

print("Retrieved docs:", len(docs))


# -----------------------------
# STEP 5 — Research gap analysis
# -----------------------------
print("\nStep 5: Research gap analysis")

analysis = analyze_docs(docs)

print("\nResearch Gap Analysis")
print("----------------------")
print(analysis)


# -----------------------------
# STEP 6 — Method frequency
# -----------------------------
print("\nMethod Frequency")

freq = detect_method_frequency(docs)

print(freq)


# -----------------------------
# STEP 7 — Literature review
# -----------------------------
print("\nLiterature Review")

review = generate_literature_review(docs)

print(review)


# -----------------------------
# STEP 8 — Paper comparison
# -----------------------------
print("\nPaper Comparison")

comparison = compare_papers(docs)

print(comparison)


# -----------------------------
# STEP 9 — Knowledge graph
# -----------------------------
print("\nBuilding Knowledge Graph")

graph = build_knowledge_graph(docs)

visualize_graph(graph)


# -----------------------------
# STEP 10 — Paper clustering
# -----------------------------
#print("\nPaper Similarity Clustering")

#clusters = cluster_papers(papers)

#print(clusters)