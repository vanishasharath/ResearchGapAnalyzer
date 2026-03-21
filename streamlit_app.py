import streamlit as st
import os
import hashlib
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.load_pdf import load_papers
from src.chunk_text import chunk_papers
from src.build_vector_db import build_vector_db
from src.retreiver import retrieve_chunks
from src.analyzer import analyze_docs
from src.features import (
    detect_method_frequency,
    generate_literature_review,
    compare_papers
)
from src.knowledge_graph import build_knowledge_graph
from src.paper_clustering import cluster_papers


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Research Paper Analyzer",
    layout="wide"
)

st.title("📚 AI Research Paper Analyzer")
st.caption("Analyze research papers using AI-powered insights")


# ---------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------

for key in ["vector_db", "documents", "papers", "last_upload_hash"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def get_upload_hash(uploaded_files):
    """Return an MD5 hash of all uploaded file contents combined."""
    combined = b"".join(f.getvalue() for f in uploaded_files)
    return hashlib.md5(combined).hexdigest()


def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temp directory and return their paths."""
    tmp_dir = tempfile.mkdtemp()
    file_paths = []
    for file in uploaded_files:
        path = os.path.join(tmp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(path)
    return file_paths


# ---------------------------------------------------
# PROCESS PAPERS
# Cached by file-content hash, not by mutable file paths.
# ---------------------------------------------------

@st.cache_resource
def process_papers(file_paths, _upload_hash: str):
    """
    Load, chunk, and embed papers.
    _upload_hash is used only as a cache-busting key.
    """
    papers = load_papers(file_paths)
    documents = chunk_papers(papers)
    vector_db = build_vector_db(documents)
    return papers, documents, vector_db


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Upload Research Papers")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

if uploaded_files:

    upload_hash = get_upload_hash(uploaded_files)

    # Re-process only when the uploaded file set actually changes.
    if st.session_state.last_upload_hash != upload_hash:
        st.session_state.last_upload_hash = upload_hash
        st.session_state.vector_db = None   # clear stale state

    if st.session_state.vector_db is None:
        with st.spinner("Processing research papers…"):
            try:
                file_paths = save_uploaded_files(uploaded_files)
                papers, documents, vector_db = process_papers(
                    file_paths, upload_hash
                )
                st.session_state.papers = papers
                st.session_state.documents = documents
                st.session_state.vector_db = vector_db
            except Exception as e:
                st.error(f"Failed to process papers: {e}")
                st.stop()


papers = st.session_state.papers
documents = st.session_state.documents
vector_db = st.session_state.vector_db


# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

if vector_db:

    # ---------------------------------------------------
    # FIX: n_papers counts unique source files, not chunks.
    # PyPDFLoader creates one Document per PAGE, so len(papers)
    # returns total pages, not total PDFs. Count unique sources.
    # ---------------------------------------------------
    unique_sources = set(
        doc.metadata.get("source", "") for doc in papers
    )
    n_papers = len(unique_sources)
    n_chunks = len(documents) if documents else 0

    st.success(f"{n_papers} paper(s) processed successfully.")

    col1, col2 = st.columns(2)
    col1.metric("Papers Loaded", n_papers)
    col2.metric("Total Chunks", n_chunks)

    # ---------------------------------------------------
    # TABS
    # ---------------------------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Research Gap Analysis",
        "Literature Review",
        "Paper Comparison",
        "Method Trends",
        "Knowledge Graph",
        "Paper Clustering",
    ])

    # ---------------------------------------------------
    # RESEARCH GAP ANALYSIS
    # ---------------------------------------------------

    with tab1:
        st.header("Research Gap Analysis")
        if st.button("Run Analysis"):
            with st.spinner("Analyzing research gaps… (this may take ~30s)"):
                try:
                    docs = retrieve_chunks(
                        vector_db,
                        "limitations future work research gap",
                        k=8
                    )
                    result = analyze_docs(docs)
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # ---------------------------------------------------
    # LITERATURE REVIEW
    # ---------------------------------------------------

    with tab2:
        st.header("Literature Review")
        if st.button("Generate Literature Review"):
            with st.spinner("Generating literature review… (this may take ~30s)"):
                try:
                    docs = retrieve_chunks(
                        vector_db,
                        "research methodology contribution findings",
                        k=8
                    )
                    review = generate_literature_review(docs)
                    st.markdown(review)
                except Exception as e:
                    st.error(f"Literature review generation failed: {e}")

    # ---------------------------------------------------
    # PAPER COMPARISON
    # ---------------------------------------------------

    with tab3:
        st.header("Paper Comparison")
        if st.button("Compare Papers"):
            with st.spinner("Comparing papers… (this may take ~30s)"):
                try:
                    docs = retrieve_chunks(
                        vector_db,
                        "methods datasets results performance evaluation",
                        k=8
                    )
                    comparison = compare_papers(docs)
                    st.markdown(comparison)
                except Exception as e:
                    st.error(f"Paper comparison failed: {e}")

    # ---------------------------------------------------
    # METHOD FREQUENCY
    # ---------------------------------------------------

    with tab4:
        st.header("Method Trends")
        if st.button("Detect Method Trends"):
            with st.spinner("Detecting methods…"):
                try:
                    # Use all documents for method detection — no LLM call,
                    # just regex so we can scan everything cheaply.
                    freq = detect_method_frequency(documents)
                    if freq:
                        df = pd.DataFrame(
                            freq.items(),
                            columns=["Method", "Frequency"]
                        ).sort_values("Frequency", ascending=False)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index("Method"))
                    else:
                        st.warning(
                            "No known methods detected. "
                            "The papers may use domain-specific terminology "
                            "not in the current method list."
                        )
                except Exception as e:
                    st.error(f"Method trend detection failed: {e}")

    # ---------------------------------------------------
    # KNOWLEDGE GRAPH
    # ---------------------------------------------------

    with tab5:
        st.header("Knowledge Graph")
        st.caption(
            "Nodes = keywords extracted from paper chunks. "
            "Edges = keywords that co-occur in the same chunk. "
            "Thicker/closer nodes = stronger co-occurrence."
        )

        max_docs_graph = st.slider(
            "Chunks to include", min_value=5,
            max_value=max(5, n_chunks), value=min(15, n_chunks), step=5
        )

        if st.button("Generate Graph"):
            with st.spinner("Building knowledge graph…"):
                try:
                    G = build_knowledge_graph(documents[:max_docs_graph])

                    if len(G.nodes) == 0:
                        st.warning("Graph is empty — try including more chunks.")
                    else:
                        fig, ax = plt.subplots(figsize=(12, 7))
                        pos = nx.spring_layout(G, seed=42, k=0.8)

                        # Scale edge width by weight
                        weights = [
                            G[u][v].get("weight", 1)
                            for u, v in G.edges()
                        ]
                        max_w = max(weights) if weights else 1

                        nx.draw_networkx_edges(
                            G, pos,
                            width=[1 + 3 * (w / max_w) for w in weights],
                            alpha=0.5, edge_color="gray", ax=ax
                        )
                        nx.draw_networkx_nodes(
                            G, pos,
                            node_color="skyblue",
                            node_size=1200, ax=ax
                        )
                        nx.draw_networkx_labels(
                            G, pos, font_size=8, ax=ax
                        )
                        ax.set_title(
                            "Keyword Co-occurrence Graph\n"
                            "Node size = keyword, Edge thickness = co-occurrence strength"
                        )
                        ax.axis("off")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Knowledge graph generation failed: {e}")

    # ---------------------------------------------------
    # PAPER CLUSTERING
    # ---------------------------------------------------

    with tab6:
        st.header("Paper Similarity Clustering")
        st.caption(
            "Papers are grouped by semantic similarity of their content. "
            "The scatter plot shows each chunk projected to 2D — "
            "chunks close together are semantically similar."
        )

        max_docs_cluster = st.slider(
            "Chunks to cluster", min_value=10,
            max_value=max(10, n_chunks), value=min(50, n_chunks), step=10
        )

        num_clusters = st.slider(
            "Number of clusters", min_value=2, max_value=8, value=3
        )

        if st.button("Run Clustering"):
            with st.spinner("Clustering papers…"):
                try:
                    fig = cluster_papers(
                        documents[:max_docs_cluster],
                        num_clusters=num_clusters
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

# ---------------------------------------------------
# DEFAULT SCREEN
# ---------------------------------------------------

else:
    st.info("Upload one or more research PDFs in the sidebar to begin.")