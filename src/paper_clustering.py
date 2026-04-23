import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def cluster_papers(papers, num_clusters=3):
    if not papers:
        raise ValueError("No documents provided for clustering.")

    model = load_embedding_model()

    # ← group chunks by source paper
    from collections import defaultdict
    import random

    paper_chunks = defaultdict(list)
    for doc in papers:
        source = doc.metadata.get("source", "unknown")
        paper_chunks[source].append(doc)

    # ← take RANDOM chunks from each paper (better representation)
    balanced_docs = []
    chunks_per_paper = max(1, num_clusters * 3)
    for source, chunks in paper_chunks.items():
        selected = random.sample(
            chunks,
            min(len(chunks), chunks_per_paper)
        )
        balanced_docs.extend(selected)

    if not balanced_docs:
        balanced_docs = papers

    texts = [doc.page_content for doc in balanced_docs]
    num_clusters = min(num_clusters, len(texts))
    embeddings = model.encode(texts, show_progress_bar=False)

    # clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    def get_label(doc):
        source = doc.metadata.get("source", "")
        if source:
            return os.path.splitext(os.path.basename(source))[0][:30]
        return doc.page_content[:40].strip() + "…"

    clusters: dict[int, list[str]] = {i: [] for i in range(num_clusters)}

    # ✅ FIX: use balanced_docs instead of papers
    for i, label in enumerate(labels):
        clusters[int(label)].append(get_label(balanced_docs[i]))

    # ✅ better duplicate removal (preserves order)
    for k in clusters:
        clusters[k] = list(dict.fromkeys(clusters[k]))

    # --- Plot ---
    reduced = PCA(n_components=2).fit_transform(embeddings)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.get_cmap("tab10", num_clusters)

    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        axes[0].scatter(
            reduced[mask, 0], reduced[mask, 1],
            label=f"Cluster {cluster_id}",
            color=cmap(cluster_id),
            s=80, alpha=0.8
        )
    axes[0].set_title(
        "Chunk Similarity Clusters (PCA)\n"
        "Points = document chunks · Colour = cluster",
        fontsize=10
    )
    axes[0].legend()
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")

    row_labels, row_data = [], []
    for cluster_id, titles in clusters.items():
        for t in titles:
            row_labels.append(f"Cluster {cluster_id}")
            row_data.append([t])

    axes[1].axis("off")
    if row_data:
        table = axes[1].table(
            cellText=row_data,
            rowLabels=row_labels,
            colLabels=["Paper"],
            loc="center",
            cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
    axes[1].set_title("Papers per Cluster", fontsize=10, pad=12)

    plt.tight_layout()
    return fig