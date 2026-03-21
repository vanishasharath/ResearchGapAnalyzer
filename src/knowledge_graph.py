import re
from itertools import combinations
from collections import Counter

import networkx as nx
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_MIN_WORD_LEN = 3


def _extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Return the top_n most frequent non-stopword tokens from text.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    tokens = [
        t for t in tokens
        if t not in _STOPWORDS and len(t) >= _MIN_WORD_LEN
    ]
    most_common = Counter(tokens).most_common(top_n)
    return [word for word, _ in most_common]


def build_knowledge_graph(documents) -> nx.Graph:

    G = nx.Graph()

    for doc in documents:

        keywords = _extract_keywords(doc.page_content)

        for word_a, word_b in combinations(keywords, 2):

            if G.has_edge(word_a, word_b):
                G[word_a][word_b]["weight"] += 1
            else:
                G.add_edge(word_a, word_b, weight=1)

    # Prune weak edges to keep the graph readable
    weak_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("weight", 1) < 2
    ]
    G.remove_edges_from(weak_edges)

    # Remove isolated nodes left behind after pruning
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G