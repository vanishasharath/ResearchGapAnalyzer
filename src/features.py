from collections import Counter
import re
from groq import Groq
import os
import streamlit as st
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

class GroqGenerator:
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": str(prompt)}],
            temperature=self.temperature
        )
        # Mimic .content attribute like ChatOllama returns
        return type("Msg", (), {"content": response.choices[0].message.content})()

# Drop-in replacement — no other code changes needed
generator = GroqGenerator(model="llama-3.3-70b-versatile", temperature=0)

_MAX_CHARS_PER_DOC = 1500
_MAX_DOCS = 6  # ← was 5, you have 6 papers

_METHODS = [
    "transformer", "bert", "gpt", "t5", "roberta", "xlnet", "albert",
    "cnn", "convolutional neural network",
    "rnn", "recurrent neural network",
    "lstm", "long short-term memory",
    "gru", "gated recurrent unit",
    "gan", "generative adversarial network",
    "vae", "variational autoencoder",
    "graph neural network", "gnn",
    "reinforcement learning", "deep reinforcement learning",
    "random forest", "gradient boosting", "xgboost", "lightgbm",
    "svm", "support vector machine",
    "logistic regression", "linear regression",
    "attention mechanism", "self-attention", "cross-attention",
    "transfer learning", "fine-tuning", "pre-training",
    "zero-shot", "few-shot", "in-context learning",
    "diffusion model", "stable diffusion",
    "contrastive learning", "self-supervised learning",
    "knowledge distillation", "pruning", "quantization",
    "named entity recognition", "ner",
    "sentiment analysis", "text classification",
    "machine translation", "summarization",
    "question answering", "information retrieval",
]


# ---------------------------------------------------
# Feature 1: Method Frequency
# ---------------------------------------------------

def detect_method_frequency(docs):
    found_methods = []
    for doc in docs:
        text = doc.page_content.lower()
        for method in _METHODS:
            if re.search(rf"\b{re.escape(method)}\b", text):
                found_methods.append(method)
    return dict(Counter(found_methods))


# ---------------------------------------------------
# Feature 2: Literature Review
# ---------------------------------------------------

def generate_literature_review(docs):
    print(f"Total docs passed: {len(docs)}")
    for doc in docs:
        print(f"  source: {doc.metadata.get('source', 'NONE')}")
    seen_sources = {}
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))  # ← basename
        if source not in seen_sources:
            seen_sources[source] = doc

    unique_docs = list(seen_sources.values())[:_MAX_DOCS]

    context = ""
    for i, doc in enumerate(unique_docs):
        source_name = os.path.basename(doc.metadata.get("source", f"Paper {i+1}"))
        snippet = doc.page_content[:_MAX_CHARS_PER_DOC].strip()
        context += f"[Paper {i+1}: {source_name}]\n{snippet}\n\n"

    prompt = f"""You are an academic writer. Write a coherent literature review based on the research excerpts below.

Structure your review as flowing paragraphs (NOT bullet points) covering:
- The main research themes and problems being addressed
- The approaches and methods used across the papers
- How the papers relate to or build upon each other
- Key findings and contributions

Write at least 4 paragraphs. Do not repeat sentences. Be specific about paper contributions.
Do NOT include any file paths, system information, or metadata in your response.
Refer to papers as "Paper 1", "Paper 2" etc. or by their topic.

RESEARCH EXCERPTS:
{context.strip()}
"""

    return generator.invoke(prompt).content

# ---------------------------------------------------
# Feature 3: Paper Comparison
# ---------------------------------------------------

import os

def compare_papers(docs):
    seen_sources = {}
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))  # ← basename only
        if source not in seen_sources:
            seen_sources[source] = doc

    unique_docs = list(seen_sources.values())[:_MAX_DOCS]

    context = ""
    for i, doc in enumerate(unique_docs):
        source_name = os.path.basename(doc.metadata.get("source", f"Paper {i+1}"))
        snippet = doc.page_content[:_MAX_CHARS_PER_DOC].strip()
        context += f"[Paper {i+1}: {source_name}]\n{snippet}\n\n"

    prompt = f"""You are a research analyst. Compare the research approaches in the excerpts below.
There are {len(unique_docs)} papers total — compare ALL of them.

Provide a structured comparison with these FOUR sections:

## Methods
Compare the specific techniques, models, and algorithms each paper uses.

## Datasets
List the datasets mentioned and which papers use them.

## Strengths
What does each approach do well?

## Weaknesses
What are the limitations or shortcomings of each approach?

Be specific and reference details from the text. Use bullet points within each section.
Refer to papers as "Paper 1", "Paper 2" etc.

RESEARCH EXCERPTS:
{context.strip()}
"""

    return generator.invoke(prompt).content  # ← was missing return