from collections import Counter
import re
from langchain_ollama import OllamaLLM

# Shared Ollama generator — same model as analyzer.py
generator = OllamaLLM(model="tinyllama", temperature=0)

_MAX_CHARS_PER_DOC = 1500
_MAX_DOCS = 5

# Expanded method list covering NLP, CV, and ML broadly
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
            # Use word boundary so "gnn" doesn't match inside "beginning"
            if re.search(rf"\b{re.escape(method)}\b", text):
                found_methods.append(method)

    return dict(Counter(found_methods))


# ---------------------------------------------------
# Feature 2: Literature Review
# ---------------------------------------------------

def generate_literature_review(docs):

    context = ""
    for doc in docs[:_MAX_DOCS]:
        snippet = doc.page_content[:_MAX_CHARS_PER_DOC].strip()
        context += snippet + "\n\n"

    prompt = f"""You are an academic writer. Write a coherent literature review based on the research excerpts below.

Structure your review as flowing paragraphs (NOT bullet points) covering:
- The main research themes and problems being addressed
- The approaches and methods used across the papers
- How the papers relate to or build upon each other
- Key findings and contributions

Write at least 4 paragraphs. Do not repeat sentences. Be specific about the papers' contributions.

RESEARCH EXCERPTS:
{context.strip()}
"""

    return generator.invoke(prompt)


# ---------------------------------------------------
# Feature 3: Paper Comparison
# ---------------------------------------------------

def compare_papers(docs):

    context = ""
    for doc in docs[:_MAX_DOCS]:
        snippet = doc.page_content[:_MAX_CHARS_PER_DOC].strip()
        context += snippet + "\n\n"

    prompt = f"""You are a research analyst. Compare the research approaches in the excerpts below.

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

RESEARCH EXCERPTS:
{context.strip()}
"""

    return generator.invoke(prompt)