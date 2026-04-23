from groq import Groq
import streamlit as st

class GroqGenerator:
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0):
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": str(prompt)}],
            temperature=self.temperature
        )
        return type("Msg", (), {"content": response.choices[0].message.content})()

generator = GroqGenerator(model="llama-3.3-70b-versatile", temperature=0)

_MAX_CHARS_PER_DOC = 1500
_MAX_DOCS = 10


def analyze_docs(docs):
    context = ""
    for doc in docs[:_MAX_DOCS]:
        snippet = doc.page_content[:_MAX_CHARS_PER_DOC].strip()
        context += snippet + "\n\n"

    prompt = f"""You are a research analyst. Analyze the following research paper excerpts.

Provide a detailed response with FOUR clearly labeled sections:

## 1. Common Methodologies
List every methodology, model, or technique mentioned.

## 2. Limitations
List every limitation or weakness the authors mention.

## 3. Research Gaps
List what problems remain unsolved or unexplored.

## 4. Future Research Directions
List concrete suggestions for future work.

Use bullet points within each section. Be specific and thorough.

RESEARCH EXCERPTS:
{context.strip()}
"""

    return generator.invoke(prompt).content  # ← was missing return