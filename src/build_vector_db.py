import atexit
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import shutil
import tempfile

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from fastembed import TextEmbedding

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ---------------------------------------------------
# Track all temp directories created this process so
# they can be cleaned up on exit via atexit.
# ---------------------------------------------------
_temp_dirs: list[str] = []


def _cleanup_temp_dirs():
    """Remove all chroma temp directories created this session."""
    for path in _temp_dirs:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def build_vector_db(documents):

    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    persist_dir = tempfile.mkdtemp(prefix="chroma_session_")

    # Register for cleanup so temp dirs don't accumulate
    # across repeated Streamlit re-runs.
    _temp_dirs.append(persist_dir)

    vector_db = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=persist_dir
    )

    return vector_db