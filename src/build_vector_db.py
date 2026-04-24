import atexit
import os
import shutil
import tempfile

_temp_dirs: list[str] = []


def _cleanup_temp_dirs():
    for path in _temp_dirs:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def build_vector_db(documents):
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS

    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vector_db = FAISS.from_documents(documents, embedding_model)

    return vector_db