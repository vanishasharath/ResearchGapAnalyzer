import logging
import os

from langchain_community.document_loaders import PyPDFLoader

# ---------------------------------------------------
# Improvement 1: Removed unused `import fitz` (PyMuPDF).
# It was imported but never referenced anywhere in the file.
# ---------------------------------------------------

logger = logging.getLogger(__name__)


def load_papers(file_paths: list[str]) -> list:
    """
    Load PDF files and return a flat list of LangChain Documents.

    Skips files that are missing or fail to parse so that one
    bad PDF does not crash the entire pipeline.
    """
    papers = []

    for path in file_paths:

        # Improvement 3: validate the file exists before loading
        # so the error message is clear rather than cryptic.
        if not os.path.exists(path):
            logger.warning("File not found, skipping: %s", path)
            continue

        # Improvement 2: catch per-file errors (corrupt PDF,
        # password-protected, etc.) so one bad file does not
        # abort the rest of the pipeline.
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            papers.extend(docs)

        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            continue

    if not papers:
        raise ValueError(
            "No documents were loaded. "
            "Check that the uploaded files are valid, readable PDFs."
        )

    return papers