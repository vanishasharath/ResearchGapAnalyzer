import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def chunk_papers(papers: list[Document]) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Improvement 2: skip and warn on empty documents rather than
    # silently producing zero chunks for a failed PDF extraction.
    valid_papers = []
    for paper in papers:
        if paper.page_content and paper.page_content.strip():
            valid_papers.append(paper)
        else:
            source = paper.metadata.get("source", "unknown")
            logger.warning("Skipping empty document: %s", source)

    # Improvement 1: split_documents() handles the split + metadata
    # copy internally — no need for a manual loop.
    documents = splitter.split_documents(valid_papers)

    return documents