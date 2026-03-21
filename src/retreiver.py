from langchain_core.vectorstores import VectorStore


def retrieve_chunks(vector_db: VectorStore, query: str, k: int = 5) -> list:

    # Improvement 2: fail clearly if the DB is not yet initialised,
    # rather than raising an unhelpful AttributeError downstream.
    if vector_db is None:
        raise ValueError(
            "vector_db is not initialised. "
            "Ensure build_vector_db() has been called before retrieving."
        )

    # Improvement 1: k is now a parameter with a sensible default,
    # so callers can request more or fewer chunks per task.
    retriever = vector_db.as_retriever(
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    return docs