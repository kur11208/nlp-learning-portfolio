"""Find similar documents using TF-IDF and cosine similarity."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS = [
    "cat drinks milk",
    "dog eats meat",
    "cat eats fish",
]
QUERIES = ["cat eats milk", "dog eats meat"]


def search_documents(
    documents: list[str], queries: list[str]
) -> list[tuple[str, str, float]]:
    """Return (query, best document, similarity) for each query.

    The vectorizer is fitted to the document collection, not to queries.
    """
    if not documents:
        raise ValueError("documents must not be empty")
    if not queries:
        return []

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vectors = vectorizer.transform(queries)
    scores = cosine_similarity(query_vectors, doc_vectors)

    results = []
    for query, row in zip(queries, scores):
        best_index = int(np.argmax(row))
        results.append((query, documents[best_index], float(row[best_index])))
    return results


if __name__ == "__main__":
    for query, document, similarity in search_documents(DOCUMENTS, QUERIES):
        print(f"query: {query}")
        print(f"match: {document} (cosine similarity: {similarity:.3f})")
