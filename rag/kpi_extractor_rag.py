from __future__ import annotations

from typing import Any


def extract_financial_metrics(retriever: Any, company: str, year: int | None = None, candidate_docs: list | None = None) -> dict[str, Any]:
    """Return a minimal metric payload using the retrieved documents.

    This project was missing the full RAG KPI extractor, so this lightweight
    implementation keeps the ingestion flow functional without crashing.
    """
    # Prefer candidate docs (e.g., BM25 results) if provided, otherwise fall back
    # to using the retriever to perform a semantic retrieval.
    if candidate_docs:
        docs = candidate_docs
    else:
        try:
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(f"{company} annual report financial metrics", k=5)
            elif hasattr(retriever, "similarity_search"):
                docs = retriever.similarity_search(f"{company} annual report financial metrics", k=5)
            elif hasattr(retriever, "invoke"):
                docs = retriever.invoke(f"{company} annual report financial metrics", company=company, year=year, top_k=5)
            else:
                docs = []
        except Exception:
            docs = []

    result = {
        "company": company,
        "year": year,
        "metrics": {},
        "source_documents": len(docs),
    }

    if docs:
        text = "\n\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", None))
        result["metrics"] = {
            "summary": text[:2000],
        }

    return result
