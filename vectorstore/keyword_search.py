import uuid
from types import SimpleNamespace
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class ChromaVectorStore:
    """
    Wrapper around a persistent Chroma vector store.
    """
    def __init__(self, persist_directory: str, embeddings):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    def upload_chunks(
        self,
        chunks: list[Document],
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        for chunk in chunks:
            chunk.metadata.update({
                "company": company,
                "year": year,
                "source_file": source_file
            })
        self.store.add_documents(chunks)
        print(f"Uploaded {len(chunks)} chunks to Chroma for {company} {year}")

    @property
    def client(self):
        return self.store


class Retriever:
    """
    BM25 keyword retriever operating over documents pulled from a local Chroma store.
    Mirrors the Azure Search Retriever interface (`invoke`).
    """
    def __init__(self, store):
        self.store = store  # this is the Chroma client (self.store from ChromaVectorStore)

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """
        Retrieve relevant chunks using BM25 ranking over documents filtered
        from Chroma by company/year metadata.
        """
        filter_dict = {}
        if company:
            filter_dict["company"] = company
        if year:
            filter_dict["year"] = str(year)

        # Pull all matching documents (metadata filter, no vector search yet)
        raw = self.store.get(
            where=filter_dict if filter_dict else None,
            include=["documents", "metadatas"]
        )

        documents = raw.get("documents", [])
        if not documents:
            return []

        # Tokenize for BM25 (simple whitespace tokenizer; swap for nltk/spacy if needed)
        tokenized_corpus = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Rank by BM25 score, descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = [
            SimpleNamespace(page_content=documents[i])
            for i in ranked_indices
        ]
        return results