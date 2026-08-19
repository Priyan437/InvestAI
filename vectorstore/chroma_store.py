from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import chromadb
from langchain_core.documents import Document


class Retriever:
    """Minimal retrieval wrapper around a Chroma collection."""

    def __init__(self, client: Any, collection_name: str = "investor_docs", k: int = 5):
        self.client = client
        self.k = k
        self.collection = getattr(client, "collection", None)
        if self.collection is None:
            self.collection = client.get_or_create_collection(name=collection_name)

    def get_relevant_documents(self, query: str, k: int | None = None) -> list[Document]:
        n_results = self.k if k is None else k
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(docs, metas)
        ]


class ChromaVectorStore:
    """A small Chroma-backed vector store that matches the project contract."""

    def __init__(self, persist_directory: str, embeddings: Any):
        self.persist_directory = str(Path(persist_directory).resolve())
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name="investor_docs")

    def _embed_text(self, text: str):
        if hasattr(self.embeddings, "embed_documents"):
            return self.embeddings.embed_documents([text])[0]
        if hasattr(self.embeddings, "embed_query"):
            return self.embeddings.embed_query(text)
        raise TypeError("Provided embeddings object does not support embed_documents/embed_query")

    def upload_chunks(
        self,
        chunks: list[Document],
        embeddings: Any,
        company: str,
        year: str | int | None,
        source_file: str,
    ) -> None:
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        vectors: list[list[float]] = []

        for index, chunk in enumerate(chunks):
            page_content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            metadata = getattr(chunk, "metadata", {}) or {}
            metadata.update({
                "company": company,
                "year": str(year) if year is not None else "",
                "source_file": source_file,
                "chunk_index": index,
            })

            documents.append(page_content)
            metadatas.append(metadata)
            ids.append(str(uuid.uuid4()))
            vectors.append(self._embed_text(page_content))

        if documents:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=vectors,
                metadatas=metadatas,
            )

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        embedding = self._embed_text(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(docs, metas)
        ]
