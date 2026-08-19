# Install filter-repo (once)
pip install git-filter-repo

# Make a backup clone
git clone --mirror ~/Desktop/InvestorRAG InvestorRAG-mirror.git

# In original repo: remove .env from history
cd ~/Desktop/InvestorRAG
git filter-repo --invert-paths --path .env

# Force-push cleaned history
git push --force origin --all
git push --force origin --tags

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Ingestion.pdf_to_markdown import PDFToMarkdownConverter
from Ingestion.semantic_chunker import chunk_markdown
from vectorstore.chroma_store import ChromaVectorStore, Retriever as ChromaRetriever
from vectorstore.keyword_search import Retriever as BM25Retriever
from rag.kpi_extractor_rag import extract_financial_metrics
from database.save_metrics import save_metrics

load_dotenv()


def parse_company_year(pdf_file: Path) -> tuple[str, str]:
    stem = pdf_file.stem
    parts = stem.split("_")
    if parts and parts[0].isdigit():
        year = parts[0]
        company = parts[-1]
    elif len(parts) >= 2:
        company = parts[0]
        year = parts[1]
    else:
        company = stem
        year = ""
    return company, year


def ingest_document(pdf_path: str, chunk_embeddings, store_embeddings, vector_store) -> None:
    pdf_file = Path(pdf_path)
    company, year = parse_company_year(pdf_file)
    print(f"Ingesting {pdf_file.name} as company={company!r}, year={year!r}")

    converter = PDFToMarkdownConverter()
    markdown_file = converter.convert_pdf(pdf_path=pdf_path, output_dir="data/markdown")

    chunks = chunk_markdown(markdown_file=markdown_file, embeddings=chunk_embeddings)
    print(f"Generated {len(chunks)} chunks for {pdf_file.name}")

    vector_store.upload_chunks(
        chunks=chunks,
        embeddings=store_embeddings,
        company=company,
        year=year,
        source_file=pdf_file.name
    )

    # Run a lightweight BM25 keyword search over the newly-uploaded chunks.
    # We use the BM25 retriever which queries the Chroma collection for
    # documents filtered by company/year and ranks them by keyword relevance.
    try:
        bm25_retriever = BM25Retriever(vector_store.collection)
        bm25_results = bm25_retriever.invoke(
            query=f"{company} financial metrics",
            company=company,
            year=int(year) if year.isdigit() else None,
            top_k=8,
        )
    except Exception:
        bm25_results = []

    metrics = extract_financial_metrics(
        retriever=ChromaRetriever(vector_store.client),
        company=company,
        year=int(year) if year.isdigit() else None,
        candidate_docs=bm25_results,
    )

    if metrics:
        save_metrics(company=company, year=int(year) if str(year).isdigit() else None, metrics=metrics)


def ingest_directory(input_dir: str) -> None:
    print(f"Looking for PDFs in: {Path(input_dir).resolve()}")

    chunk_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    store_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

    vector_store = ChromaVectorStore(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embeddings=store_embeddings
    )

    pdf_files = list(Path(input_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_file in pdf_files:
        ingest_document(
            pdf_path=str(pdf_file),
            chunk_embeddings=chunk_embeddings,
            store_embeddings=store_embeddings,
            vector_store=vector_store
        )


if __name__ == "__main__":
    ingest_directory("data/raw_pdfs")