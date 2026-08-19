from pathlib import Path
from langchain_core.documents import Document
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.
    Args:
        markdown_file: Markdown file path.
    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown.
    Args:
        markdown_file: Markdown file path.
        embeddings: Gemini embedding model.
    Returns:
        List of semantic chunks.
    """
    markdown_content = read_markdown(markdown_file)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    return splitter.create_documents([markdown_content])


if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing Gemini credentials. Set GOOGLE_API_KEY in .env."
        )

    markdown_file = "../data/markdown/2024_Apple.md"

    embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2",
       model_kwargs={"device": "cpu"},
       encode_kwargs={"normalize_embeddings": True}
       )
    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks\n")
    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()