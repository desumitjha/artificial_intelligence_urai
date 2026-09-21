"""Small, dependency-free building blocks for a RAG pipeline."""
import math
from typing import List, Sequence, Tuple


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """Split text into paragraph-level chunks of at most max_chars characters.

    Paragraphs are kept whole when possible. A paragraph longer than max_chars
    is split on sentence-like boundaries so no chunk exceeds the limit.
    """
    chunks: List[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        current = ""
        for sentence in paragraph.replace("\n", " ").split(". "):
            piece = sentence if sentence.endswith(".") else sentence + "."
            if current and len(current) + len(piece) + 1 > max_chars:
                chunks.append(current.strip())
                current = ""
            current += " " + piece
        if current.strip():
            chunks.append(current.strip())
    return chunks


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors: 1 = same direction, 0 = unrelated."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k(query_vec: Sequence[float], doc_vecs: Sequence[Sequence[float]], k: int = 3) -> List[Tuple[int, float]]:
    """Return (index, score) for the k most similar document vectors, best first."""
    scored = [(i, cosine_similarity(query_vec, v)) for i, v in enumerate(doc_vecs)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def build_prompt(question: str, context_chunks: Sequence[str]) -> str:
    """Augment the question with the retrieved context (the 'A' in RAG)."""
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )


def embed_texts(client, model: str, texts: Sequence[str]) -> List[List[float]]:
    """Turn texts into embedding vectors with the Gemini embedding API."""
    result = client.models.embed_content(model=model, contents=list(texts))
    return [list(e.values) for e in result.embeddings]
