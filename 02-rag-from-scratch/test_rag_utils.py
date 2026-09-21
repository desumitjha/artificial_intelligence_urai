import math

import pytest

from rag_utils import build_prompt, chunk_text, cosine_similarity, top_k


def test_cosine_identical_vectors_is_one():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine_similarity([1, 1], [-1, -1]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_safe():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_top_k_orders_best_first():
    docs = [[0, 1], [1, 0], [1, 1]]
    result = top_k([1, 0], docs, k=2)
    assert [i for i, _ in result] == [1, 2]
    assert result[0][1] == pytest.approx(1.0)
    assert result[1][1] == pytest.approx(1 / math.sqrt(2))


def test_chunk_text_keeps_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."
    assert chunk_text(text) == ["First paragraph.", "Second paragraph."]


def test_chunk_text_respects_limit():
    long_par = " ".join(f"Sentence number {i}." for i in range(40))
    chunks = chunk_text(long_par, max_chars=120)
    assert len(chunks) > 1
    assert all(len(c) <= 140 for c in chunks)


def test_build_prompt_contains_question_and_context():
    prompt = build_prompt("What is X?", ["X is a thing."])
    assert "What is X?" in prompt
    assert "[1] X is a thing." in prompt
    assert "ONLY the context" in prompt
