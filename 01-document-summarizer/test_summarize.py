from summarize import build_prompt, read_file


def test_build_prompt_contains_text_and_instruction():
    prompt = build_prompt("Hello world")
    assert "Hello world" in prompt
    assert "Summarize" in prompt


def test_read_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("abc", encoding="utf-8")
    assert read_file(str(f)) == "abc"
