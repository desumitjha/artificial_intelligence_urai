"""Summarize a text file with the Google Gemini API.

Usage:
    python summarize.py                 # summarizes sample_notes.txt
    python summarize.py my_notes.txt    # summarizes your own file
"""
import os
import sys

from dotenv import load_dotenv
from google import genai

DEFAULT_FILE = "sample_notes.txt"
DEFAULT_MODEL = "gemini-3.7-flash"  # override with GEMINI_MODEL in your .env


def read_file(path: str) -> str:
    """Return the text inside a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(text: str) -> str:
    """Wrap the document in clear instructions for the model."""
    return (
        "Summarize the following text in 3 to 5 short bullet points. "
        "Use simple words and keep every important fact.\n\n"
        f"TEXT:\n{text}"
    )


def summarize(client: genai.Client, model: str, text: str) -> str:
    """Send the prompt to Gemini and return the summary."""
    response = client.models.generate_content(model=model, contents=build_prompt(text))
    return response.text


def main() -> int:
    load_dotenv()  # reads GEMINI_API_KEY from the .env file
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY. Copy .env.example to .env and add your key.")
        return 1

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    try:
        text = read_file(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return 1
    if not text.strip():
        print(f"File is empty: {path}")
        return 1

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    print(summarize(client, model, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
