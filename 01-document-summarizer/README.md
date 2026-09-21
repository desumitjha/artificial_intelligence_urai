# 01 · Document summarizer (Python CLI)

A small command-line tool that reads a text file and asks a Gemini model for a short bullet-point summary.
It was my first step from "calling an API" to a proper little project with configuration, error handling and tests.

## How it works

1. `load_dotenv()` reads `GEMINI_API_KEY` from a local `.env` file (never committed to Git).
2. The file name comes from the command line (`sys.argv`), with `sample_notes.txt` as the default.
3. `build_prompt()` wraps the text in clear instructions.
4. `client.models.generate_content()` sends it to the model and returns the summary.

```
text file  ->  build_prompt  ->  Gemini API  ->  bullet summary
```

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your key
python summarize.py             # uses sample_notes.txt
python summarize.py my_notes.txt
```

Run the tests (no API key needed): `pytest`

## What I learned

- Keep secrets in `.env` and list it in `.gitignore`.
- Handle the boring cases: missing key, missing file, empty file.
- Small pure functions (`read_file`, `build_prompt`) are easy to test without calling the API.

## Ideas to extend

- Summary length and language options (English or German)
- Summarize a web page or PDF
- A simple Gradio front end
