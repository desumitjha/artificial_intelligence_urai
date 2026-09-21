# artificial_intelligence_urai

Hands-on projects from my journey into applied AI, built step by step with Python and the Google Gemini API.
I come from 10+ years of data engineering and cloud architecture (Azure, Databricks) and I learn by building small,
finished projects and writing down what each one taught me.

| # | Project | What it shows | Status |
|---|---------|---------------|--------|
| 1 | [Document summarizer](01-document-summarizer/) | Calling an LLM API from a Python command-line tool, config and secrets handling | Done |
| 2 | [RAG from scratch](02-rag-from-scratch/) | Chunking, embeddings, cosine-similarity retrieval, grounded answers | Done |
| 3 | Self-healing RAG | Checks its own retrieved context and retries when the answer is weak | Planned |
| 4 | LLM guardrails gateway | Input and output checks around an LLM call | Planned |
| 5 | LLM evaluation pipeline | Automatic tests for LLM answers | Planned |

## Run any project

Each project folder has its own README, `requirements.txt` and `.env.example`.

```bash
cd 01-document-summarizer            # or 02-rag-from-scratch
python -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # then put your own key in .env
```

You need a free API key from [Google AI Studio](https://aistudio.google.com/). The `.env` file is ignored by Git, so your key is never uploaded.

## Tests

Every project has unit tests that run without an API key. GitHub Actions runs them on each push (see `.github/workflows/tests.yml`).

## Author

Sumit Kumar Jha · [Portfolio](https://desumitjha.github.io) · [LinkedIn](https://www.linkedin.com/in/sumit-j-b978b1157/)
