# NLP Intelligence Hub

Unified Sentiment Analysis & Text Summarization web app — built with Gradio and Hugging Face Transformers.

---

## Overview

**NLP Intelligence Hub** is a lightweight Gradio-based web app that provides:
- **Text summarization** (T5-based),
- **Sentiment analysis** (T5-based).

It loads models lazily to reduce startup RAM overhead and uses a small, dark-themed Gradio UI for quick exploration. The app is designed to run either locally or as a Hugging Face Space.  
(Implemented in `app.py`.) :contentReference[oaicite:0]{index=0}

---

## Features

- Paste or type text and get:
  - a concise **summary** (T5 summarizer),
  - a **sentiment label** (positive/negative/other).
- Lazy model loading (models load on first use to save memory).
- Example inputs built into the UI for quick testing.
- Dark, responsive UI with copyable summary output and a sentiment label.

---

## Models

This Space uses the following model IDs (defined in `app.py`):

- `HussainR/t5-summarizer` — summarization model.  
- `HussainR/t5-sentiment-analysis` — sentiment analysis model.  

These are loaded via `AutoTokenizer` and `AutoModelForSeq2SeqLM`. :contentReference[oaicite:1]{index=1}

---

## Files in this repo

- `app.py` — main Gradio app and model manager (lazy loader). :contentReference[oaicite:2]{index=2}  
- `requirements.txt` — runtime dependencies (gradio, transformers, torch, accelerate, sentencepiece, protobuf). :contentReference[oaicite:3]{index=3}  
- `README.md` — (this file)

---

## How it works (implementation notes)

- A small `ModelManager` class handles lazy loading:
  - On first request, it loads the tokenizer and model for the requested model ID and moves the model to the available device (`cuda` if present, otherwise `cpu`).
  - Summarization uses a T5-style prefix `summarize: ` and `model.generate()` with beam search and length constraints.
  - Sentiment analysis also runs generation and inspects the decoded output to determine if it contains `positive` or `negative`. :contentReference[oaicite:4]{index=4}

---

## Run locally

> Recommended: use a Python virtual environment. GPU is optional but recommended for speed.

```bash
# create venv (optional)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate     # Windows

# install dependencies
pip install -r requirements.txt

# run the app
python app.py
