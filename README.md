# Lecture Notes Summarizer

A Python-based assistant for turning lecture notes into concise, readable summaries and answering questions about the content.

## What the project does

This project supports three main workflows:

1. Manual text input
   - Users enter lecture notes directly in the terminal.
   - The app generates a concise summary.
   - Users can ask questions about the notes.

2. PDF input
   - Users provide a PDF file.
   - The text is extracted from the document.
   - A summary is generated and questions can be asked about the content.

3. Image input
   - Users provide an image of handwritten notes.
   - Gemini OCR extracts the handwritten text.
   - The extracted notes are summarized and can be queried.

## Current features

- Summarization of lecture notes from plain text
- PDF text extraction and summarization
- Handwritten note OCR from images
- Question answering based on the extracted or entered notes
- Academic-style summaries that focus on the main topic, key ideas, and takeaway
- A built-in fallback summarizer when LLM access is unavailable

## Technologies used

- Python 3.12+
- Gemini API for OCR and summarization
- PDF extraction via pypdf
- Standard Python libraries for CLI interaction

## Setup

1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
   - or `uv sync`
4. Create a `.env` file in the project root with one of the following:
   - `GEMINI_API_KEY=your_key`
   - `GOOGLE_API_KEY=your_key`
5. Optionally set:
   - `LLM_PROVIDER=gemini`

## Run the project

### Text notes
```bash
python text.py
```

### PDF notes
```bash
python textPDF.py path/to/notes.pdf
```

### Image notes
```bash
python ocrimage.py
```

You can also run them with `uv` if preferred:

```bash
uv run text.py
uv run textPDF.py path/to/notes.pdf
uv run ocrimage.py
```

## Testing

Run the test suite with:

```bash
pytest -q
```

## Notes

- The summarization flow was improved to produce clearer, less extractive summaries.
- If Gemini is unavailable, the project keeps using its existing fallback summarizer.
- The core workflows for text, PDF, image, and question answering remain intact.
