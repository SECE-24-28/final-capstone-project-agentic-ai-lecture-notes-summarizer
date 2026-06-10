import argparse
from pathlib import Path
from typing import Optional

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - depends on environment
    try:
        from PyPDF2 import PdfReader
    except ImportError:  # pragma: no cover - depends on environment
        PdfReader = None

from text import answer_question, summarize_notes

DEFAULT_PDF_PATH = Path(
    r"C:\Users\devad\github-classroom\SECE-24-28\final-capstone-project-agentic-ai-lecture-notes-summarizer\Sample pdf.pdf"
)


def resolve_pdf_path(pdf_path: Optional[str | Path] = None) -> Path:
    if pdf_path is not None:
        candidate = Path(pdf_path)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"The PDF file was not found: {candidate}")
        return candidate

    if DEFAULT_PDF_PATH.exists():
        return DEFAULT_PDF_PATH

    workspace_dir = Path(__file__).resolve().parent
    candidates = sorted(workspace_dir.glob("*.pdf"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError("No PDF file was found in the workspace directory. Place a PDF there or pass a path.")


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    if PdfReader is None:
        raise ImportError("Install 'pypdf' to read PDF files. For example: pip install pypdf")

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    return "\n".join(pages)


def summarize_pdf_document(pdf_path: Optional[str | Path] = None, use_llm: bool = True) -> str:
    path = resolve_pdf_path(pdf_path)
    notes = extract_text_from_pdf(path)
    if not notes.strip():
        return "The uploaded PDF did not contain readable text."
    return summarize_notes(notes, use_llm=use_llm)


def answer_pdf_question(question: str, pdf_path: Optional[str | Path] = None) -> str:
    path = resolve_pdf_path(pdf_path)
    notes = extract_text_from_pdf(path)
    if not notes.strip():
        return "The uploaded PDF did not contain readable text."
    return answer_question(question, notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a lecture PDF and answer questions about it.")
    parser.add_argument("pdf_path", nargs="?", help="Optional path to a PDF file")
    args = parser.parse_args()

    try:
        pdf_path = resolve_pdf_path(args.pdf_path)
    except FileNotFoundError as exc:
        print(exc)
        return

    try:
        notes = extract_text_from_pdf(pdf_path)
    except ImportError as exc:
        print(exc)
        return

    if not notes.strip():
        print("The uploaded PDF did not contain readable text.")
        return

    print(f"Loaded PDF: {pdf_path}")
    print("\nSummary of the PDF:")
    print(summarize_notes(notes))
    print("\nAsk questions about the PDF. Type 'exit' to stop.\n")

    while True:
        user_question = input("Question: ").strip()
        if not user_question or user_question.lower() in {"exit", "quit"}:
            break
        print(answer_question(user_question, notes))
        print()


if __name__ == "__main__":
    main()
