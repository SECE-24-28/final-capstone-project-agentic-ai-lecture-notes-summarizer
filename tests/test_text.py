import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text import answer_question, summarize_notes
from textPDF import answer_pdf_question, summarize_pdf_document


def test_summarize_notes_creates_a_contextual_summary(monkeypatch):
    notes = (
        "Artificial intelligence is transforming healthcare. "
        "It improves diagnosis through data analysis. "
        "It also automates routine tasks. "
        "Hospitals use it to support doctors."
    )

    monkeypatch.setattr("text._get_llm_summary", lambda notes_value: None)

    summary = summarize_notes(notes)

    assert summary.lower() != notes.split(".")[0].lower()
    assert "healthcare" in summary.lower()
    assert "diagnosis" in summary.lower() or "routine" in summary.lower()
    assert summary.lower().startswith("title:")
    assert "summary:" in summary.lower()
    assert len(summary.splitlines()) >= 2


def test_summarize_notes_uses_llm_summary_when_available(monkeypatch):
    notes = "Artificial intelligence is transforming healthcare."

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=10):
        return FakeResponse({"choices": [{"message": {"content": "A polished LLM summary."}}]})

    monkeypatch.setattr("text.urllib_request.urlopen", fake_urlopen)

    summary = summarize_notes(notes, use_llm=True)

    assert "polished llm summary" in summary.lower()
    assert "summary:" in summary.lower()


def test_summarize_notes_uses_gemini_prompt_for_academic_summary(monkeypatch):
    notes = "Art can challenge political authority during social change."

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": "Main Topic: art and politics\n\nSummary: A concise academic summary."}]}}]})

    monkeypatch.setattr("text.urllib_request.urlopen", fake_urlopen)

    summary = summarize_notes(notes, use_llm=True)

    assert "main topic" in summary.lower()
    assert "summary" in summary.lower()
    assert "gemini-2.5-flash" in captured["url"]
    assert "main topic" in captured["payload"]["contents"][0]["parts"][0]["text"].lower()


def test_summarize_notes_uses_groq_when_available(monkeypatch):
    notes = "Artificial intelligence is transforming healthcare."

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "A polished Groq summary."}}]})

    monkeypatch.setattr("text.urllib_request.urlopen", fake_urlopen)

    summary = summarize_notes(notes, use_llm=True)

    assert "polished groq summary" in summary.lower()
    assert "summary:" in summary.lower()
    assert "api.groq.com" in captured["url"]
    assert "llama-3.1-8b-instant" in captured["payload"]["model"]


def test_answer_question_uses_openai_when_available(monkeypatch):
    notes = "Artificial intelligence is transforming healthcare."

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=10):
        return FakeResponse({"choices": [{"message": {"content": "Artificial intelligence is a branch of computer science."}}]})

    monkeypatch.setattr("text.urllib_request.urlopen", fake_urlopen)

    answer = answer_question("What is artificial intelligence?", notes, use_llm=True)

    assert "branch of computer science" in answer.lower()


def test_answer_question_uses_gemini_when_available(monkeypatch):
    notes = "Artificial intelligence is transforming healthcare."

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": "AI stands for artificial intelligence."}]}}]})

    monkeypatch.setattr("text.urllib_request.urlopen", fake_urlopen)

    answer = answer_question("What is AI?", notes, use_llm=True)

    assert "ai stands for artificial intelligence" in answer.lower()
    assert "gemini-2.5-flash" in captured["url"]


def test_answer_question_uses_groq_when_available(monkeypatch):
    notes = "Artificial intelligence is transforming healthcare."

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "Groq answers: AI is artificial intelligence."}}]})

    monkeypatch.setattr("text.urllib_request.urlopen", fake_urlopen)

    answer = answer_question("What is AI?", notes, use_llm=True)

    assert "groq answers" in answer.lower()
    assert "api.groq.com" in captured["url"]
    assert "llama-3.1-8b-instant" in captured["payload"]["model"]


def test_answer_question_prefers_the_sentence_about_the_requested_topic():
    notes = (
        "The speaker addresses many questions about public policy. "
        "The lecture discusses the role of art in social change."
    )

    answer = answer_question("Does it talk about art?", notes, use_llm=False)

    assert "art" in answer.lower()
    assert "social" in answer.lower() or "change" in answer.lower()


def test_summarize_pdf_document_uses_extracted_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "lecture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "textPDF.extract_text_from_pdf",
        lambda path: "Artificial intelligence improves diagnostic accuracy. It also automates routine tasks.",
    )

    summary = summarize_pdf_document(pdf_path, use_llm=False)

    assert "diagnostic" in summary.lower() or "routine" in summary.lower()
    assert summary.lower().startswith("title:")
    assert "summary:" in summary.lower()


def test_answer_pdf_question_uses_extracted_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "lecture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "textPDF.extract_text_from_pdf",
        lambda path: "Artificial intelligence improves diagnostic accuracy. It also automates routine tasks.",
    )

    answer = answer_pdf_question("How does AI improve diagnostics?", pdf_path, use_llm=False)

    assert "diagnostic" in answer.lower() or "routine" in answer.lower()


def test_answer_question_fallback_heuristic_prefers_definition():
    notes = (
        "It would be impossible to overestimate the importance of photosynthesis in the maintenance of life on Earth. "
        "Photosynthesis completely transformed Earth’s environment and biosphere. "
        "If photosynthesis ceased, there would soon be little food."
    )
    answer = answer_question("what is photosynthesis", notes, use_llm=False)
    assert answer == "Photosynthesis completely transformed Earth’s environment and biosphere."
