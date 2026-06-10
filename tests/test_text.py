import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text import summarize_notes


def test_summarize_notes_creates_a_contextual_summary():
    notes = (
        "Artificial intelligence is transforming healthcare. "
        "It improves diagnosis through data analysis. "
        "It also automates routine tasks. "
        "Hospitals use it to support doctors."
    )

    summary = summarize_notes(notes)

    assert summary.lower() != notes.split(".")[0].lower()
    assert "healthcare" in summary.lower()
    assert "diagnosis" in summary.lower() or "routine" in summary.lower()
    assert summary.lower().startswith("the notes")
    assert len(summary.split(".")) >= 3 or "\n" in summary


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
