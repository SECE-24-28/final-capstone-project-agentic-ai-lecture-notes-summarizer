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
    assert len(summary.split()) <= 20
