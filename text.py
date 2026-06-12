import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib import request as urllib_request


def _load_env_file():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


def get_notes_input():
    print("Enter your lecture notes. Press Enter on a blank line to finish:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            if lines:
                break
            print("No notes entered. Please type your notes and press Enter.")
            continue
        lines.append(line.strip())
    return "\n".join(lines)


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "below", "but",
    "by", "can", "contain", "contains", "during", "each", "eating", "for", "from",
    "gives", "good", "have", "has", "health", "helps", "help", "her", "his", "in",
    "include", "includes", "important", "into", "is", "it", "its", "just", "like",
    "many", "more", "much", "not", "of", "often", "on", "or", "our", "over", "provide",
    "provides", "should", "some", "such", "that", "the", "their", "them", "these",
    "this", "those", "to", "too", "use", "used", "variety", "very", "was", "were",
    "when", "which", "while", "with", "you", "your"
}


def split_into_sentences(text_value):
    sentences = re.split(r"(?<=[.!?])\s+", text_value.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize(text_value):
    return re.findall(r"[a-zA-Z]+", text_value.lower())


def normalize_token(token):
    normalized = token.lower()
    if len(normalized) > 5 and normalized.endswith("ing"):
        normalized = normalized[:-3]
    elif len(normalized) > 4 and normalized.endswith("ed"):
        normalized = normalized[:-2]
    elif len(normalized) > 4 and normalized.endswith("es") and not normalized.endswith(("ses", "xes", "zes")):
        normalized = normalized[:-2]
    elif len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith(("ss", "us", "is")):
        normalized = normalized[:-1]
    return normalized


def _compress_sentence(sentence, max_terms=6):
    tokens = [token for token in tokenize(sentence) if token not in STOP_WORDS]
    if not tokens:
        return ""

    compressed_tokens = []
    for token in tokens:
        if token not in compressed_tokens and len(token) > 2:
            compressed_tokens.append(token)
        if len(compressed_tokens) >= max_terms:
            break
    return " ".join(compressed_tokens)


def _build_academic_summary_prompt(notes):
    return (
        "You are a careful academic study assistant. Write a concise academic summary of the lecture notes in clear, student-friendly language.\n"
        "Follow these rules:\n"
        "- Identify the main topic in one sentence.\n"
        "- Explain the key ideas using your own words.\n"
        "- Mention important arguments or concepts.\n"
        "- End with a final takeaway.\n"
        "- Avoid copying large chunks of the original text.\n"
        "- Keep the response around 80-150 words.\n"
        "Format the response as:\n"
        "Main Topic:\n"
        "<one sentence>\n\n"
        "Summary:\n"
        "<one paragraph>\n\n"
        f"Lecture notes:\n{notes}"
    )


def _get_llm_summary(notes):
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider in {"openai", "gpt"}:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful academic assistant that summarizes lecture notes clearly, concisely, and in polished prose.",
                },
                {
                    "role": "user",
                    "content": _build_academic_summary_prompt(notes),
                },
            ],
            "temperature": 0.3,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    elif provider in {"gemini", "google", "google_gemini"}:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": _build_academic_summary_prompt(notes),
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 220},
        }
        headers = {"Content-Type": "application/json"}
    else:
        return None

    try:
        request = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except Exception:
        return None

    if provider in {"openai", "gpt"}:
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return None

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None


def summarize_notes(notes, max_sentences=3, use_llm=True):
    if use_llm:
        llm_summary = _get_llm_summary(notes)
        if llm_summary:
            return llm_summary

    sentences = split_into_sentences(notes)
    if not sentences:
        return "No notes were provided."

    word_counts = Counter(
        token for sentence in sentences for token in tokenize(sentence) if token not in STOP_WORDS
    )

    scored_sentences = []
    for sentence in sentences:
        tokens = [token for token in tokenize(sentence) if token not in STOP_WORDS]
        if not tokens:
            continue
        score = sum(word_counts[token] for token in tokens) + len(tokens) / 3
        scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    top_sentences = [sentence for _, sentence in scored_sentences[:max_sentences]]
    top_sentences.sort(key=sentences.index)

    if not top_sentences:
        return "The notes describe several key ideas."

    first_sentence = top_sentences[0].strip().rstrip(".")
    summary_sentences = [f"The notes explain that {first_sentence.lower()}."]

    if len(top_sentences) > 1:
        second_sentence = top_sentences[1].strip().rstrip(".")
        summary_sentences.append(f"They also highlight {second_sentence.lower()}.")

    if len(top_sentences) > 2:
        third_sentence = top_sentences[2].strip().rstrip(".")
        summary_sentences.append(f"Overall, the main takeaway is that {third_sentence.lower()}.")

    return "\n".join(summary_sentences)


def answer_question(question, notes):
    if not question.strip():
        return "Please ask a question about the notes."

    lowered_question = question.lower()
    if "summary" in lowered_question or "summarize" in lowered_question:
        return summarize_notes(notes)

    sentences = split_into_sentences(notes)
    question_tokens = {
        normalize_token(token)
        for token in tokenize(question)
        if token not in STOP_WORDS and len(token) > 2
    }

    best_sentence = "I could not find a clear answer in the provided notes."
    best_score = 0

    for sentence in sentences:
        sentence_tokens = {
            normalize_token(token)
            for token in tokenize(sentence)
            if token not in STOP_WORDS and len(token) > 2
        }
        overlap = len(question_tokens & sentence_tokens)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence

    return best_sentence


def main():
    notes = get_notes_input()
    print("\nSummary of the notes:")
    print(summarize_notes(notes))
    print("\nAsk questions about the notes. Type 'exit' to stop.\n")

    while True:
        user_question = input("Question: ").strip()
        if not user_question or user_question.lower() in {"exit", "quit"}:
            break
        print(answer_question(user_question, notes))
        print()


if __name__ == "__main__":
    main()


