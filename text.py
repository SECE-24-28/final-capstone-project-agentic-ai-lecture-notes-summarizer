import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib import error as urllib_error
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
        "- Identify the main topic in one short sentence.\n"
        "- Explain the key ideas using your own words.\n"
        "- Mention important arguments or concepts.\n"
        "- End with a final takeaway.\n"
        "- Avoid copying large chunks of the original text.\n"
        "- Keep the response around 80-150 words.\n"
        "Format the response as:\n"
        "Title:\n"
        "<one short title>\n\n"
        "Main Topic:\n"
        "<one sentence>\n\n"
        "Summary:\n"
        "<one paragraph>\n\n"
        f"Lecture notes:\n{notes}"
    )


def _get_llm_summary(notes):
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider in {"openai", "gpt", "groq"}:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            url = "https://api.groq.com/openai/v1/chat/completions"
            model = "llama-3.1-8b-instant"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            url = "https://api.openai.com/v1/chat/completions"
            model = "gpt-4o-mini"

        if not api_key:
            return None

        payload = {
            "model": model,
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
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
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
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
    except urllib_error.HTTPError as e:
        if e.code == 429:
            print(f"\n[Warning: {provider.upper()} API rate limit or quota exceeded. Falling back to rule-based summarizer.]\n", file=sys.stderr)
        else:
            print(f"\n[Warning: {provider.upper()} API returned HTTP error {e.code}. Falling back to rule-based summarizer.]\n", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\n[Warning: Failed to contact LLM provider ({e}). Falling back to rule-based summarizer.]\n", file=sys.stderr)
        return None

    if provider in {"openai", "gpt", "groq"}:
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return None

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None


def _get_llm_answer(question, notes):
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    prompt = (
        "You are an academic assistant. Answer the user's question using only the provided lecture notes. "
        "Keep the answer concise, accurate, and student-friendly. If the answer cannot be found in the notes, say so.\n\n"
        f"Lecture Notes:\n{notes}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    if provider in {"openai", "gpt", "groq"}:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            url = "https://api.groq.com/openai/v1/chat/completions"
            model = "llama-3.1-8b-instant"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            url = "https://api.openai.com/v1/chat/completions"
            model = "gpt-4o-mini"

        if not api_key:
            return None

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful academic assistant that answers questions based on lecture notes.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.3,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
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
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
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
    except urllib_error.HTTPError as e:
        if e.code == 429:
            print(f"\n[Warning: {provider.upper()} API rate limit or quota exceeded. Falling back to rule-based Q&A.]\n", file=sys.stderr)
        else:
            print(f"\n[Warning: {provider.upper()} API returned HTTP error {e.code}. Falling back to rule-based Q&A.]\n", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\n[Warning: Failed to contact LLM provider ({e}). Falling back to rule-based Q&A.]\n", file=sys.stderr)
        return None

    if provider in {"openai", "gpt", "groq"}:
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return None

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None


def _derive_title(notes, fallback_text=""):
    source_text = fallback_text or notes
    sentences = split_into_sentences(source_text)
    if not sentences:
        sentences = split_into_sentences(notes)

    if not sentences:
        return "Lecture Notes Summary"

    word_counts = Counter(
        token for sentence in sentences for token in tokenize(sentence) if token not in STOP_WORDS and len(token) > 2
    )

    keyword_candidates = [
        token for token, _ in word_counts.most_common(6) if token not in {"lecture", "notes", "summary", "topic"}
    ]

    if keyword_candidates:
        main_keyword = keyword_candidates[0].capitalize()
        if len(keyword_candidates) > 1:
            secondary_keyword = keyword_candidates[1].capitalize()
            if secondary_keyword.lower() not in {"and", "the", "for", "with"}:
                return f"{main_keyword} and {secondary_keyword}"
        return main_keyword

    return "Lecture Notes Summary"


def _build_fallback_summary(notes, max_sentences=3):
    sentences = split_into_sentences(notes)
    if not sentences:
        return "Title: Lecture Notes Summary\n\nSummary: No notes were provided."

    if len(sentences) <= 2:
        target_sentences = 1
    elif len(sentences) <= 4:
        target_sentences = 2
    else:
        target_sentences = min(max_sentences, len(sentences))

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
    top_sentences = [sentence for _, sentence in scored_sentences[:target_sentences]]
    top_sentences.sort(key=sentences.index)

    if not top_sentences:
        return "Title: Lecture Notes Summary\n\nSummary: The notes describe several key ideas."

    paragraph_sentences = []
    for index, sentence in enumerate(top_sentences):
        cleaned_sentence = sentence.strip().rstrip(".")
        if index == 0:
            paragraph_sentences.append(cleaned_sentence)
        elif index == 1:
            paragraph_sentences.append(f"It also explains that {cleaned_sentence.lower()}.")
        else:
            paragraph_sentences.append(f"Overall, it shows that {cleaned_sentence.lower()}.")

    paragraph = " ".join(paragraph_sentences)
    title = _derive_title(notes, paragraph)
    return f"Title: {title}\n\nSummary: {paragraph}"


def _normalize_summary_output(notes, llm_summary):
    if not llm_summary or not llm_summary.strip():
        return _build_fallback_summary(notes)

    # Match sections using regex on the raw unflattened response
    pattern = r"(?i)(title|main\s+topic|summary):\s*(.*?)(?=\s*(?:title|main\s+topic|summary):|$)"
    matches = re.findall(pattern, llm_summary, re.DOTALL)
    
    sections = {}
    for key, val in matches:
        k = key.lower().strip()
        v = re.sub(r"\s+", " ", val).strip()
        sections[k] = v

    if "title" in sections and "summary" in sections:
        if "main topic" in sections:
            return f"Title: {sections['title']}\n\nMain Topic: {sections['main topic']}\n\nSummary: {sections['summary']}"
        return f"Title: {sections['title']}\n\nSummary: {sections['summary']}"

    if "main topic" in sections and "summary" in sections:
        return f"Title: {sections['main topic']}\n\nMain Topic: {sections['main topic']}\n\nSummary: {sections['summary']}"

    cleaned_summary = re.sub(r"\s+", " ", llm_summary).strip()
    title = _derive_title(notes, cleaned_summary)
    return f"Title: {title}\n\nSummary: {cleaned_summary}"


def summarize_notes(notes, max_sentences=3, use_llm=True):
    if use_llm:
        llm_summary = _get_llm_summary(notes)
        if llm_summary:
            return _normalize_summary_output(notes, llm_summary)

    return _build_fallback_summary(notes, max_sentences=max_sentences)


def _extract_question_keywords(question):
    generic_words = {
        "about", "after", "all", "also", "an", "and", "any", "are", "as", "ask",
        "asked", "be", "been", "being", "can", "could", "did", "do", "does",
        "for", "from", "have", "has", "how", "is", "it", "its", "may", "mean",
        "means", "mention", "mentioned", "not", "of", "or", "our", "question",
        "questions", "should", "talk", "that", "the", "their", "them", "there",
        "this", "those", "to", "what", "when", "where", "which", "who", "why",
        "would", "you", "your"
    }
    tokens = [
        normalize_token(token)
        for token in tokenize(question)
        if token not in STOP_WORDS and len(token) > 2
    ]
    return [token for token in tokens if token not in generic_words and len(token) > 2]


def answer_question(question, notes, use_llm=True):
    if not question.strip():
        return "Please ask a question about the notes."

    lowered_question = question.lower()
    if "summary" in lowered_question or "summarize" in lowered_question:
        return summarize_notes(notes, use_llm=use_llm)

    if use_llm:
        llm_answer = _get_llm_answer(question, notes)
        if llm_answer:
            return llm_answer

    sentences = split_into_sentences(notes)
    question_keywords = _extract_question_keywords(question)

    if not question_keywords:
        return "I could not find a clear answer in the provided notes."

    best_sentence = "I could not find a clear answer in the provided notes."
    best_score = -1.0

    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_tokens = [
            normalize_token(token)
            for token in tokenize(sentence)
            if token not in STOP_WORDS and len(token) > 2
        ]
        
        # Calculate overlap
        overlap = sum(1 for keyword in question_keywords if keyword in sentence_tokens)
        if overlap == 0:
            continue
            
        score = float(overlap)
        
        # 1. Boost if definition pattern matched for definition-seeking questions
        is_definition_question = any(q_word in lowered_question for q_word in ["what", "define", "definition", "who", "explain"])
        if is_definition_question:
            for kw in question_keywords:
                pattern = r"\b" + re.escape(kw) + r"\b\s+(?:is|are|was|were|refers|refers\s+to|means|defined\s+as|completely|transformed|produces|creates)\b"
                if re.search(pattern, sentence_lower):
                    score += 2.0
                elif any(word in sentence_lower for word in ["is a", "refers to", "process", "defined"]):
                    score += 0.5

        # 2. Keyword density / sentence length boost
        if sentence_tokens:
            score += overlap / len(sentence_tokens)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score < 0:
        return "I could not find a clear answer in the provided notes."

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


