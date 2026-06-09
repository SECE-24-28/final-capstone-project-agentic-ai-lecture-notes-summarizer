import re
from collections import Counter


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


def summarize_notes(notes, max_sentences=4):
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
    return " ".join(top_sentences)


def answer_question(question, notes):
    if not question.strip():
        return "Please ask a question about the notes."

    lowered_question = question.lower()
    if "summary" in lowered_question or "summarize" in lowered_question:
        return summarize_notes(notes)

    sentences = split_into_sentences(notes)
    question_tokens = {
        token for token in tokenize(question) if token not in STOP_WORDS and len(token) > 2
    }

    best_sentence = "I could not find a clear answer in the provided notes."
    best_score = 0

    for sentence in sentences:
        sentence_tokens = {
            token for token in tokenize(sentence) if token not in STOP_WORDS and len(token) > 2
        }
        overlap = len(question_tokens & sentence_tokens)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence

    return best_sentence


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


