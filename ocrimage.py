import cv2
import easyocr
from collections import Counter
import re


# -----------------------------
# OCR SECTION
# -----------------------------

def preprocess_image(image_path):
    """
    Preprocess image for better OCR accuracy.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    return thresh


def extract_text_from_image(image_path):
    """
    Extract handwritten text from image.
    """

    processed = preprocess_image(image_path)

    reader = easyocr.Reader(['en'])

    results = reader.readtext(processed)

    text = ""

    for result in results:
        text += result[1] + " "

    return text.strip()


# -----------------------------
# FALLBACK SUMMARIZER
# -----------------------------

def summarize_text(text, num_sentences=3):

    if not text:
        return "No text found."

    sentences = re.split(r'(?<=[.!?]) +', text)

    if len(sentences) <= num_sentences:
        return text

    words = re.findall(r'\w+', text.lower())

    stop_words = {
        "the", "is", "a", "an", "and", "or", "in",
        "on", "at", "to", "for", "of", "with",
        "by", "from", "that", "this", "it",
        "as", "are", "was", "were"
    }

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    word_freq = Counter(filtered_words)

    sentence_scores = {}

    for sentence in sentences:

        sentence_words = re.findall(
            r'\w+',
            sentence.lower()
        )

        score = sum(
            word_freq.get(word, 0)
            for word in sentence_words
        )

        sentence_scores[sentence] = score

    summary_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:num_sentences]

    return " ".join(summary_sentences)


# -----------------------------
# SIMPLE QA
# -----------------------------

def answer_question(context, question):

    sentences = re.split(
        r'(?<=[.!?]) +',
        context
    )

    question_words = set(
        question.lower().split()
    )

    best_sentence = ""
    best_score = 0

    for sentence in sentences:

        sentence_words = set(
            sentence.lower().split()
        )

        score = len(
            question_words.intersection(
                sentence_words
            )
        )

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score == 0:
        return "Answer not found in notes."

    return best_sentence


# -----------------------------
# MAIN PROGRAM
# -----------------------------

def main():

    print("\n===== OCR IMAGE SUMMARIZER =====\n")

    image_path = input(
        "Enter image path: "
    ).strip()

    try:

        print("\nExtracting text...\n")

        notes = extract_text_from_image(
            image_path
        )

        if not notes:
            print(
                "No text detected in image."
            )
            return

        print("===== EXTRACTED TEXT =====\n")
        print(notes)

        summary = summarize_text(notes)

        print("\n===== SUMMARY =====\n")
        print(summary)

        while True:

            question = input(
                "\nAsk a question (or type exit): "
            )

            if question.lower() == "exit":
                break

            answer = answer_question(
                notes,
                question
            )

            print("\nAnswer:")
            print(answer)

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()