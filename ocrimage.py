import json
import os
import base64
import time
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError

from text import (
    summarize_notes,
    answer_question,
    _load_env_file
)

_load_env_file()


def extract_text_from_image(image_path):

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env"
        )

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_bytes = image_path.read_bytes()

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    mime_type = "image/png"

    if image_path.suffix.lower() in [
        ".jpg",
        ".jpeg"
    ]:
        mime_type = "image/jpeg"

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": """
Extract all handwritten text from this image.

Rules:
- Return only the extracted text.
- Do not summarize.
- Do not explain.
- Preserve paragraphs.
- Preserve headings.
- Correct obvious OCR mistakes only.
"""
                    },
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_image
                        }
                    }
                ]
            }
        ]
    }

    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
        method="POST"
    )

    max_retries = 5

    for attempt in range(max_retries):

        try:

            with urllib_request.urlopen(
                request,
                timeout=120
            ) as response:

                body = response.read().decode(
                    "utf-8"
                )

                data = json.loads(body)

                candidates = data.get(
                    "candidates",
                    []
                )

                if not candidates:
                    print(
                        "\nGemini returned:"
                    )
                    print(
                        json.dumps(
                            data,
                            indent=2
                        )
                    )

                    raise Exception(
                        "No text extracted."
                    )

                return (
                    candidates[0]
                    ["content"]["parts"][0]
                    ["text"]
                    .strip()
                )

        except HTTPError as e:

            error_body = ""

            try:
                error_body = (
                    e.read()
                    .decode("utf-8")
                )
            except:
                pass

            print("\n" + "=" * 60)
            print("GEMINI ERROR")
            print("=" * 60)
            print(error_body)
            print("=" * 60)

            if e.code == 429:

                wait_time = (
                    2 ** attempt
                ) * 10

                print(
                    f"\nRate limit hit."
                )

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

            raise

        except Exception as e:

            print(
                f"\nUnexpected error: {e}"
            )

            raise

    raise Exception(
        "Gemini quota/rate limit exceeded after multiple retries."
    )


def main():

    image_path = input(
        "Enter image path: "
    ).strip()

    try:

        print(
            "\nExtracting handwritten text..."
        )

        notes = extract_text_from_image(
            image_path
        )

        print("\n" + "=" * 60)
        print("EXTRACTED NOTES")
        print("=" * 60)
        print(notes)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # LOCAL summarizer
        summary = summarize_notes(
            notes,
            use_llm=False
        )

        print(summary)

        print(
            "\nAsk questions about the notes."
        )

        print(
            "Type 'exit' to stop.\n"
        )

        while True:

            question = input(
                "Question: "
            ).strip()

            if (
                question.lower() == "exit"
                or question.lower() == "quit"
            ):
                break

            answer = answer_question(
                question,
                notes
            )

            print("\n" + answer + "\n")

    except Exception as e:

        print(f"\nError: {e}")


if __name__ == "__main__":
    main()