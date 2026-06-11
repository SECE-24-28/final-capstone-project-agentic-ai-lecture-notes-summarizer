import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def extract_text(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6"
    )

    return text


if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    )

    text = extract_text(image_path)

    print("\nExtracted Text:\n")
    print(text)