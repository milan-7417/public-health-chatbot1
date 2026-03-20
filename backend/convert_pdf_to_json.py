import os
import json
from pypdf import PdfReader

PDF_FOLDER = "backend/data/pdfs"
OUTPUT_FILE = "backend/data/docs.json"

CHUNK_SIZE = 500  # characters


def clean_text(text):
    return text.replace("\n", " ").strip()


def chunk_text(text, size=500):
    chunks = []
    for i in range(0, len(text), size):
        chunk = text[i:i + size]
        if len(chunk) > 100:  # ignore tiny chunks
            chunks.append(chunk)
    return chunks


def extract_text_from_pdfs():
    all_chunks = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            reader = PdfReader(path)

            print(f"Processing: {file}")

            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""

            full_text = clean_text(full_text)

            chunks = chunk_text(full_text, CHUNK_SIZE)

            for chunk in chunks:
                all_chunks.append({"text": chunk})

    return all_chunks


def main():
    documents = extract_text_from_pdfs()

    print(f"Total chunks: {len(documents)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print("✅ docs.json created successfully!")


if __name__ == "__main__":
    main()