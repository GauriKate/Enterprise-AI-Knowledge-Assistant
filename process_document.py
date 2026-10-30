
from pathlib import Path

from pdf_reader import extract_pages
from text_chunker import split_text


def process_pdf(pdf_path: str) -> list[dict]:
    """Extract PDF pages and create chunks with source metadata."""

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    pages = extract_pages(str(path))

    if not pages:
        raise ValueError(
            "No readable text found in the PDF."
        )

    all_chunks = []
    empty_pages = 0

    for page in pages:
        page_number = page.get("page")
        page_text = page.get("text", "")

        if not isinstance(page_text, str) or not page_text.strip():
            empty_pages += 1
            continue

        chunks = split_text(
            page_text,
            chunk_size=500,
            overlap=80,
        )

        for chunk in chunks:
            if not chunk.strip():
                continue

            all_chunks.append({
                "chunk_id": len(all_chunks),
                "text": chunk,
                "page": page_number,
                "document": path.name,
            })

    if not all_chunks:
        raise ValueError(
            "No text chunks could be created from the PDF."
        )

    print(f"PDF: {path.name}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Pages without readable text: {empty_pages}")
    print(f"Total chunks created: {len(all_chunks)}")

    return all_chunks


if __name__ == "__main__":
    pdf_path = input(
        "Enter the path to your PDF: "
    ).strip().strip('"')

    try:
        chunks = process_pdf(pdf_path)

        for index, chunk in enumerate(chunks[:3], start=1):
            print(f"\n--- Chunk {index} ---")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Document: {chunk['document']}")
            print(f"Page: {chunk['page']}")
            print(f"Words: {len(chunk['text'].split())}")
            print(chunk["text"][:500])

    except Exception as error:
        print(f"Document processing failed: {error}")
