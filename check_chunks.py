
from process_document import process_pdf

pdf_path = (
    r"documents\IEEE_Conference_blood_group_detection_using_fingerprint (2).pdf"
)

chunks = process_pdf(pdf_path)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    text = chunk["text"]

    print(f"\nChunk: {i}")
    print("Chunk ID:", chunk.get("chunk_id", "N/A"))
    print("Page:", chunk["page"])
    print("Word count:", len(text.split()))
    print("First 150 characters:", text[:150].replace("\n", " "))
    print(
        "Contains research objective:",
        "objective" in text.lower()
    )
    print(
        "Contains limitations:",
        "limitations" in text.lower()
    )
    print(
        "Contains AlexNet:",
        "alexnet" in text.lower()
    )
    print("-" * 50)
