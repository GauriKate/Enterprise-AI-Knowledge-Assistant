
from pdf_reader import extract_pages
from process_document import process_pdf

pdf_path = "documents/IEEE_Conference_blood_group_detection_using_fingerprint (2).pdf"


# Step 1: Check PDF extraction
print("=" * 60)
print("CHECKING PDF EXTRACTION")
print("=" * 60)

pages = extract_pages(pdf_path)

if not pages:
    print("No pages or readable text found in the PDF.")
else:
    print(f"Total pages processed: {len(pages)}")

    print("\nPage extraction summary:")

    for page in pages:
        text = page["text"]

        print(
            f"Page {page['page']}: "
            f"{len(text.split())} words, "
            f"{len(text)} characters"
        )

        # Warn when a page may contain image-based text
        if len(text.strip()) < 150:
            print(
                f"  WARNING: Page {page['page']} has very little "
                "extracted text. Check whether OCR is needed."
            )


# Step 2: Check the extracted text on pages 4 and 8
print("\n" + "=" * 60)
print("CHECKING POSSIBLY IMAGE-BASED PAGES")
print("=" * 60)

for page in pages:
    if page["page"] in (4, 8):
        print(f"\nPage {page['page']} extracted text:")
        print("-" * 40)
        print(page["text"])
        print("-" * 40)


# Step 3: Check document chunking
print("\n" + "=" * 60)
print("CHECKING CHUNKING")
print("=" * 60)

try:
    chunks = process_pdf(pdf_path)

    print("Total chunks:", len(chunks))

    if not chunks:
        print("WARNING: No chunks were created.")
    else:
        for number, chunk in enumerate(chunks, start=1):
            print(
                f"Chunk {number}: "
                f"Page {chunk['page']}, "
                f"{len(chunk['text'].split())} words, "
                f"{len(chunk['text'])} characters"
            )

            # Display a short preview of each chunk
            preview = chunk["text"][:200].replace("\n", " ")
            print("  Preview:", preview)

except Exception as error:
    print("Error while processing the PDF:", error)


# Step 4: Final summary
print("\n" + "=" * 60)
print("CHECK COMPLETED")
print("=" * 60)

print("Pages extracted:", len(pages))
print("Chunks created:", len(chunks) if "chunks" in locals() else 0)

print("\nNote:")
print("- Few words on a page may indicate image-based content.")
print("- This test identifies possible OCR issues; it does not fix them.")
print("- OCR improvements must be made in pdf_reader.py.")
