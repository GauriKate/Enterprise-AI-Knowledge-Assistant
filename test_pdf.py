
import pymupdf

pdf_path = input("Enter PDF path: ").strip().strip('"')

try:
    document = pymupdf.open(pdf_path)
    print("Number of pages:", len(document))

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()
        print(f"\nPage {page_number}: {len(text)} characters extracted")
        print(text[:1000])

    document.close()

except Exception as error:
    print("PDF extraction failed:", error)
