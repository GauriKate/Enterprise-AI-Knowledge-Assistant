
from pathlib import Path

import pymupdf
from pypdf import PdfReader


MIN_TEXT_LENGTH = 150
OCR_SCALE = 2.5


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from each PDF page.

    PyMuPDF is the primary extractor.
    pypdf is a fallback for pages with insufficient text.
    EasyOCR is used only when text extraction remains insufficient.

    Every page is retained with its original 1-based page number.
    """

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    # Step 1: Extract text using PyMuPDF.
    try:
        with pymupdf.open(str(path)) as document:
            if document.needs_pass:
                raise ValueError(
                    "This PDF is password-protected. "
                    "Please provide an accessible PDF."
                )

            pages = [
                {
                    "page": page_number,
                    "text": page.get_text("text").strip(),
                }
                for page_number, page in enumerate(
                    document, start=1
                )
            ]

    except ValueError:
        raise
    except Exception as error:
        print(f"PyMuPDF extraction failed: {error}")
        pages = []

    # Step 2: If primary extraction failed, try pypdf.
    if not pages:
        pages = _extract_with_pypdf(path)

    if not pages:
        raise ValueError(
            "Could not read pages from this PDF."
        )

    # Step 3: Retry weak pages with pypdf, then OCR if needed.
    weak_pages = [
        item["page"]
        for item in pages
        if len(item["text"].strip()) < MIN_TEXT_LENGTH
    ]

    if weak_pages:
        fallback_pages = _extract_with_pypdf(
            path,
            target_pages=set(weak_pages),
        )

        fallback_by_page = {
            item["page"]: item["text"]
            for item in fallback_pages
        }

        for item in pages:
            if item["page"] not in weak_pages:
                continue

            fallback_text = fallback_by_page.get(
                item["page"], ""
            ).strip()

            if len(fallback_text) > len(item["text"].strip()):
                item["text"] = fallback_text

    # Step 4: OCR only pages that still have insufficient text.
    weak_pages = [
        item["page"]
        for item in pages
        if len(item["text"].strip()) < MIN_TEXT_LENGTH
    ]

    if weak_pages:
        _ocr_pages(path, pages, set(weak_pages))

    # Keep all pages, even when no text could be extracted.
    return pages


def _extract_with_pypdf(
    path: Path,
    target_pages: set[int] | None = None,
) -> list[dict]:
    """Extract text using pypdf, optionally for selected pages."""

    try:
        reader = PdfReader(str(path), strict=False)

        if reader.is_encrypted:
            try:
                result = reader.decrypt("")
                if not result:
                    print(
                        "pypdf could not open the encrypted PDF."
                    )
                    return []
            except Exception:
                print("Could not decrypt the PDF using pypdf.")
                return []

        pages = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):
            if (
                target_pages is not None
                and page_number not in target_pages
            ):
                continue

            try:
                text = (page.extract_text() or "").strip()
            except Exception as error:
                print(
                    f"pypdf failed on page {page_number}: {error}"
                )
                text = ""

            pages.append({
                "page": page_number,
                "text": text,
            })

        return pages

    except Exception as error:
        print(f"pypdf extraction failed: {error}")
        return []


def _ocr_pages(
    path: Path,
    pages: list[dict],
    page_numbers: set[int],
) -> None:
    """Apply OCR to selected pages and update their text in place."""

    try:
        import easyocr
    except ImportError:
        print(
            "OCR unavailable. Install it using "
            "'python -m pip install easyocr'."
        )
        return

    try:
        reader = easyocr.Reader(["en"], gpu=False)
    except Exception as error:
        print(f"Could not initialize EasyOCR: {error}")
        return

    page_lookup = {
        item["page"]: item for item in pages
    }

    try:
        with pymupdf.open(str(path)) as document:
            for page_number in sorted(page_numbers):
                if page_number < 1 or page_number > len(document):
                    continue

                item = page_lookup[page_number]

                try:
                    pdf_page = document[page_number - 1]

                    pixmap = pdf_page.get_pixmap(
                        matrix=pymupdf.Matrix(
                            OCR_SCALE, OCR_SCALE
                        ),
                        alpha=False,
                    )

                    image_bytes = pixmap.tobytes("png")

                    results = reader.readtext(
                        image_bytes,
                        detail=0,
                        paragraph=True,
                    )

                    ocr_text = "\n".join(
                        str(result).strip()
                        for result in results
                        if str(result).strip()
                    ).strip()

                    original_text = item["text"].strip()

                    # Prefer the text extraction result that contains
                    # more content rather than duplicating OCR text.
                    if len(ocr_text) > len(original_text):
                        item["text"] = ocr_text

                    if not item["text"].strip():
                        print(
                            f"No readable text on page {page_number}."
                        )

                except Exception as error:
                    print(
                        f"OCR failed on page {page_number}: {error}"
                    )

    except Exception as error:
        print(f"Could not open PDF for OCR: {error}")


def extract_text(pdf_path: str) -> str:
    """Extract PDF text as one string with page labels."""

    pages = extract_pages(pdf_path)

    return "\n\n".join(
        f"Page {page['page']}:\n{page['text']}"
        for page in pages
        if page["text"].strip()
    )


if __name__ == "__main__":
    pdf_file = input(
        "Enter the PDF file path: "
    ).strip().strip('"')

    try:
        pages = extract_pages(pdf_file)

        total_characters = sum(
            len(page["text"]) for page in pages
        )

        readable_pages = sum(
            bool(page["text"].strip()) for page in pages
        )

        print("\nPDF extraction completed.")
        print(f"Total PDF pages: {len(pages)}")
        print(f"Pages with extracted text: {readable_pages}")
        print(f"Pages without extracted text: {len(pages) - readable_pages}")
        print(f"Total extracted characters: {total_characters}")

        for page in pages[:3]:
            print(f"\n--- Page {page['page']} ---")
            print(page["text"][:1000] or "[No text extracted]")

    except Exception as error:
        print(f"Could not read the PDF: {error}")
