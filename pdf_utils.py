"""PDF text extraction utilities."""

import pymupdf


def extract_pdf_text(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from the first N pages of a PDF using pymupdf."""
    doc = pymupdf.open(pdf_path)
    try:
        pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text = page.get_text().strip()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
    finally:
        doc.close()

    if not pages:
        raise ValueError(f"No text extracted from {pdf_path}")

    return "\n\n".join(pages)
