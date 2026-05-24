"""
Quick test: what can pymupdf extract directly from the PDF without any ML model?
This tells us how much of the printed text is already embedded.
"""

import fitz
import sys

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/baranka/Downloads/2022 June P1 (Solutions).pdf"

doc = fitz.open(pdf_path)
print(f"PDF: {pdf_path}")
print(f"Pages: {len(doc)}")
print("=" * 60)

for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        print(f"\n--- Page {i+1} ---")
        print(text[:500])
        if len(text) > 500:
            print(f"  ... ({len(text)} chars total)")
    else:
        print(f"\n--- Page {i+1} --- (no embedded text)")

doc.close()
