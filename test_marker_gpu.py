"""
Test Marker on the solutions PDF (printed + handwritten).
Run this on a machine with a GPU (RTX 4060 or similar).

Usage:
    python test_marker_gpu.py
    python test_marker_gpu.py path/to/your.pdf
    python test_marker_gpu.py path/to/your.pdf --pages 3 4 5
"""

import sys
import os
import time
import fitz
from pathlib import Path


def extract_pages(src_path, dst_path, pages):
    """Extract specific pages into a smaller PDF."""
    src = fitz.open(src_path)
    dst = fitz.open()
    for i in pages:
        dst.insert_pdf(src, from_page=i, to_page=i)
    dst.save(dst_path)
    dst.close()
    src.close()
    print(f"Extracted pages {[p+1 for p in pages]} -> {dst_path}")
    return dst_path


def run_marker(pdf_path, output_dir):
    """Run Marker and return markdown output."""
    os.makedirs(output_dir, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    # Lower batch sizes to fit in 8GB VRAM
    config_parser = ConfigParser({
        "output_format": "markdown",
    })

    config_dict = config_parser.generate_config_dict()
    # Reduce batch sizes to prevent OOM on 8GB GPUs
    for key in config_dict:
        if hasattr(config_dict[key], 'batch_size'):
            config_dict[key].batch_size = 4

    converter = PdfConverter(
        config=config_dict,
        artifact_dict=create_model_dict(),
    )

    start = time.time()
    rendered = converter(pdf_path)
    elapsed = time.time() - start

    result = rendered.markdown
    out_file = os.path.join(output_dir, "marker_output.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output saved to: {out_file}")
    print(f"Output length: {len(result):,} chars")
    return result, elapsed


def run_pymupdf(pdf_path):
    """Extract embedded text (printed content only)."""
    doc = fitz.open(pdf_path)
    print(f"\n--- pymupdf direct extraction (printed text only) ---")
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            print(f"\nPage {i+1} ({len(text)} chars):")
            print(text[:300])
            if len(text) > 300:
                print(f"  ... ({len(text)} chars total)")
    doc.close()


def main():
    # Parse arguments
    args = sys.argv[1:]
    pages = None
    pdf_path = "sample_papers/2022 June P1 (Solutions).pdf"

    if "--pages" in args:
        idx = args.index("--pages")
        pages = [int(p) - 1 for p in args[idx + 1:]]  # convert to 0-indexed
        # PDF path is before --pages if provided
        if idx > 0:
            pdf_path = args[0]
    elif args:
        pdf_path = args[0]

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    pdf_name = Path(pdf_path).stem
    output_dir = f"output/{pdf_name}"

    print(f"PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    doc.close()

    # If specific pages requested, extract them first
    test_pdf = pdf_path
    if pages:
        test_pdf = f"output/{pdf_name}_subset.pdf"
        os.makedirs("output", exist_ok=True)
        extract_pages(pdf_path, test_pdf, pages)

    print("\n" + "=" * 60)
    print("STEP 1: pymupdf direct text extraction")
    print("=" * 60)
    run_pymupdf(test_pdf)

    print("\n" + "=" * 60)
    print("STEP 2: Marker OCR (full page including handwriting)")
    print("=" * 60)
    result, elapsed = run_marker(test_pdf, output_dir)

    print("\n" + "=" * 60)
    print("MARKER OUTPUT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
