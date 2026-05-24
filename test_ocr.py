"""
OCR Test Script — Compare Marker vs Nougat on A-level past papers.
Converts PDF pages to images, runs both models, saves output for comparison.
"""

import sys
import os
import time

# Force CPU to avoid MPS OOM
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_DEVICE"] = "cpu"

import fitz  # pymupdf
from pathlib import Path


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300, max_pages: int = 0) -> list[str]:
    """Convert PDF pages to PNG images."""
    doc = fitz.open(pdf_path)
    image_paths = []
    os.makedirs(output_dir, exist_ok=True)

    total = min(len(doc), max_pages) if max_pages > 0 else len(doc)
    for i in range(total):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        pix.save(img_path)
        image_paths.append(img_path)
        print(f"  Extracted page {i+1}/{total}")

    doc.close()
    return image_paths


def test_marker(pdf_path: str, output_dir: str) -> tuple[str, float]:
    """Run Marker on a PDF and return (output_text, time_taken)."""
    print("\n--- Testing Marker ---")
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    try:
        import torch
        # Force CPU
        torch.set_default_device("cpu")

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
        )
        rendered = converter(pdf_path)
        result_text = rendered.markdown

        elapsed = time.time() - start

        out_file = os.path.join(output_dir, "marker_output.md")
        with open(out_file, "w") as f:
            f.write(result_text)

        print(f"  Done in {elapsed:.1f}s")
        print(f"  Output saved to {out_file}")
        print(f"  Output length: {len(result_text)} chars")
        return result_text, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}", elapsed


def test_nougat(pdf_path: str, output_dir: str) -> tuple[str, float]:
    """Run Nougat on a PDF and return (output_text, time_taken)."""
    print("\n--- Testing Nougat ---")
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    try:
        # Monkey-patch albumentations to fix the broken nougat dependency
        import albumentations as alb
        original_ic = alb.ImageCompression
        def patched_ic(*args, **kwargs):
            if args and isinstance(args[0], int):
                kwargs['quality_range'] = (args[0], 100)
                args = args[1:]
            return original_ic(*args, **kwargs)
        alb.ImageCompression = patched_ic

        from nougat import NougatModel
        from nougat.utils.dataset import LazyDataset
        from nougat.utils.checkpoint import get_checkpoint
        from nougat.postprocessing import markdown_compatible
        import torch
        from torch.utils.data import DataLoader

        checkpoint = get_checkpoint(None, model_tag="0.1.0-small")
        model = NougatModel.from_pretrained(checkpoint)

        device = "cpu"
        print(f"  Using device: {device}")
        model.to(device)
        model.eval()

        dataset = LazyDataset(pdf_path, None, None)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_text = []
        for i, (sample, is_last) in enumerate(dataloader):
            sample = sample.to(device)
            with torch.no_grad():
                output = model.inference(image_tensors=sample)
                for o in output["predictions"]:
                    page_text = markdown_compatible(o)
                    all_text.append(page_text)
                    print(f"  Processed page {i+1}")

        result_text = "\n\n---\n\n".join(all_text)
        elapsed = time.time() - start

        out_file = os.path.join(output_dir, "nougat_output.md")
        with open(out_file, "w") as f:
            f.write(result_text)

        print(f"  Done in {elapsed:.1f}s")
        print(f"  Output saved to {out_file}")
        print(f"  Output length: {len(result_text)} chars")
        return result_text, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}", elapsed


def main():
    if len(sys.argv) < 2:
        pdf_path = "/Users/baranka/Downloads/2021 June P1.pdf"
    else:
        pdf_path = sys.argv[1]

    # Optional: limit pages for faster testing
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    pdf_name = Path(pdf_path).stem
    base_output = f"/Users/baranka/Projects/ocr-test/output/{pdf_name}"

    print(f"Testing OCR on: {pdf_path}")
    if max_pages:
        print(f"Limiting to first {max_pages} pages")
    print("=" * 60)

    # Step 1: Extract pages as images
    print("\n[1/3] Extracting PDF pages as images...")
    images = pdf_to_images(pdf_path, f"{base_output}/pages", max_pages=max_pages)
    print(f"  Total pages extracted: {len(images)}")

    # Step 2: Run Marker
    print("\n[2/3] Running Marker (CPU mode — will be slower but won't OOM)...")
    marker_text, marker_time = test_marker(pdf_path, f"{base_output}/marker")

    # Step 3: Run Nougat
    print("\n[3/3] Running Nougat (CPU mode)...")
    nougat_text, nougat_time = test_nougat(pdf_path, f"{base_output}/nougat")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Pages: {len(images)}")
    print()
    print(f"Marker:  {marker_time:6.1f}s  |  {len(marker_text):,} chars")
    print(f"Nougat:  {nougat_time:6.1f}s  |  {len(nougat_text):,} chars")
    print()
    print(f"Outputs saved to: {base_output}/")
    print(f"  - marker/marker_output.md")
    print(f"  - nougat/nougat_output.md")
    print(f"  - pages/*.png (individual page images)")
    print()
    print("Open the .md files to compare quality!")


if __name__ == "__main__":
    main()
