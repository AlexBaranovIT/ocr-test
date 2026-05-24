"""
Test pix2tex (LaTeX-OCR) on handwritten math equations.
Pix2tex is specifically designed to convert math images → LaTeX.
"""

import time
import os
from PIL import Image

OUTPUT_DIR = "/Users/baranka/Projects/ocr-test/output/pix2tex_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test on the full page first
PAGE_3 = "/Users/baranka/Projects/ocr-test/output/solutions_preview/page_3.png"
PAGE_4 = "/Users/baranka/Projects/ocr-test/output/solutions_preview/page_4.png"
PAGE_5 = "/Users/baranka/Projects/ocr-test/output/solutions_preview/page_5.png"


def crop_handwritten_regions():
    """Crop just the handwritten parts from the pages."""
    crops = []

    # Page 3: handwritten solution is roughly in the middle area
    img = Image.open(PAGE_3)
    w, h = img.size
    # The handwritten math is between ~35-55% height
    crop1 = img.crop((0, int(h * 0.33), w, int(h * 0.55)))
    path1 = f"{OUTPUT_DIR}/crop_p3_solution.png"
    crop1.save(path1)
    crops.append(("Page 3 - fraction/sqrt solution", path1))

    # Page 4: handwritten solution at the bottom
    img4 = Image.open(PAGE_4)
    w4, h4 = img4.size
    crop2 = img4.crop((0, int(h4 * 0.70), w4, int(h4 * 0.95)))
    path2 = f"{OUTPUT_DIR}/crop_p4_solution.png"
    crop2.save(path2)
    crops.append(("Page 4 - algebra solution", path2))

    # Page 5: handwritten solution at the top
    img5 = Image.open(PAGE_5)
    w5, h5 = img5.size
    crop3 = img5.crop((0, int(h5 * 0.02), w5, int(h5 * 0.55)))
    path3 = f"{OUTPUT_DIR}/crop_p5_solution.png"
    crop3.save(path3)
    crops.append(("Page 5 - continued algebra", path3))

    return crops


def test_pix2tex_on_image(model, image_path, label):
    """Run pix2tex on a single image."""
    print(f"\n  [{label}]")
    start = time.time()
    try:
        img = Image.open(image_path)
        result = model(img)
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s")
        print(f"  LaTeX: {result}")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        return str(e), elapsed


def main():
    print("Loading pix2tex model...")
    start = time.time()
    from pix2tex.cli import LatexOCR
    model = LatexOCR()
    print(f"Model loaded in {time.time() - start:.1f}s\n")

    # First test on full pages
    print("=" * 60)
    print("TEST 1: Full pages (mixed printed + handwritten)")
    print("=" * 60)
    for name, path in [("Page 3", PAGE_3), ("Page 4", PAGE_4), ("Page 5", PAGE_5)]:
        test_pix2tex_on_image(model, path, name)

    # Then test on cropped handwritten regions
    print("\n" + "=" * 60)
    print("TEST 2: Cropped handwritten regions only")
    print("=" * 60)
    crops = crop_handwritten_regions()
    results = []
    for label, path in crops:
        result, elapsed = test_pix2tex_on_image(model, path, label)
        results.append((label, result, elapsed))

    # Save results
    with open(f"{OUTPUT_DIR}/results.txt", "w") as f:
        for label, result, elapsed in results:
            f.write(f"[{label}] ({elapsed:.1f}s)\n{result}\n\n")

    print(f"\nResults saved to {OUTPUT_DIR}/results.txt")


if __name__ == "__main__":
    main()
