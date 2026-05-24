"""
Test local OCR libraries on handwritten math from A-level solutions.
No API calls — everything runs locally.

Tests:
1. EasyOCR — general purpose OCR, supports handwriting
2. Surya OCR — document-focused OCR (installed with Marker)
"""

import time
import os

# Page 3 has handwritten solutions with fractions and square roots
IMAGE_PATH = "/Users/baranka/Projects/ocr-test/output/solutions_preview/page_3.png"
OUTPUT_DIR = "/Users/baranka/Projects/ocr-test/output/handwriting_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_easyocr():
    print("\n" + "=" * 60)
    print("TEST 1: EasyOCR")
    print("=" * 60)
    start = time.time()
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(IMAGE_PATH, detail=1)

        elapsed = time.time() - start
        print(f"Time: {elapsed:.1f}s")
        print(f"Detected {len(results)} text regions\n")

        full_text = []
        for bbox, text, conf in results:
            label = "HIGH" if conf > 0.7 else "MED " if conf > 0.4 else "LOW "
            print(f"  [{label} {conf:.2f}] {text}")
            full_text.append(text)

        out = "\n".join(full_text)
        with open(f"{OUTPUT_DIR}/easyocr_output.txt", "w") as f:
            f.write(out)
        return out, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return str(e), elapsed


def test_surya():
    print("\n" + "=" * 60)
    print("TEST 2: Surya OCR")
    print("=" * 60)
    start = time.time()
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from PIL import Image

        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()

        image = Image.open(IMAGE_PATH)
        predictions = rec_predictor([image], [["en"]], det_predictor)

        elapsed = time.time() - start
        print(f"Time: {elapsed:.1f}s")

        full_text = []
        for page in predictions:
            for line in page.text_lines:
                conf = line.confidence
                label = "HIGH" if conf > 0.7 else "MED " if conf > 0.4 else "LOW "
                print(f"  [{label} {conf:.2f}] {line.text}")
                full_text.append(line.text)

        out = "\n".join(full_text)
        with open(f"{OUTPUT_DIR}/surya_output.txt", "w") as f:
            f.write(out)
        return out, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return str(e), elapsed


def main():
    print(f"Testing OCR on: {IMAGE_PATH}")
    print("This page has printed questions + handwritten solutions in blue ink")

    easy_text, easy_time = test_easyocr()
    surya_text, surya_time = test_surya()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"EasyOCR:  {easy_time:6.1f}s  |  {len(easy_text):,} chars")
    print(f"Surya:    {surya_time:6.1f}s  |  {len(surya_text):,} chars")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
