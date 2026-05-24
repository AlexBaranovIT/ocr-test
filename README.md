# OCR Test — Handwritten Math Recognition

Testing local OCR models for recognizing handwritten equations from A-level exam past papers.

## Goal
Build a model/pipeline that can read PDFs containing both **printed exam questions** and **handwritten student solutions**, without relying on paid API calls.

## What We've Tested So Far (on Mac M1 8GB)

| Tool | Printed Text | Handwritten Math | Speed | Verdict |
|------|-------------|-----------------|-------|---------|
| **pymupdf** | Perfect (instant) | Can't read | <1s | Use for printed text extraction |
| **EasyOCR** | Decent | Garbage | 39s | No good for math |
| **Pix2Tex** | Not for full pages | Garbage on handwriting | 3-18s | No good for handwriting |
| **Marker** | Great quality | Untested (too slow on CPU) | 2.3hrs/20pg on CPU | Needs GPU |
| **Nougat** | Crashed (dependency issues) | Untested | N/A | Broken deps |

## Next Step: Test on GPU (RTX 4060)

Marker produced excellent results on printed math (LaTeX output was accurate). Need to test it on the handwritten solutions with a proper GPU.

## Setup (Windows + RTX 4060)

### 1. Install Python 3.11+
Download from python.org if not installed.

### 2. Create environment
```bash
mkdir ocr-test
cd ocr-test
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install dependencies
```bash
pip install marker-pdf pymupdf easyocr pix2tex Pillow
```

### 5. Run the Marker test on handwritten solutions
```bash
python test_marker_gpu.py
```

## Sample Papers
- `sample_papers/2022 June P1 (Solutions).pdf` — Pearson Edexcel IGCSE Further Pure Maths with handwritten solutions in blue ink

## Key Files
- `test_marker_gpu.py` — Main test: runs Marker on the solutions PDF (use this on GPU machine)
- `test_pymupdf.py` — Quick test: extract embedded text directly (no model needed)
- `test_handwriting.py` — EasyOCR + Surya comparison test
- `test_pix2tex.py` — Pix2Tex (LaTeX-OCR) test
- `test_ocr.py` — Original full comparison script (Marker vs Nougat)

## Findings So Far
1. **pymupdf extracts all printed text instantly** — no model needed for exam questions
2. **No off-the-shelf local OCR handles handwritten math** — EasyOCR and Pix2Tex both failed
3. **Marker is the best candidate** but needs GPU (8GB+ VRAM) to run at usable speed
4. If Marker also fails on handwriting, next option is fine-tuning TrOCR on labeled data
