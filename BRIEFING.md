# OCR Model — Briefing & Task Assignment

## Context

We're building a system that reads A-level exam papers containing both printed questions and handwritten student solutions, and converts everything to structured text (Markdown + LaTeX). The goal is to do this locally, without relying on paid AI API calls, so we can process thousands of papers at scale without massive costs.

## What We Tested

We tested 5 different local OCR tools on a real past paper with handwritten solutions (Pearson Edexcel IGCSE Further Pure Maths, 2022 June P1 with handwritten solutions in blue ink).

| Tool | What It Does | Printed Text | Handwritten Math | Verdict |
|------|-------------|-------------|-----------------|---------|
| **pymupdf** | Extracts embedded text from PDFs | Perfect, instant | Cannot read (not embedded) | Use for printed text only |
| **EasyOCR** | General-purpose OCR library | Decent | Complete failure — misreads symbols, no math structure | Not viable |
| **Pix2Tex** | Math image to LaTeX converter | Not designed for full pages | Complete failure on handwriting | Not viable |
| **Nougat** (Meta) | Academic document OCR | Untested (dependency issues) | Untested | Broken dependencies |
| **Marker** | Multi-model pipeline: layout detection + OCR + equation processing | Excellent — accurate LaTeX output | Partially works (see below) | Best candidate |

## What Marker Got Right (Handwritten)

Marker successfully read handwritten algebra from the student solutions:

- `H = (8x + y)(6x + y) = 48x² + 14xy + y²` — read correctly
- `K = H - 2·9x² - 2·15x²` — read correctly
- `(y + 108)(y - 10) = 0` — read correctly
- `y = -108, y = 10` — read correctly

These were output as clean LaTeX, ready to use.

## What Marker Got Wrong (Handwritten)

Marker struggled with more complex handwritten notation:

- Fractions written by hand (numerator over denominator with a line)
- Square root symbols (√)
- Example: `(38 - 22√3) / 2 = 19 - 11√3` was read as `8-2253 = 19-1153`

## Key Findings

1. **Printed text doesn't need a model at all** — pymupdf extracts it instantly from the PDF for free.
2. **No off-the-shelf local tool handles handwritten math well** — this is an unsolved problem in open-source.
3. **Marker is the best starting point** — its pipeline architecture works, but the handwriting recognition component (Surya text recognition model) needs fine-tuning.
4. **The sample we tested had clean handwriting** — real student work will be messier, so we need the model to be even more robust than what we've seen.
5. **Hardware requirement**: GPU with 8GB+ VRAM. Tested on RTX 4060 Laptop GPU (8.6GB VRAM), 3 pages processed in ~16 minutes. CPU is not viable (2+ hours for 20 pages on Apple M1).

## Current Architecture

```
PDF Input
    │
    ├── pymupdf ──────────── Printed text extracted instantly (free, no model)
    │
    └── Marker Pipeline ──── Handwritten content
            │
            ├── Layout Detection (Surya) ──── Identifies text/equation/table/diagram regions
            ├── Text Recognition (Surya) ──── Converts detected regions to text ← THIS NEEDS FINE-TUNING
            ├── Equation Processor ────────── Converts math regions to LaTeX
            └── Table Recognition ─────────── Handles table structures
```

## What Needs to Happen Next

The Surya text recognition model inside Marker needs to be fine-tuned on handwritten math data so it can reliably read:
- Handwritten fractions
- Square root symbols
- Exponents and subscripts
- Messy/rushed student handwriting
- Mixed printed + handwritten pages

---

## Task for Team Member

**Assignment:** Write a clear technical explanation for a potential investor covering:

1. **What exactly are we fine-tuning?** — Which model within the Marker pipeline, what architecture is it (transformer-based encoder-decoder), what was it originally trained on, and why does it fail on handwriting.

2. **How will we get training data?** — Explain the data pipeline: collecting handwritten samples from real student papers, labeling them (using a vision API for initial labeling as a one-time cost), augmenting with public datasets (CROHME, HME100K), and synthetic data generation.

3. **How does fine-tuning work technically?** — Explain transfer learning: we take the pre-trained Surya model (already good at printed text), freeze/unfreeze layers, and train it further on our handwritten math dataset so it learns to recognize handwritten notation while keeping its existing capabilities.

4. **What makes our approach better than alternatives?** — Why fine-tuning an existing pipeline (Marker + Surya) is smarter than training from scratch. Why local inference beats API costs at scale. Why the hybrid approach (pymupdf for printed + fine-tuned model for handwritten only) is more efficient.

5. **What hardware and resources are needed?** — Training hardware (GPU requirements for fine-tuning), inference hardware (what we need to deploy), dataset size needed, and estimated timeline.

6. **What does success look like?** — Target accuracy metrics, how we measure them, and what the end product delivers (structured text output from any student paper).

Make it investor-friendly: technically accurate but understandable to someone who isn't an ML engineer. Include the test results from this briefing as evidence that the base approach works.
