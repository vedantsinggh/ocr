# Receipt OCR Pipeline

A two-stage pipeline that turns receipt images into structured JSON. The first stage extracts text from images using PaddleOCR. The second stage sends that text to a local LLM running in Ollama and gets back clean, structured data — store name, date, line items, totals, tax, and payment method. Results are written to disk as they arrive, so you can check the first batch before the rest of the dataset finishes.

---

## How it works

### Stage 1 — Preprocessing and OCR

Each image goes through a preprocessor (`preprocessor.py`, not included here but expected alongside this pipeline) that applies CLAHE contrast enhancement to improve legibility. The enhanced image is saved to `preprocessed/` as a PNG, then handed to PaddleOCR.

PaddleOCR runs with angle classification enabled, which helps with rotated or skewed receipts. It returns bounding boxes and text strings. The pipeline takes only the text strings, filters out blank lines, and writes them to `ocr_raw/<stem>.txt` — one line per detected text region, in reading order.

The OCR stage runs in subprocesses, not threads. PaddleOCR's underlying C++ engine (PaddlePaddle) is not thread-safe — sharing its state across threads causes segfaults. Each subprocess gets its own isolated PaddlePaddle runtime. PaddleOCR is also imported inside the worker function rather than at the top of the file; this prevents the runtime from being initialised in the parent process before forking, which is another known crash source.

### Stage 2 — LLM structuring

The raw OCR lines are sent to a local Ollama instance. Rather than one request per receipt, receipts are batched — up to `BATCH_SIZE` (default 8) receipts go into a single prompt, and the LLM returns a JSON object keyed by index. This reduces the number of Ollama round-trips from one per image to roughly one per eight images, which matters on large datasets.

The LLM is instructed to fix common OCR errors (broken words, garbled spacing), normalise dates to YYYY-MM-DD, identify the store name from the header rather than the branch address, combine tax lines, and use null for any field it cannot find rather than guessing. Temperature is set to 0.05 to keep outputs deterministic and reduce hallucination.

LLM responses frequently contain minor JSON errors — trailing commas, single-quoted strings, Python-style `None` instead of `null`. The parser makes four attempts before failing: it tries the raw output, then strips markdown fences, then extracts the outermost `{ }` block, then runs a repair pass that fixes the common syntax mistakes with targeted regex substitutions. The repair pass is conservative — it only removes or replaces characters, never inserts them, so it cannot introduce wrong values. If a whole batch fails after repair, each receipt in that batch is retried individually so one bad response does not discard the rest.

Each receipt JSON is written to disk as soon as the LLM returns it, not after the full dataset completes. On a run of 371 images with batch size 8 and 2 LLM workers, the first results appear within the first minute.

### Output

```
<output>/
  preprocessed/       enhanced PNG for each input image
  ocr_raw/            raw OCR lines as plain text, one file per image
  <stem>.json         structured receipt data for each image
  summary.json        aggregate stats and all records in one file
```

Each `<stem>.json` looks like this:

```json
{
  "_meta": {
    "source": "images/42.jpg",
    "preprocessed_path": "output/preprocessed/42.png",
    "ocr_raw_path": "output/ocr_raw/42.txt",
    "ocr_line_count": 34,
    "elapsed_ocr_s": 2.1
  },
  "store": "DMart",
  "date": "2024-03-15",
  "items": [
    { "name": "Tata Salt 1kg", "qty": 2, "price": "28.00" },
    { "name": "Amul Butter 500g", "qty": 1, "price": "275.00" }
  ],
  "subtotal": "331.00",
  "tax": "16.55",
  "total": "347.55",
  "currency": "INR",
  "payment_method": "UPI"
}
```

If processing fails at either stage, the file still gets written with an `"error"` key so the summary can account for it.

---

## Requirements

```
paddleocr
paddlepaddle   (or paddlepaddle-gpu if you have a CUDA-capable card)
opencv-python
Pillow
requests
ollama         (the server binary, running separately)
```

You also need a model pulled in Ollama. The default is `llama3:8b`:

```bash
ollama pull llama3:8b
ollama serve
```

---

## Usage

**Process a folder of images:**

```bash
python pipeline.py ./receipts/ --output ./output
```

**Process a single image:**

```bash
python pipeline.py receipt.jpg --output ./output
```

**Re-run only the LLM step without redoing OCR:**

If you want to tweak the prompt, swap the model, or fix LLM failures without waiting 13 minutes for OCR again, point `--from-ocr` at the `ocr_raw/` folder from a previous run:

```bash
python pipeline.py --from-ocr ./output/ocr_raw/ --output ./output_v2
```

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--output DIR` | `./output` | Where to write results. Created if it does not exist. |
| `--workers N` | `2` | OCR subprocess count. Match this to your physical core count. Going above 4 rarely helps and increases memory pressure. |
| `--llm-workers N` | `2` | Parallel threads sending prompts to Ollama. Local Ollama serialises GPU inference anyway, so above 3 you are mostly queuing. |
| `--batch-size N` | `8` | Receipts per LLM prompt. Lower this to 4 if you see truncated or missing items in the output. Raise it to 12 if everything looks clean and you want more speed. |
| `--no-enhance` | off | Skip CLAHE preprocessing. Try this if enhanced images look worse than the originals for your particular receipts. |
| `--from-ocr DIR` | — | Load OCR text from a previous run's `ocr_raw/` folder and skip straight to the LLM stage. |

---

## Tuning for large batches

With 300+ images, OCR is the slow part and the LLM is the tunable part.

The OCR stage is CPU and memory bound. Two workers is a safe default; four is reasonable if you have the cores and RAM. More than that and you will likely see diminishing returns or memory pressure.

The LLM stage with `--batch-size 8` and `--llm-workers 2` results in about 47 Ollama calls for 371 images instead of 371. Each call takes longer than a single-receipt call, but the total wall time is substantially lower. The bottleneck is Ollama's inference speed, not HTTP overhead, so adding more LLM workers beyond 2 or 3 mostly just fills a queue.

If you have a GPU, make sure you installed `paddlepaddle-gpu` and that Ollama is using it. Both stages benefit significantly from GPU acceleration.

To swap the Ollama model, edit `MODEL` at the top of `postprocessor.py`. `llama3.2:3b` is roughly twice as fast as `llama3:8b` with acceptable accuracy on clean receipts. If your receipts have heavy OCR noise, the 8b model holds up better.

---

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Entry point. Orchestrates preprocessing, OCR, LLM, and output writing. |
| `postprocessor.py` | Handles all Ollama communication, batching, JSON repair, and validation. |
| `preprocessor.py` | Image preprocessing (CLAHE, deskew, etc.). Expected alongside these files. |
