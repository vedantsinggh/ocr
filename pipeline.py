"""
pipeline.py
-----------
Hybrid receipt OCR pipeline:

  ┌──────────────┐
  │  Raw image   │
  └──────┬───────┘
         │  preprocessor.preprocess()
         ▼
  ┌──────────────┐
  │Preprocessed  │  (deskewed, denoised, upscaled)
  │   image      │
  └──────┬───────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
Tesseract    Donut
(store name, (menu items,
  date)        total)
    │          │
    └────┬─────┘
         │  postprocessor.postprocess()
         ▼
  ┌──────────────┐
  │  result.json │
  └──────────────┘

Usage:
    python pipeline.py --input ./images --output ./outputs
    python pipeline.py --input ./images --output ./outputs --single receipt.jpg
    python pipeline.py --input ./images --output ./outputs --device cuda

NOTE: Donut is loaded via HuggingFace transformers directly (VisionEncoderDecoderModel)
      instead of the donut-python library to avoid architecture/checkpoint mismatches.
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path

import torch
import cv2
import numpy as np
import pytesseract
from PIL import Image

# ── Load Donut via transformers directly (bypasses donut-python arch mismatch) ──
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
)

from preprocessor import preprocess
from postprocessor import postprocess  # use_llm kwarg added in updated version

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
MODEL_ID    = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"

# Tesseract config: treat the image as a single block of text, output UTF-8
TESS_CONFIG = r'--oem 3 --psm 4'


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (transformers VisionEncoderDecoder — no donut-python needed)
# ─────────────────────────────────────────────────────────────────────────────

def load_donut(device: str = "cpu"):
    """
    Load Donut via transformers.VisionEncoderDecoderModel.
    This avoids the Swin encoder architecture mismatch that occurs when using
    the donut-python wrapper (DonutModel.from_pretrained) with mismatched versions.
    """
    log.info(f"Loading Donut model ({MODEL_ID}) via transformers")
    log.info("First run downloads ~800 MB — subsequent runs load from HuggingFace cache.")

    processor = DonutProcessor.from_pretrained(MODEL_ID)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    log.info(f"Donut ready on {device}")
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Per-image inference
# ─────────────────────────────────────────────────────────────────────────────

def run_tesseract(preprocessed_img: np.ndarray) -> str:
    """
    Run Tesseract on the preprocessed OpenCV image.
    Converts to PIL RGB for pytesseract (it expects RGB, not BGR).
    Returns the raw OCR text string.
    """
    if len(preprocessed_img.shape) == 2:
        pil_img = Image.fromarray(preprocessed_img)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))

    text = pytesseract.image_to_string(pil_img, config=TESS_CONFIG)
    return text


def run_donut(processor: DonutProcessor,
              model: VisionEncoderDecoderModel,
              image_path: str,
              device: str = "cpu") -> dict:
    """
    End-to-end Donut inference via transformers:
      image → pixel_values → generated token ids → decoded JSON string → dict

    Returns the parsed CORD prediction dict, or {} on failure.
    """
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs exactly as the CORD fine-tune expects
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Build the decoder prompt
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    # Decode token ids → markup string → JSON dict
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")
    # Strip the task prompt prefix
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    try:
        parsed = processor.token2json(sequence)
    except Exception as e:
        log.warning(f"token2json failed ({e}), returning empty dict")
        parsed = {}

    return parsed


def process_single(image_path: str,
                   processor: DonutProcessor,
                   model: VisionEncoderDecoderModel,
                   output_dir: str,
                   device: str = "cpu",
                   use_llm: bool = True) -> dict:
    """
    Full pipeline for a single receipt image:
      1. Preprocess (deskew, denoise, upscale)
      2. Tesseract → store name + date
      3. Donut → items + total block
      4. Postprocess (validate, clean, score)
      5. Write <stem>.json to output_dir
    """
    start = time.time()
    stem = Path(image_path).stem
    result = {
        "file":                image_path,
        "status":              "ok",
        "error":               None,
        "data":                None,
        "processing_time_sec": None,
    }

    try:
        # ── Step 1: Preprocess ────────────────────────────────────────────
        log.info(f"  Preprocessing {stem}")
        preprocessed = preprocess(image_path)

        # ── Step 2: Tesseract OCR (store name + date) ─────────────────────
        log.info(f"  Running Tesseract on {stem}")
        tess_text = run_tesseract(preprocessed)
        log.debug(f"  Tesseract text:\n{tess_text[:300]}")

        # ── Step 3: Donut inference (items + totals) ──────────────────────
        log.info(f"  Running Donut on {stem}")
        donut_raw = run_donut(processor, model, image_path, device)

        if not donut_raw:
            result.update({
                "status": "empty",
                "error":  "Donut returned no predictions",
            })
            return result

        # ── Step 4: Postprocess ───────────────────────────────────────────
        structured = postprocess(tess_text, donut_raw, use_llm=use_llm)
        result["data"] = structured

        # ── Step 5: Write JSON ────────────────────────────────────────────
        out_path = os.path.join(output_dir, f"{stem}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log.info(
            f"  ✓ {stem} | store={structured['store_name']['value']} "
            f"| date={structured['date']['value']} "
            f"| items={len(structured['items'])} "
            f"| total={structured['total_amount']['value']} "
            f"| conf={structured['overall_confidence']:.2f} "
            f"| flagged={structured['flagged_fields']}"
        )

    except Exception as e:
        result.update({"status": "error", "error": str(e)})
        log.error(f"  ✗ {stem}: {e}", exc_info=True)

    result["processing_time_sec"] = round(time.time() - start, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(input_dir: str, output_dir: str, device: str = "cpu", use_llm: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(
        str(p) for p in Path(input_dir).rglob('*')
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not images:
        log.error(f"No supported images found in {input_dir}")
        return

    log.info(f"Found {len(images)} image(s) in {input_dir}")
    processor, model = load_donut(device)
    all_results = []

    for i, path in enumerate(images, 1):
        log.info(f"[{i}/{len(images)}] {os.path.basename(path)}")
        all_results.append(
            process_single(path, processor, model, output_dir, device, use_llm)
        )

    # Dump combined results
    combined_path = os.path.join(output_dir, '_all_results.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    ok    = sum(1 for r in all_results if r["status"] == "ok")
    error = sum(1 for r in all_results if r["status"] == "error")
    empty = sum(1 for r in all_results if r["status"] == "empty")
    log.info(f"DONE | ok={ok} | error={error} | empty={empty} | total={len(all_results)}")
    log.info(f"Results written to {combined_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Tesseract + Donut receipt OCR pipeline"
    )
    parser.add_argument('--input',  required=True,
                        help="Input directory containing receipt images")
    parser.add_argument('--output', default='./outputs',
                        help="Output directory for JSON results (default: ./outputs)")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help="Device for Donut model (default: cpu)")
    parser.add_argument('--single', default=None,
                        help="Process a single image file instead of a directory")
    parser.add_argument('--no-llm', action='store_true',
                        help="Disable the LLM correction pass (faster, offline-safe)")
    args = parser.parse_args()

    use_llm = not args.no_llm

    if args.single:
        os.makedirs(args.output, exist_ok=True)
        processor, m = load_donut(args.device)
        result = process_single(args.single, processor, m, args.output, args.device, use_llm)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        run_batch(args.input, args.output, args.device, use_llm)
