import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

from preprocessor import preprocess
from extractor import extract_all

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

TESS_CONFIG = r'--oem 3 --psm 6' #TODO: configure


def init_reader(languages: list[str] = ['eng']) -> None:
    lang_str = '+'.join(languages)
    log.info(f"Tesseract OCR ready (languages: {lang_str})")
    return lang_str 


def run_ocr(lang: str, image: np.ndarray) -> tuple[list[str], list[float]]:
    """Run Tesseract with per-word confidence scores."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(
        pil_img, lang=lang, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    from collections import defaultdict
    line_words = defaultdict(list)
    line_confs = defaultdict(list)
    for i, word in enumerate(data['text']):
        word = word.strip()
        if not word:
            continue
        conf = int(data['conf'][i])
        if conf < 0: 
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        line_words[key].append(word)
        line_confs[key].append(conf / 100.0)

    lines, confs = [], []
    for key in sorted(line_words.keys()):
        line_text = ' '.join(line_words[key])
        avg_conf = sum(line_confs[key]) / len(line_confs[key])
        lines.append(line_text)
        confs.append(avg_conf)

    return lines, confs


def process_single(
    image_path: str,
    reader: str,
    output_dir: str,
    save_preprocessed: bool = False
) -> dict:
    start = time.time()
    stem = Path(image_path).stem

    result = {
        "file": os.path.basename(image_path),
        "status": "ok",
        "error": None,
        "data": None,
        "processing_time_sec": None,
    }

    try:
        img = preprocess(image_path)

        if save_preprocessed:
            pre_path = os.path.join(output_dir, f"{stem}_preprocessed.jpg")
            cv2.imwrite(pre_path, img)

        lines, confs = run_ocr(reader, img)  

        if not lines:
            result["status"] = "empty"
            result["error"] = "No text detected after preprocessing"
            return result

        extracted = extract_all(lines, confs)
        extracted["raw_text"] = lines  

        result["data"] = extracted

        out_path = os.path.join(output_dir, f"{stem}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log.info(f"✓ {stem} | store={extracted['store_name']['value']} "
                 f"| total={extracted['total_amount']['value']} "
                 f"| conf={extracted['overall_confidence']:.2f}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        log.error(f"✗ {stem}: {e}")

    result["processing_time_sec"] = round(time.time() - start, 2)
    return result


def generate_summary(all_results: list[dict]) -> dict:
    """Generate financial summary across all receipts."""
    total_spend = 0.0
    transactions = 0
    store_spend = {}
    failed = []
    low_conf = []

    for r in all_results:
        if r["status"] != "ok" or not r["data"]:
            failed.append(r["file"])
            continue

        data = r["data"]
        transactions += 1

        total_field = data.get("total_amount", {})
        raw_total = total_field.get("value", "")
        if raw_total:
            nums = re.findall(r'\d+\.\d{2}', str(raw_total))
            if nums:
                amt = float(nums[0])
                total_spend += amt

                store = data.get("store_name", {}).get("value") or "Unknown"
                store_spend[store] = round(store_spend.get(store, 0.0) + amt, 2)

        if data.get("overall_confidence", 1.0) < 0.6:
            low_conf.append(r["file"])

    return {
        "total_spend": round(total_spend, 2),
        "total_transactions": transactions,
        "failed_images": len(failed),
        "low_confidence_receipts": low_conf,
        "spend_per_store": dict(sorted(store_spend.items(), key=lambda x: -x[1])),
        "average_transaction_value": round(total_spend / transactions, 2) if transactions else 0,
    }


def run_batch(input_dir: str, output_dir: str, languages: list[str] = ['eng']):
    """Process all receipt images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    images = [
        str(p) for p in Path(input_dir).rglob('*')
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not images:
        log.error(f"No images found in {input_dir}")
        return

    log.info(f"Found {len(images)} images. Starting pipeline...")
    reader = init_reader(languages)

    all_results = []
    for i, path in enumerate(images, 1):
        log.info(f"[{i}/{len(images)}] Processing: {os.path.basename(path)}")
        result = process_single(path, reader, output_dir, True)
        all_results.append(result)

    master_path = os.path.join(output_dir, '_all_results.json')
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = generate_summary(all_results)
    summary_path = os.path.join(output_dir, '_financial_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("\n" + "="*50)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Processed : {len(all_results)}")
    log.info(f"  Successful: {sum(1 for r in all_results if r['status']=='ok')}")
    log.info(f"  Failed    : {sum(1 for r in all_results if r['status']!='ok')}")
    log.info(f"  Total Spend: {summary['total_spend']}")
    log.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt OCR Pipeline")
    parser.add_argument('--input', required=True, help='Input image directory')
    parser.add_argument('--output', default='./outputs', help='Output JSON directory')
    parser.add_argument('--lang', nargs='+', default=['eng'], help='OCR languages e.g. --lang en ms')
    parser.add_argument('--single', default=None, help='Process a single image file')
    args = parser.parse_args()

    if args.single:
        os.makedirs(args.output, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        reader = init_reader(args.lang)
        result = process_single(args.single, reader, args.output, True)
        print(json.dumps(result, indent=2))
    else:
        run_batch(args.input, args.output, args.lang)
