import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def _ocr_worker(args):
    image_path_str, output_dir_str, enhance = args

    import cv2
    from PIL import Image
    from paddleocr import PaddleOCR
    from preprocessor import preprocess

    image_path = Path(image_path_str)
    output_dir = Path(output_dir_str)

    pre_dir = output_dir / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)
    pre_path = pre_dir / f"{image_path.stem}.png"

    ocr_raw_dir = output_dir / "ocr_raw"
    ocr_raw_dir.mkdir(parents=True, exist_ok=True)
    ocr_raw_path = ocr_raw_dir / f"{image_path.stem}.txt"

    conf_dir = output_dir / "confidence"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / f"{image_path.stem}.json"

    t0 = time.perf_counter()
    try:
        img_array = preprocess(str(image_path), enhance=enhance)
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(pre_path))

        ocr    = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(str(pre_path), cls=True)

        lines = []
        confidence_records = []
        if result and result[0]:
            for ln in result[0]:
                text, score = ln[1][0], ln[1][1]
                if text.strip():
                    lines.append(text)
                    confidence_records.append({
                        "text":       text,
                        "confidence": round(float(score), 4),
                    })

        # Persist raw OCR lines so they can be reused with --from-ocr
        ocr_raw_path.write_text("\n".join(lines), encoding="utf-8")

        # Persist per-line confidence scores to confidence/<stem>.json
        conf_payload = {
            "source":    str(image_path),
            "line_count": len(confidence_records),
            "avg_confidence": round(
                sum(r["confidence"] for r in confidence_records) / len(confidence_records), 4
            ) if confidence_records else None,
            "lines": confidence_records,
        }
        conf_path.write_text(json.dumps(conf_payload, indent=2, ensure_ascii=False))

        elapsed = round(time.perf_counter() - t0, 2)
        avg_conf = conf_payload["avg_confidence"]
        print(
            f"  [OCR] {image_path.name:<40}  {len(lines):>3} lines"
            f"  avg_conf={avg_conf if avg_conf is not None else 'N/A'}  {elapsed}s",
            flush=True,
        )
        return {
            "source":            str(image_path),
            "preprocessed_path": str(pre_path),
            "ocr_raw_path":      str(ocr_raw_path),
            "confidence_path":   str(conf_path),
            "ocr_lines":         lines,
            "elapsed_ocr_s":     elapsed,
        }

    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"  [OCR] FAILED {image_path.name}: {exc}", flush=True)
        return {
            "source":            str(image_path),
            "preprocessed_path": None,
            "ocr_raw_path":      None,
            "confidence_path":   None,
            "ocr_lines":         [],
            "elapsed_ocr_s":     elapsed,
            "error":             str(exc),
        }


def _load_ocr_raw(ocr_raw_dir: Path) -> dict:
    txt_files = sorted(ocr_raw_dir.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in --from-ocr directory: %s", ocr_raw_dir)
        sys.exit(1)

    results = {}
    for txt_path in txt_files:
        lines = [l for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        key = str(txt_path)
        results[key] = {
            "source":            key,
            "preprocessed_path": None,   # not available in this mode
            "ocr_raw_path":      key,
            "confidence_path":   None,   # not available in this mode
            "ocr_lines":         lines,
            "elapsed_ocr_s":     0.0,
            "stem":              txt_path.stem,
        }
        print(f"  [RAW] {txt_path.name:<40}  {len(lines):>3} lines", flush=True)

    logger.info("Loaded %d OCR file(s) from %s", len(results), ocr_raw_dir)
    return results


def run(input_path, output_dir, ocr_workers=2, llm_workers=2, enhance=True,
        from_ocr_dir=None, batch_size=8):
    import postprocessor as _pp
    _pp.BATCH_SIZE = max(1, batch_size)   # apply CLI override before any calls
    from postprocessor import postprocess_batch

    output_dir.mkdir(parents=True, exist_ok=True)

    t_ocr = time.perf_counter()

    if from_ocr_dir is not None:
        logger.info("--from-ocr mode: skipping OCR, loading text from %s", from_ocr_dir)
        ocr_results = _load_ocr_raw(from_ocr_dir)
        # Use the txt paths as the iteration order for Stage 3
        ordered_keys = list(ocr_results.keys())
    else:
        if input_path.is_dir():
            images = sorted(
                p for p in input_path.iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            )
        elif input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTS:
            images = [input_path]
        else:
            logger.error("Input must be an image file or a folder of images: %s", input_path)
            sys.exit(1)

        if not images:
            logger.error("No supported images found in %s", input_path)
            sys.exit(1)

        cpu_count   = os.cpu_count() or 4
        ocr_workers = max(1, min(ocr_workers, cpu_count, len(images)))
        llm_workers = max(1, min(llm_workers, 8))

        logger.info(
            "Processing %d image(s)  |  OCR processes: %d  |  LLM threads: %d",
            len(images), ocr_workers, llm_workers,
        )

        worker_args = [(str(p), str(output_dir), enhance) for p in images]
        ocr_results = {}
        with ProcessPoolExecutor(max_workers=ocr_workers) as pool:
            futures = {pool.submit(_ocr_worker, a): a[0] for a in worker_args}
            for fut in as_completed(futures):
                res = fut.result()
                ocr_results[res["source"]] = res

        ordered_keys = [str(p) for p in images]

    ocr_elapsed = time.perf_counter() - t_ocr
    logger.info("OCR stage done in %.1fs", ocr_elapsed)

    t_llm = time.perf_counter()

    lines_map = {
        src: res["ocr_lines"]
        for src, res in ocr_results.items()
        if res.get("ocr_lines") and not res.get("error")
    }

    llm_results = postprocess_batch(
        lines_map,
        max_workers=llm_workers,
        ocr_results=ocr_results,
        output_dir=output_dir,
    )
    llm_elapsed = time.perf_counter() - t_llm
    logger.info("LLM stage done in %.1fs", llm_elapsed)

    final_records = []

    for src in ordered_keys:
        ocr = ocr_results.get(src, {})
        stem = ocr.get("stem") or Path(src).stem

        if ocr.get("error") or not ocr.get("ocr_lines"):
            record = {
                "_meta": {
                    "source":            ocr.get("source", src),
                    "preprocessed_path": ocr.get("preprocessed_path"),
                    "ocr_raw_path":      ocr.get("ocr_raw_path"),
                    "confidence_path":   ocr.get("confidence_path"),
                    "ocr_line_count":    len(ocr.get("ocr_lines", [])),
                    "elapsed_ocr_s":     ocr.get("elapsed_ocr_s"),
                },
                "error": ocr.get("error") or "OCR returned no text",
            }
            out_json = output_dir / f"{stem}.json"
            out_json.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            final_records.append(record)
        else:
            # Already written by postprocess_batch; read it back for the summary
            out_json = output_dir / f"{stem}.json"
            try:
                record = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                record = llm_results.get(src, {"error": "result missing"})
            final_records.append(record)

    ok     = [r for r in final_records if "error" not in r]
    failed = [r for r in final_records if "error" in r]

    summary = {
        "stats": {
            "total":           len(final_records),
            "success":         len(ok),
            "failed":          len(failed),
            "elapsed_ocr_s":   round(ocr_elapsed, 2),
            "elapsed_llm_s":   round(llm_elapsed, 2),
            "elapsed_total_s": round(ocr_elapsed + llm_elapsed, 2),
        },
        "receipts": final_records,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(
        "Done — %d OK / %d failed | %.1fs total -> %s",
        len(ok), len(failed), ocr_elapsed + llm_elapsed, summary_path,
    )
    return summary_path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Receipt OCR pipeline — image(s) -> structured JSON",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("input", nargs="?", default=None,
                   help="Single image file OR folder of images.\n"
                        "Not required when --from-ocr is set.")
    p.add_argument("--output", "-o", default="./output",
                   help="Output folder (default: ./output)")
    p.add_argument("--workers", type=int, default=2,
                   help="OCR worker processes (default: 2)")
    p.add_argument("--llm-workers", type=int, default=2,
                   help="LLM threads for Ollama (default: 2).\n"
                        "Each thread sends one batch prompt concurrently.\n"
                        "Local Ollama rarely benefits above 3.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Receipts per LLM prompt (default: 8).\n"
                        "Higher = fewer round-trips; lower if responses truncate.")
    p.add_argument("--no-enhance", action="store_true",
                   help="Skip CLAHE contrast enhancement")
    p.add_argument("--from-ocr", metavar="DIR", default=None,
                   help="Skip OCR — load raw .txt files from DIR (ocr_raw/ from\n"
                        "a previous run) and send lines straight to the LLM.\n"
                        "Useful for re-running or tuning the LLM step.")
    args = p.parse_args()

    if args.from_ocr is None and args.input is None:
        p.error("'input' argument is required unless --from-ocr is provided.")

    return args


if __name__ == "__main__":
    # spawn is required on macOS/Windows, and safer on Linux with paddle
    mp.set_start_method("spawn", force=True)

    args = _parse_args()
    run(
        input_path   = Path(args.input) if args.input else None,
        output_dir   = Path(args.output),
        ocr_workers  = args.workers,
        llm_workers  = args.llm_workers,
        enhance      = not args.no_enhance,
        from_ocr_dir = Path(args.from_ocr) if args.from_ocr else None,
        batch_size   = args.batch_size,
    )
