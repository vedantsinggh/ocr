import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "qwen2.5:7b"
TEMPERATURE = 0.05
TIMEOUT_S   = 180      # longer timeout to cover multi-receipt batches
MAX_RETRIES = 3
BATCH_SIZE  = 8        # receipts per LLM call; tune down if responses truncate
NUM_PREDICT = 4096     # must fit BATCH_SIZE full receipt JSONs

# ── Prompts ───────────────────────────────────────────────────────────────────

# Single-receipt system prompt (used as fallback / retry)
_SYSTEM_SINGLE = """\
You extract structured data from receipt OCR text.
Return ONLY a single valid JSON object — no prose, no markdown fences, no trailing commas.

Schema:
{
  "store":          string,
  "date":           string (YYYY-MM-DD if possible, else as printed),
  "items":          [{"name": string, "qty": number|null, "price": string}],
  "subtotal":       string|null,
  "tax":            string|null,
  "total":          string,
  "currency":       string  (e.g. "INR", "USD"),
  "payment_method": string|null
}

Rules:
- Fix broken OCR words ("Mil k" -> "Milk", "Rs" -> currency INR)
- store: use the prominent header/brand name, not branch address
- date: convert DD-MM-YYYY to YYYY-MM-DD
- items: only lines with a price; omit headers/dividers
- qty: null if not stated
- total: grand/final total if multiple appear
- tax: combine all tax lines into one value
- Ignore GST numbers, phone numbers, URLs, loyalty points
- Use null for genuinely missing fields — never invent data"""

# Multi-receipt system prompt
_SYSTEM_MULTI = """\
You extract structured data from multiple receipt OCR texts.
You will receive N receipts, each identified by a numeric key.
Return ONLY a single valid JSON object mapping each key to its receipt data.
No prose, no markdown fences, no trailing commas.

Output format:
{
  "0": { <receipt schema> },
  "1": { <receipt schema> },
  ...
}

Receipt schema for each entry:
{
  "store":          string,
  "date":           string (YYYY-MM-DD if possible, else as printed),
  "items":          [{"name": string, "qty": number|null, "price": string}],
  "subtotal":       string|null,
  "tax":            string|null,
  "total":          string,
  "currency":       string  (e.g. "INR", "USD"),
  "payment_method": string|null
}

Rules:
- Fix broken OCR words ("Mil k" -> "Milk", "Rs" -> currency INR)
- store: use the prominent header/brand name, not branch address
- date: convert DD-MM-YYYY to YYYY-MM-DD
- items: only lines with a price; omit headers/dividers
- qty: null if not stated
- total: grand/final total if multiple appear
- tax: combine all tax lines into one value
- Ignore GST numbers, phone numbers, URLs, loyalty points
- Use null for genuinely missing fields — never invent data"""


# ── Public API ────────────────────────────────────────────────────────────────

def postprocess(lines: list[str]) -> dict[str, Any]:
    """
    Convert a list of OCR text lines into a structured receipt dict.
    Uses single-receipt prompt.  Retries up to MAX_RETRIES on any failure.
    """
    prompt = "OCR lines:\n" + "\n".join(lines) + "\n\nExtract the receipt data as JSON."
    raw  = _call_ollama(prompt, _SYSTEM_SINGLE)
    data = _parse_json_single(raw)
    return _validate(data)


def postprocess_batch(
    lines_map: dict[str, list[str]],
    max_workers: int = 2,
    ocr_results: dict | None = None,
    output_dir: "Path | None" = None,
) -> dict[str, dict[str, Any]]:
    """
    Process all receipts using multi-receipt batching for speed.

    Splits lines_map into chunks of BATCH_SIZE, sends each chunk in one
    Ollama call, then unpacks and validates results.  Any chunk that fails
    to parse as a whole is retried one receipt at a time.

    Results are written to output_dir as JSON files immediately as each
    batch completes — you can inspect the first results without waiting
    for the full dataset to finish.

    Args:
        lines_map:   {source_key: [ocr_lines]}
        max_workers: parallel Ollama threads.  2–3 for local Ollama; the
                     bottleneck is GPU/CPU inference, not HTTP overhead.
        ocr_results: full OCR result dicts keyed by source_key, used to
                     build the _meta block written into each JSON file.
                     Optional — omit if you only want the return value.
        output_dir:  folder to write per-receipt JSON files into as results
                     arrive.  Optional — pass together with ocr_results.

    Returns:
        {source_key: structured_receipt_or_error_dict}
    """
    from pathlib import Path as _Path

    # Split into ordered chunks of BATCH_SIZE
    items      = list(lines_map.items())
    chunks     = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    results: dict[str, dict] = {}

    def _write_now(src: str, llm: dict) -> None:
        """Write a single receipt JSON to disk immediately after LLM returns."""
        if output_dir is None or ocr_results is None:
            return
        ocr  = ocr_results.get(src, {})
        stem = ocr.get("stem") or _Path(src).stem
        record = {
            "_meta": {
                "source":            ocr.get("source", src),
                "preprocessed_path": ocr.get("preprocessed_path"),
                "ocr_raw_path":      ocr.get("ocr_raw_path"),
                "ocr_line_count":    len(ocr.get("ocr_lines", [])),
                "elapsed_ocr_s":     ocr.get("elapsed_ocr_s"),
            }
        }
        if llm.get("error"):
            record["error"] = llm["error"]
        else:
            record.update({k: v for k, v in llm.items() if not k.startswith("_")})
        out_json = _Path(output_dir) / f"{stem}.json"
        out_json.write_text(
            __import__("json").dumps(record, indent=2, ensure_ascii=False)
        )

    def _process_chunk(chunk: list[tuple[str, list[str]]]) -> list[tuple[str, dict]]:
        """Returns [(source_key, result_dict), ...] for all items in the chunk."""
        if len(chunk) == 1:
            # Avoid batch overhead for singleton chunks
            key, lines = chunk[0]
            try:
                data = postprocess(lines)
                logger.info("[LLM] %-44s store=%-18s total=%s",
                            key, data.get("store", "?"), data.get("total", "?"))
                _write_now(key, data)
                return [(key, data)]
            except Exception as exc:
                logger.error("[LLM] Failed %s: %s", key, exc)
                err = {"error": str(exc)}
                _write_now(key, err)
                return [(key, err)]

        # Build multi-receipt prompt
        prompt_parts = []
        for idx, (key, lines) in enumerate(chunk):
            prompt_parts.append(f"--- Receipt {idx} ---\n" + "\n".join(lines))
        prompt = "\n\n".join(prompt_parts) + "\n\nExtract all receipts as a JSON object keyed by index."

        raw = _call_ollama(prompt, _SYSTEM_MULTI)

        # Try to parse the whole batch response
        try:
            batch_data = _parse_json_single(raw)
            out = []
            for idx, (key, lines) in enumerate(chunk):
                entry = batch_data.get(str(idx)) or batch_data.get(idx)
                if entry is None:
                    logger.warning("[LLM] Batch missing index %d (%s) — retrying solo", idx, key)
                    result = _solo_fallback(key, lines)
                    _write_now(result[0], result[1])
                    out.append(result)
                else:
                    validated = _validate(entry)
                    logger.info("[LLM] %-44s store=%-18s total=%s",
                                key, validated.get("store", "?"), validated.get("total", "?"))
                    _write_now(key, validated)
                    out.append((key, validated))
            return out

        except Exception as exc:
            # Whole batch parse failed — fall back to individual calls
            logger.warning("[LLM] Batch parse failed (%s) — retrying %d items solo", exc, len(chunk))
            out = []
            for result in [_solo_fallback(key, lines) for key, lines in chunk]:
                _write_now(result[0], result[1])
                out.append(result)
            return out

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process_chunk, chunk) for chunk in chunks]
        for fut in as_completed(futures):
            for key, data in fut.result():
                results[key] = data

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _solo_fallback(key: str, lines: list[str]) -> tuple[str, dict]:
    """Retry a single receipt with the single-receipt prompt."""
    try:
        data = postprocess(lines)
        logger.info("[LLM] %-44s store=%-18s total=%s (solo retry)",
                    key, data.get("store", "?"), data.get("total", "?"))
        return key, data
    except Exception as exc:
        logger.error("[LLM] Failed %s: %s", key, exc)
        return key, {"error": str(exc)}


def _call_ollama(prompt: str, system: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":   MODEL,
                    "system":  system,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": NUM_PREDICT,
                    },
                },
                timeout=TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as exc:
            logger.warning("Ollama attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Ollama unreachable after {MAX_RETRIES} attempts")


def _repair_json(text: str) -> str:
    """
    Fix the most common JSON mistakes LLMs make:
      1. Trailing commas before } or ]
      2. Single-quoted keys or string values
      3. Bare Python-style None / True / False
      4. Truncated output — find the last complete } and cut there

    This is intentionally conservative: it only removes/replaces characters
    rather than inserting them, so it cannot introduce wrong data.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Bare None/True/False (Python repr leaking through)
    text = re.sub(r'\bNone\b',  'null',  text)
    text = re.sub(r'\bTrue\b',  'true',  text)
    text = re.sub(r'\bFalse\b', 'false', text)

    # Single-quoted strings → double-quoted.
    # Use a capturing group for the preceding delimiter instead of a look-behind;
    # Python's re module requires look-behinds to be fixed-width, and \s* is not.
    text = re.sub(r"([:{,\[]\s*)'([^']*)'", r'\1"\2"', text)
    # Keys: 'key':
    text = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', text)
    # Also handle keys: 'key':

    # Trailing commas before closing brace/bracket
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # If the text is cut off mid-object, try to close it cleanly.
    # Strategy: truncate to the last complete top-level } we can find.
    if not text.endswith("}"):
        last_brace = text.rfind("}")
        if last_brace != -1:
            text = text[: last_brace + 1]
            # Remove any trailing comma that now appears before nothing
            text = re.sub(r',\s*$', '', text)

    return text


def _extract_json_object(text: str) -> str:
    """
    Return the substring spanning the outermost { ... } in text.
    This handles models that emit prose before or after the JSON block,
    e.g. "Here is the data:\n```json\n{...}\n```".
    Returns the original text unchanged if no braces are found.
    """
    start = text.find("{")
    if start == -1:
        return text
    # Walk forward counting depth to find the matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Unbalanced — return from first { to last } as best effort
    end = text.rfind("}") + 1
    return text[start:end] if end > start else text


def _parse_json_single(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from LLM output, applying repair steps on failure.

    Attempt order:
      1. Raw output as-is (fast path for well-behaved models)
      2. Strip markdown fences, then extract outermost { } block — this
         handles models that emit prose before/after the JSON, e.g.
         "Here is the data:" or trailing commentary
      3. Full repair pass (trailing commas, single quotes, bare None/True/False)
         applied to the extracted block, then parse again

    Raises ValueError if all three attempts fail.
    """
    original = text.strip()

    # Attempt 1: parse as-is
    try:
        return json.loads(original)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences, then isolate the outermost { } block.
    # Doing both together means prose preambles like
    # "Here is the extracted receipt data in JSON format:" are removed
    # before we try to parse, instead of causing a parse error.
    cleaned = re.sub(r"```(?:json)?", "", original).strip().rstrip("`").strip()
    extracted = _extract_json_object(cleaned)

    # Attempt 2: extracted block before any repair
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair pass on the extracted block
    repaired = _repair_json(extracted)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as final_exc:
        raise ValueError(
            f"JSON unparseable after repair.\n"
            f"Error: {final_exc}\n"
            f"Raw output (first 400 chars):\n{original[:400]}"
        ) from final_exc


def _clean_price(value) -> str | None:
    """
    Normalise a price string returned by the LLM.
    - Strips trailing tax-code letters that OCR picks up, e.g. "17.99 A" -> "17.99"
    - Strips stray whitespace
    - Returns None if nothing numeric remains
    """
    if value is None:
        return None
    s = str(value).strip()
    # Remove trailing uppercase letter(s) that are tax-code annotations
    s = re.sub(r"\s+[A-Z]+$", "", s).strip()
    # Must contain at least one digit to be a valid price
    return s if re.search(r"\d", s) else None


def _validate(data: dict) -> dict:
    """Ensure all schema keys exist, clean prices, and drop items without a price."""
    for key in ("store", "date", "subtotal", "tax", "total", "currency", "payment_method"):
        data.setdefault(key, None)
    data.setdefault("items", [])
    data["currency"] = data["currency"] or "INR"

    # Clean prices on line items
    for item in data["items"]:
        item["price"] = _clean_price(item.get("price"))

    # Clean top-level monetary fields
    for field in ("subtotal", "tax", "total"):
        if data[field] is not None:
            data[field] = _clean_price(data[field])

    # Drop items that ended up with no valid price after cleaning
    data["items"] = [
        i for i in data["items"]
        if i.get("price") not in (None, "", "null")
    ]
    return data
