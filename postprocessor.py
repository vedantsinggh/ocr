"""
postprocessor.py
----------------
Post-processing layer for the hybrid Tesseract + Donut receipt pipeline.

Responsibilities:
  1. Extract store name and date from raw Tesseract OCR text using regex rules.
  2. Validate and clean items extracted by Donut:
       - Reject lines that look like store name / address / phone / footer garbage
       - Ensure every item has a valid numeric price
       - Strip currency symbols and stray characters from prices
  3. Optional LLM correction pass (Claude claude-haiku-4-5-20251001 via Anthropic API) to:
       - Fix truncated/garbled item names using Tesseract raw text as ground truth
       - Fill missing items Donut missed by re-parsing Tesseract lines
       - Normalise store name and parse date if Tesseract heuristics failed
  4. Validate the overall result and flag low-confidence fields.
  5. Return a clean, structured dict with raw_ocr debug block included.
"""

import re
import os
import json
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# ── Date patterns ─────────────────────────────────────────────────────────────
_DATE_PATTERNS = [
    # MM/DD/YYYY or MM-DD-YYYY  (American — checked FIRST so "06-28-2014" hits here)
    (re.compile(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b'), 'us'),
    # YYYY/MM/DD or YYYY-MM-DD
    (re.compile(r'\b(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b'), 'iso'),
    # DD/MM/YY or MM/DD/YY  (two-digit year — try both interpretations)
    (re.compile(r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2})\b'), 'short'),
    # D Month YYYY  e.g. "1 January 2024"
    (re.compile(
        r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
        r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b', re.I), 'dmy_text'),
    # Month D, YYYY  e.g. "January 1, 2024"
    (re.compile(
        r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
        r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b', re.I), 'mdy_text'),
]

# ── Store name heuristics ─────────────────────────────────────────────────────
_BIZ_SUFFIX = re.compile(
    r'\b(sdn\.?\s*bhd|berhad|bhd|inc\.?|llc\.?|ltd\.?|limited|'
    r'enterprise|trading|industries|holdings|mart|'
    r'supermarket|hypermarket|restaurant|cafe|bistro|kitchen|'
    r'pharmacy|clinic|hardware|electronics)\b',
    re.I
)

# Lines that are clearly NOT a store name — NOTE: "store" removed so "TRADER JOE'S" isn't penalised
_NOT_STORE = re.compile(
    r'\b(tel|phone|fax|email|www\.|http|@|\.com|\.my|\.org|\.net|'
    r'jalan|lorong|taman|bandar|blok|level|floor|suite|no\.|lot|unit|'
    r'apt|street|st\.|ave|road|rd\.|postcode|zip|p\.?o\.?\s*box|'
    r'gst|sst|tax.?id|reg\.?no|invoice|receipt|'
    r'total|subtotal|amount|balance|cash|change|paid|'
    r'thank|thanks|welcome|enjoy|visit|operator|cashier|open)\b',
    re.I
)

def safe_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, list) and x:
        return safe_dict(x[0])
    return {}

def safe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        return str(x.get("text", "")).strip()
    if isinstance(x, list):
        return " ".join(safe_str(i) for i in x if i)
    return str(x).strip()

_PHONE_RE   = re.compile(r'\b\d{3,4}[-\s]\d{3,4}[-\s]?\d{3,4}\b')
_POSTCODE   = re.compile(r'\b\d{5}\b')
_LONG_NUM   = re.compile(r'\d{6,}')   # long digit run → likely barcode/ID

# ── Item garbage filters ──────────────────────────────────────────────────────
_SUMMARY_KW = re.compile(
    r'\b(total|subtotal|sub.total|grand.total|amount.payable|amt.payable|'
    r'amount.due|net.total|balance.due|rounding|adjustment|'
    r'tax|gst|sst|vat|service.charge|discount|service.tax|'
    r'cash|paid|change|tender|cashprice|refund|tip|deposit|'
    r'balance|void|nett|non.taxable|grocery.non)\b',
    re.I
)

_FOOTER_KW = re.compile(
    r'\b(thank|thanks|welcome|enjoy|visit|receipt|invoice|'
    r'exchange|cashier|served.by|operator|terminal|transaction|'
    r'not.responsible|void|valid|hotline|website|facebook|instagram|'
    r'twitter|please|keep|this|your|copy|open|daily|items)\b',
    re.I
)

_HEADER_KW = re.compile(
    r'\b(jalan|lorong|taman|bandar|blok|level|floor|suite|'
    r'no\.|lot|unit|apt|street|st\.|ave|road|rd\.|'
    r'tel|phone|fax|email|www\.|@|\.com|\.my|'
    r'gst.?id|sst.?id|tax.?id|invoice.?no|reg.?no)\b',
    re.I
)

# A name that looks like a store brand/header — short all-caps with no price context
_STORE_BRAND_RE = re.compile(r"^[A-Z][A-Z '\.\-]+\'?S?$")

# Price cleaning
_PRICE_STRIP   = re.compile(r'[^\d.,\-]')
_PRICE_VALID   = re.compile(r'^\-?\d{1,6}[.,]\d{1,2}$')
_PRICE_INTEGER = re.compile(r'^\-?\d{1,6}$')

# Tesseract item line: "ITEM NAME    1.29" style
_TESS_ITEM_LINE = re.compile(
    r'^([A-Z][A-Z0-9 \-\./\(\)\'&,#%]{2,40}?)\s{2,}(\d{1,4}\.\d{2})\s*$'
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Store name extraction from Tesseract text
# ─────────────────────────────────────────────────────────────────────────────

def extract_store_name(tess_text: str) -> dict:
    """
    Scan the first ~30% of Tesseract lines for the most likely store name.

    Priority rule: a standalone all-caps line near the top that isn't an
    address, phone, or financial keyword scores highest.
    """
    if not tess_text:
        return {"value": None, "confidence": 0.0}

    lines = [l.strip() for l in tess_text.splitlines() if l.strip()]
    if not lines:
        return {"value": None, "confidence": 0.0}

    n = len(lines)
    search_window = lines[:max(6, int(n * 0.30))]

    candidates = []
    for line in search_window:
        if len(line) < 3:
            continue

        score = 0.0

        # Strong positive: all-caps non-address/non-financial word(s)
        if line.isupper() and not _NOT_STORE.search(line) and not _PHONE_RE.search(line):
            score += 4
        # Business suffix
        if _BIZ_SUFFIX.search(line):
            score += 3
        # Has enough alpha chars
        if len(re.findall(r'[A-Za-z]', line)) >= 4:
            score += 1

        # Negatives
        if _NOT_STORE.search(line):
            score -= 3
        if _PHONE_RE.search(line):
            score -= 3
        if _POSTCODE.search(line) or _LONG_NUM.search(line):
            score -= 3
        if re.match(r'^\d', line):
            score -= 2
        # Lines that contain an address-style number+street pattern
        if re.search(r'\d{3,}\s+\w', line):
            score -= 2

        if score > 0:
            candidates.append((line, score))

    if not candidates:
        return {"value": None, "confidence": 0.0}

    best_line, best_score = max(candidates, key=lambda x: x[1])
    conf = min(0.55 + best_score * 0.06, 0.95)
    return {"value": best_line, "confidence": round(conf, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Date extraction from Tesseract text
# ─────────────────────────────────────────────────────────────────────────────

def _try_parse_date(raw: str, mode: str) -> Optional[str]:
    """
    Try to parse a date string with awareness of US (MM-DD-YYYY) vs ISO order.
    Returns ISO YYYY-MM-DD string or None.
    """
    sep = raw[2] if len(raw) > 2 and raw[2] in '/-.' else '/'
    parts = re.split(r'[\/\-\.]', raw)

    if mode == 'iso' and len(parts) == 3:
        try:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2])).strftime('%Y-%m-%d')
        except ValueError:
            return None

    if mode == 'us' and len(parts) == 3:
        y = int(parts[2]) if len(parts[2]) == 4 else None
        if y is None:
            return None
        # Try MM-DD-YYYY first (American), fall back to DD-MM-YYYY
        for m_idx, d_idx in [(0, 1), (1, 0)]:
            try:
                return datetime(y, int(parts[m_idx]), int(parts[d_idx])).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    if mode == 'short' and len(parts) == 3:
        y_short = int(parts[2])
        y = 2000 + y_short if y_short < 70 else 1900 + y_short
        for m_idx, d_idx in [(0, 1), (1, 0)]:
            try:
                return datetime(y, int(parts[m_idx]), int(parts[d_idx])).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    if mode == 'dmy_text':
        for fmt in ('%d %B %Y', '%d %b %Y'):
            try:
                return datetime.strptime(raw, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    if mode == 'mdy_text':
        raw_clean = raw.replace(',', '')
        for fmt in ('%B %d %Y', '%b %d %Y'):
            try:
                return datetime.strptime(raw_clean, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    return None


def extract_date(tess_text: str) -> dict:
    """
    Scan all Tesseract lines for the first parseable date string.
    Prioritises lines that also contain a time component (higher confidence).
    Returns ISO YYYY-MM-DD or None.
    """
    if not tess_text:
        return {"value": None, "confidence": 0.0}

    for line in tess_text.splitlines():
        line = line.strip()
        if not line:
            continue

        for pattern, mode in _DATE_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            raw = m.group(1)
            iso = _try_parse_date(raw, mode)
            if iso:
                has_time = bool(re.search(r'\d{1,2}:\d{2}', line))
                conf = 0.95 if has_time else 0.88
                return {"value": iso, "confidence": conf, "_raw": raw}

    return {"value": None, "confidence": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Price cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_price(raw) -> Optional[str]:
    """
    Strip currency symbols, spaces, and stray characters from a price string.
    Returns a normalised decimal string (e.g. "12.50") or None if invalid.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    s = _PRICE_STRIP.sub('', s)

    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    elif ',' in s and '.' in s:
        s = s.replace(',', '')

    if not s:
        return None

    if _PRICE_VALID.match(s):
        try:
            val = float(s)
            # Sanity check: single receipt item shouldn't exceed $9999
            if abs(val) > 9999:
                return None
            return s
        except ValueError:
            return None

    if _PRICE_INTEGER.match(s):
        try:
            val = float(s)
            if abs(val) > 9999:
                return None
            return s + '.00'
        except ValueError:
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Item validation / cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _safe_text(x):
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        return str(x.get("text", "")).strip()
    return str(x).strip()



def _is_garbage_item(name: str, price: Optional[str] = None) -> bool:
    """Return True if the item name looks like header/footer/summary garbage."""
    if isinstance(name, dict):
        name = str(name.get("text", ""))
    elif not isinstance(name, str):
        name = str(name or "")

    t = name.strip()

    if not t or len(t) < 2:
        return True

    # Summary / financial lines
    if _SUMMARY_KW.search(t):
        return True
    # Footer / thank-you lines
    if _FOOTER_KW.search(t):
        return True
    # Address / contact header lines
    if _HEADER_KW.search(t):
        return True
    # Phone number
    if _PHONE_RE.search(t):
        return True
    # Barcode / GST id — long numeric run
    if _LONG_NUM.search(t):
        return True
    # Purely numeric
    if re.match(r'^[\d\s\.,\-]+$', t):
        return True

    # If name looks like a store brand (e.g. "TRADER JOE'S") AND price looks
    # like a zip/postcode (5+ digits before decimal), reject it
    if price:
        try:
            val = float(price)
            if val >= 1000:   # no grocery item costs $1000+
                return True
        except (ValueError, TypeError):
            pass

    return False


def validate_items(donut_menu: list) -> list:
    """
    Accept the raw menu list from Donut's CORD output.
    Each entry may have: nm, unitprice, cnt, price.
    Returns a clean list of items with validated prices.
    """
    cleaned_items = []

    for entry in donut_menu:
        if isinstance(entry, str):
            possible_price = clean_price(entry)
            if possible_price:
                cleaned_items.append({
                    "name": entry.strip(),
                    "price": possible_price,
                    "quantity": "1",
                    "confidence": 0.4,
                })
            continue

        if not isinstance(entry, dict):
            continue

        name      = safe_str(entry.get("nm") or "")
        price_raw = safe_str(entry.get("price") or entry.get("unitprice") or "")
        qty_raw   = safe_str(entry.get("cnt", "1") or "1")

        price = clean_price(price_raw)

        # Reject before garbage check so we can pass price for context
        if _is_garbage_item(name, price):
            continue

        if price is None:
            continue

        conf = 0.60
        if name:
            conf += 0.15
        if entry.get("cnt"):
            conf += 0.10
        if entry.get("unitprice") and entry.get("price"):
            conf += 0.10
        conf = round(min(conf, 0.95), 2)

        cleaned_items.append({
            "name":       name,
            "price":      price,
            "quantity":   qty_raw,
            "confidence": conf,
        })

    return cleaned_items


# ─────────────────────────────────────────────────────────────────────────────
# 4b.  Tesseract fallback item parser
#      When Donut misses many items, extract them directly from OCR text.
# ─────────────────────────────────────────────────────────────────────────────

def _parse_items_from_tesseract(tess_text: str) -> list:
    """
    Best-effort item extraction from Tesseract output.
    Looks for lines of the form:  ITEM NAME    1.29
    Only used as a fallback / supplement when Donut returns few items.
    """
    items = []
    for line in tess_text.splitlines():
        line = line.strip()
        m = _TESS_ITEM_LINE.match(line)
        if not m:
            continue
        name  = m.group(1).strip()
        price = clean_price(m.group(2))
        if price and not _is_garbage_item(name, price):
            items.append({
                "name":       name,
                "price":      price,
                "quantity":   "1",
                "confidence": 0.55,
                "_source":    "tesseract_fallback",
            })
    return items


def _llm_correct(structured: dict, tess_text: str, donut_raw: dict) -> dict:
    """
    Local LLM correction using Ollama instead of Claude.
    """

    import os, json, re, requests

    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")  # change if needed

    system_prompt = """You are a receipt data correction assistant.

You will be given:
1. A draft JSON of extracted receipt data (may have errors / missing items).
2. The raw Tesseract OCR text from the same receipt.
3. The raw Donut model output dict.

Your job:
- Fix garbled or truncated item names using the Tesseract text as the source of truth.
- Add any items present in the Tesseract text that were missed. Set confidence=0.70 and _source="llm".
- If store_name.value is wrong or null, correct it from Tesseract text.
- If date.value is null, extract it in ISO YYYY-MM-DD.
- Do NOT invent items.
- Do NOT change prices.
- Return ONLY valid JSON. No markdown. No explanation.
"""

    user_msg = json.dumps({
        "draft": structured,
        "tesseract": tess_text,
        "donut_raw": donut_raw,
    }, ensure_ascii=False)

    prompt = f"""<system>
{system_prompt}
</system>

<user>
{user_msg}
</user>
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,   # IMPORTANT: keep deterministic
                    "num_ctx": 4096
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(response.text)

        raw_resp = response.json()["response"].strip()

        # clean junk (local models love to hallucinate markdown)
        raw_resp = re.sub(r'^```json\s*', '', raw_resp)
        raw_resp = re.sub(r'^```\s*', '', raw_resp)
        raw_resp = re.sub(r'\s*```$', '', raw_resp)

        corrected = json.loads(raw_resp)

        corrected["raw_ocr"] = structured.get("raw_ocr", {})
        corrected["llm_correction"] = True

        return corrected

    except Exception as e:
        log.warning(f"Ollama correction failed ({e}) — using original structured data")
        return structured
# ─────────────────────────────────────────────────────────────────────────────
# 6.  Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def _overall_confidence(store_conf: float, date_conf: float,
                        total_conf: float, items: list) -> float:
    item_avg = (sum(i["confidence"] for i in items) / len(items)) if items else 0.0
    return round(
        0.15 * store_conf +
        0.20 * date_conf  +
        0.45 * total_conf +
        0.20 * item_avg,
        3
    )


def flag_low_confidence(structured: dict, threshold: float = 0.70) -> list:
    flagged = []
    for field in ("store_name", "date", "total_amount"):
        if structured[field].get("confidence", 0) < threshold:
            flagged.append(field)
    low = [i for i in structured.get("items", []) if i.get("confidence", 1) < threshold]
    if low:
        flagged.append(f"items ({len(low)} low-confidence)")
    if not structured.get("items"):
        flagged.append("items (none detected)")
    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Master entry point
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(tess_text: str, donut_output: dict,
                use_llm: bool = True) -> dict:
    """
    Combine Tesseract OCR text (for store/date) with Donut output (for items).

    Parameters
    ----------
    tess_text    : raw string from pytesseract.image_to_string()
    donut_output : dict from DonutModel.inference()["predictions"][0]
    use_llm      : if True (default), attempt a Claude Haiku correction pass.
                   Set False to skip (useful for unit tests / offline runs).

    Returns
    -------
    Structured receipt dict with confidence scores, flagged fields,
    and a raw_ocr debug block.
    """
    # ── Extract store name & date from Tesseract ──────────────────────────
    store = extract_store_name(tess_text)
    date  = extract_date(tess_text)

    # ── Extract & validate items from Donut ──────────────────────────────
    menu = donut_output.get("menu", [])
    if not isinstance(menu, list):
        menu = []
    items = validate_items(menu)

    # ── Supplement with Tesseract fallback if Donut returned too few items ─
    # (heuristic: if Donut found <30% of what Tesseract sees, use Tesseract)
    tess_items = _parse_items_from_tesseract(tess_text)
    if len(items) < max(3, len(tess_items) * 0.3):
        log.info(
            f"Donut returned only {len(items)} items vs {len(tess_items)} "
            "from Tesseract — merging"
        )
        # Merge: keep Donut items (higher confidence), fill gaps from Tesseract
        donut_names = {i["name"].upper() for i in items}
        for ti in tess_items:
            if ti["name"].upper() not in donut_names:
                items.append(ti)

    # ── Total ─────────────────────────────────────────────────────────────
    total_block = safe_dict(donut_output.get("total"))
    sub_block   = safe_dict(donut_output.get("sub_total"))

    total_raw = (
        safe_get(total_block, "total_price")
        or safe_get(total_block, "total_etc")
        or safe_get(sub_block, "subtotal_price")
        or ""
    )

    total_price = clean_price(total_raw)
    total_conf  = 0.96 if total_price else 0.0

    tax_raw = (
        safe_get(sub_block, "tax_price")
        or safe_get(sub_block, "tax")
        or ""
    )
    tax_price = clean_price(tax_raw)

    # ── Assemble result ───────────────────────────────────────────────────
    overall = _overall_confidence(
        store["confidence"], date["confidence"], total_conf, items
    )

    result = {
        "store_name":   store,
        "date":         date,
        "items":        items,
        "total_amount": {
            "value":      total_price,
            "confidence": total_conf,
        },
        "tax_amount": {
            "value": tax_price,
        },
        "cash_given": clean_price(safe_get(total_block, "cashprice")),
        "change":     clean_price(safe_get(total_block, "changeprice")),
        "overall_confidence": overall,
        "ocr_engines": {
            "store_date": "tesseract",
            "items":      "donut",
        },
        # ── Raw OCR debug block ───────────────────────────────────────────
        "raw_ocr": {
            "tesseract_text": tess_text,
            "donut_raw":      donut_output,
            "donut_menu_raw": menu,
        },
    }

    result["flagged_fields"] = flag_low_confidence(result)

    # ── Optional LLM correction pass ─────────────────────────────────────
    if use_llm:
        result = _llm_correct(result, tess_text, donut_output)
        # Re-compute flags after LLM may have fixed store/date/items
        result["flagged_fields"] = flag_low_confidence(result)
        result["overall_confidence"] = _overall_confidence(
            result["store_name"].get("confidence", 0),
            result["date"].get("confidence", 0),
            result["total_amount"].get("confidence", 0),
            result.get("items", []),
        )

    return result
