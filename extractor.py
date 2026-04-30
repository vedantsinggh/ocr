import re
from datetime import datetime
from typing import Optional


DATE_PATTERNS = [
    (r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b', '%d/%m/%Y'),      # 29/01/2018
    (r'\b(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b', '%Y/%m/%d'),      # 2018/01/29
    (r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2})\b', '%d/%m/%y'),      # 29/01/18
    (r'\b(\d{1,2}[\/\-\.]\d{4})\s+\d{1,2}:\d{2}', '%m/%Y'),      # 1/2018 4:40 (truncated)
    (r'\b(\d{1,2}\s+\w{3,9}\s+\d{4})\b', '%d %B %Y'),            # 29 January 2018
    (r'\b(\w{3,9}\s+\d{1,2},?\s+\d{4})\b', '%B %d, %Y'),         # January 29, 2018
]

AMOUNT_PATTERN = re.compile(
    r'(?:RM|MYR|USD|\$|£|€|SGD|IDR|THB|PHP|INR|RS|₹)?\s*(\d{1,6}[.,]\d{2})',
    re.IGNORECASE
)

TOTAL_KEYWORDS = re.compile(
    r'\b(total|grand\s*total|amount\s*due|amount\s*payable|'
    r'subtotal|sub\s*total|net\s*total|balance\s*due|'
    r'total\s*amt|ttl|jumlah)\b',
    re.IGNORECASE
)

ITEM_EXCLUSION = re.compile(
    r'\b(tax|gst|sst|vat|service\s*charge|discount|rounding|'
    r'change|cash|tender|paid|balance|tip)\b',
    re.IGNORECASE
)


def _normalize_date(raw: str) -> Optional[str]:
    raw = raw.strip().replace('-', '/').replace('.', '/')
    for _, fmt in DATE_PATTERNS:
        try:
            adjusted_fmt = fmt.replace('-', '/').replace('.', '/')
            return datetime.strptime(raw, adjusted_fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return raw


def extract_date(lines: list[str]) -> dict:
    for line in lines:
        for pattern, _ in DATE_PATTERNS:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                normalized = _normalize_date(m.group(1))
                has_time = bool(re.search(r'\d{1,2}:\d{2}', line))
                confidence = 0.92 if has_time else 0.85
                return {"value": normalized, "confidence": round(confidence, 2),
                        "raw": m.group(1)}
    return {"value": None, "confidence": 0.0, "raw": None}


def extract_store_name(lines: list[str]) -> dict:
    biz_keywords = re.compile(
        r'\b(sdn\.?\s*bhd|berhad|inc|llc|ltd|limited|store|mart|'
        r'restaurant|cafe|shop|supermarket|hypermarket|market|'
        r'enterprise|trading|industries|group|holdings)\b',
        re.IGNORECASE
    )
    candidates = []
    for i, line in enumerate(lines[:8]):
        line = line.strip()
        if not line or len(line) < 3:
            continue
        score = 0.5
        if line.isupper():
            score += 0.2
        if biz_keywords.search(line):
            score += 0.2
        if i < 3:
            score += 0.1
        if re.search(r'\d{3,}', line):  
            score -= 0.3
        if re.search(r'\b(tel|fax|gst|tax|invoice|receipt|date)\b', line, re.IGNORECASE):
            score -= 0.3
        candidates.append((line, round(min(score, 0.98), 2)))

    if not candidates:
        return {"value": None, "confidence": 0.0}

    best = max(candidates, key=lambda x: x[1])
    return {"value": best[0], "confidence": best[1]}


def extract_total(lines: list[str]) -> dict:
    """
    Find total amount — prioritize lines with 'total payable' / 'grand total'
    over plain 'total', to avoid picking subtotals.
    """
    PRIORITY_KEYWORDS = re.compile(
        r'\b(total\s*amt\s*payable|amount\s*payable|grand\s*total|'
        r'total\s*payable|net\s*total|balance\s*due)\b',
        re.IGNORECASE
    )

    best = None
    best_conf = 0.0

    for line in lines:
        if not TOTAL_KEYWORDS.search(line):
            continue
        m = AMOUNT_PATTERN.search(line)
        if not m:
            continue
        amount_str = m.group(1).replace(',', '.')
        try:
            float(amount_str)
        except ValueError:
            continue

        conf = 0.75
        if PRIORITY_KEYWORDS.search(line):
            conf = 0.95
        elif re.search(r'\btotal\b', line, re.IGNORECASE):
            conf = 0.85

        if conf > best_conf:
            best_conf = conf
            currency_match = re.search(r'(RM|MYR|USD|\$|£|€|SGD|INR)', line, re.IGNORECASE)
            currency = currency_match.group(1).upper() if currency_match else ''
            best = {"value": f"{currency} {amount_str}".strip(),
                    "confidence": round(conf, 2)}

    if best:
        return best
    return {"value": None, "confidence": 0.0}


def extract_items(lines: list[str]) -> list[dict]:
    items = []
    for i, line in enumerate(lines):
        if ITEM_EXCLUSION.search(line):
            continue
        if TOTAL_KEYWORDS.search(line):
            continue

        m = AMOUNT_PATTERN.search(line)
        if m:
            price_str = m.group(1).replace(',', '.')
            name = line[:m.start()].strip()
            name = re.sub(r'[\*\|#@]', '', name).strip()

            if len(name) < 2:
                if i > 0 and len(lines[i - 1].strip()) > 2:
                    name = lines[i - 1].strip()
                else:
                    continue

            has_qty = bool(re.search(r'\b\d+\s*[xX@]\b|\bqty\b|\bpcs\b', line, re.IGNORECASE))
            conf = 0.82 if has_qty else 0.70

            items.append({
                "name": name,
                "price": price_str,
                "confidence": round(conf, 2)
            })

    return items


def compute_overall_confidence(fields: dict) -> float:
    weights = {"store_name": 0.2, "date": 0.2, "total_amount": 0.4, "items": 0.2}
    score = 0.0
    score += weights["store_name"] * fields["store_name"]["confidence"]
    score += weights["date"] * fields["date"]["confidence"]
    score += weights["total_amount"] * fields["total_amount"]["confidence"]
    if fields["items"]:
        avg_item_conf = sum(i["confidence"] for i in fields["items"]) / len(fields["items"])
        score += weights["items"] * avg_item_conf
    return round(score, 3)


def flag_low_confidence(fields: dict, threshold: float = 0.7) -> list[str]:
    flagged = []
    for field in ["store_name", "date", "total_amount"]:
        if fields[field]["confidence"] < threshold:
            flagged.append(field)
    if fields["items"]:
        low_items = [i["name"] for i in fields["items"] if i["confidence"] < threshold]
        if low_items:
            flagged.append(f"items ({len(low_items)} low-confidence)")
    return flagged


def extract_all(ocr_lines: list[str], ocr_confidences: list[float] = None) -> dict:
    ocr_avg_conf = round(sum(ocr_confidences) / len(ocr_confidences), 3) \
        if ocr_confidences else None

    store = extract_store_name(ocr_lines)
    date = extract_date(ocr_lines)
    total = extract_total(ocr_lines)
    items = extract_items(ocr_lines)

    if ocr_avg_conf and ocr_avg_conf > 0.85:
        for field in [store, date, total]:
            if field["value"]:
                field["confidence"] = round(min(field["confidence"] + 0.03, 0.99), 2)

    result = {
        "store_name": store,
        "date": date,
        "items": items,
        "total_amount": total,
        "ocr_engine_confidence": ocr_avg_conf,
    }

    result["overall_confidence"] = compute_overall_confidence(result)
    result["flagged_fields"] = flag_low_confidence(result)

    return result
