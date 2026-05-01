"""
summary_report.py
-----------------
Reads the summary.json produced by pipeline.py and generates:
  - financial_summary.json   (structured aggregates)
  - report.html              (self-contained visual dashboard)

Usage:
    python summary_report.py --results ./output/summary.json
    python summary_report.py --results ./output/summary.json --output-dir ./reports

Confidence scores are loaded automatically from ./output/confidence/*.json
when present (produced by the OCR pipeline's _ocr_worker).
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_amount(value) -> float:
    if not value:
        return 0.0
    nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    return float(nums[0]) if nums else 0.0


def stem(source: str) -> str:
    return Path(source).stem if source else "unknown"


def load_confidence_data(results_path: Path) -> dict:
    """
    Load all confidence/<stem>.json files from the sibling confidence/ folder.
    Returns a dict keyed by stem -> confidence payload.
    """
    conf_dir = Path("./output/confidence")
    if not conf_dir.exists():
        return {}
    data = {}
    for p in conf_dir.glob("*.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            data[p.stem] = payload
        except Exception:
            pass
    return data


def confidence_tier(avg: float | None) -> str:
    """Map average confidence to a human-readable tier."""
    if avg is None:
        return "n/a"
    if avg >= 0.95:
        return "high"
    if avg >= 0.80:
        return "medium"
    return "low"


# ── Build structured summary ──────────────────────────────────────────────────

def build_summary(receipts: list[dict], stats: dict, conf_data: dict) -> dict:
    total_spend = 0.0
    store_spend = defaultdict(float)
    monthly_spend = defaultdict(float)
    currency_counts = defaultdict(int)
    payment_counts = defaultdict(int)
    items_all = []
    failed_files = []
    ocr_line_counts = []

    # Confidence aggregates
    conf_avgs = []               # one float per receipt that has conf data
    low_conf_files = []          # stems where avg_confidence < 0.80
    per_receipt_conf = {}        # stem -> avg_confidence (for the receipt table)

    for r in receipts:
        meta = r.get("_meta", {})
        src = meta.get("source", "")
        s = stem(src)

        # Pull in confidence for this receipt
        if s in conf_data:
            avg_c = conf_data[s].get("avg_confidence")
            if avg_c is not None:
                conf_avgs.append(avg_c)
                per_receipt_conf[s] = avg_c
                if avg_c < 0.80:
                    low_conf_files.append(s)

        if r.get("error"):
            failed_files.append(src or "unknown")
            continue

        amt = parse_amount(r.get("total"))
        total_spend += amt

        store = r.get("store") or "Unknown"
        store_spend[store] += amt

        date_val = r.get("date")
        if date_val:
            try:
                month = datetime.strptime(date_val[:10], "%Y-%m-%d").strftime("%Y-%m")
                monthly_spend[month] += amt
            except Exception:
                pass

        currency = r.get("currency") or "Unknown"
        currency_counts[currency] += 1

        pm = r.get("payment_method") or "Unknown"
        payment_counts[pm] += 1

        for item in r.get("items", []):
            items_all.append({
                "store": store,
                "name": item.get("name", ""),
                "qty": item.get("qty"),
                "price": parse_amount(item.get("price")),
            })

        lines = meta.get("ocr_line_count") or 0
        if lines:
            ocr_line_counts.append(lines)

    transactions = stats.get("success", 0)
    avg_ocr_lines = round(sum(ocr_line_counts) / len(ocr_line_counts), 1) if ocr_line_counts else 0

    # Confidence summary stats
    overall_avg_conf = round(sum(conf_avgs) / len(conf_avgs), 4) if conf_avgs else None
    min_conf = round(min(conf_avgs), 4) if conf_avgs else None
    max_conf = round(max(conf_avgs), 4) if conf_avgs else None

    # Confidence distribution buckets
    conf_dist = {"high (>=0.95)": 0, "medium (0.80-0.95)": 0, "low (<0.80)": 0}
    for v in conf_avgs:
        if v >= 0.95:
            conf_dist["high (>=0.95)"] += 1
        elif v >= 0.80:
            conf_dist["medium (0.80-0.95)"] += 1
        else:
            conf_dist["low (<0.80)"] += 1

    # Top items by frequency
    item_freq = defaultdict(int)
    for it in items_all:
        if it["name"]:
            item_freq[it["name"].title()] += 1
    top_items = sorted(item_freq.items(), key=lambda x: -x[1])[:20]

    return {
        "generated_at": datetime.now().isoformat(),
        "pipeline_stats": stats,
        "total_spend": round(total_spend, 2),
        "total_transactions": transactions,
        "failed_count": len(failed_files),
        "average_transaction": round(total_spend / transactions, 2) if transactions else 0,
        "total_items_detected": len(items_all),
        "avg_ocr_lines_per_receipt": avg_ocr_lines,
        "unique_stores": len(store_spend),
        "spend_per_store": dict(sorted(store_spend.items(), key=lambda x: -x[1])),
        "spend_per_month": dict(sorted(monthly_spend.items())),
        "currency_distribution": dict(sorted(currency_counts.items(), key=lambda x: -x[1])),
        "payment_method_distribution": dict(sorted(payment_counts.items(), key=lambda x: -x[1])),
        "top_items": [{"name": n, "count": c} for n, c in top_items],
        "failed_files": failed_files,
        # ── Confidence section ──────────────────────────────────────────────
        "ocr_confidence": {
            "receipts_with_data": len(conf_avgs),
            "overall_avg": overall_avg_conf,
            "min": min_conf,
            "max": max_conf,
            "distribution": conf_dist,
            "low_confidence_files": low_conf_files,
            "per_receipt": {s: round(v, 4) for s, v in sorted(
                per_receipt_conf.items(), key=lambda x: x[1]
            )},
        },
    }


# ── HTML renderer ─────────────────────────────────────────────────────────────

def render_html(summary: dict, receipts: list[dict], conf_data: dict) -> str:

    def bar(pct, color):
        return (
            f'<div style="flex:1;height:8px;background:#f0f0ee;border-radius:4px;overflow:hidden">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px"></div></div>'
        )

    def conf_badge(avg):
        if avg is None:
            return '<span class="badge badge-na">n/a</span>'
        tier = confidence_tier(avg)
        cls = {"high": "badge-conf-high", "medium": "badge-conf-med", "low": "badge-conf-low"}[tier]
        return f'<span class="badge {cls}">{avg:.2%}</span>'

    # Stat cards
    cards_data = [
        ("transactions", str(summary["total_transactions"])),
        ("total spend", f"{summary['total_spend']:,.2f}"),
        ("avg transaction", f"{summary['average_transaction']:,.2f}"),
        ("unique stores", str(summary["unique_stores"])),
        ("items detected", str(summary["total_items_detected"])),
        ("failed", str(summary["failed_count"])),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="card-val">{val}</div><div class="card-lbl">{lbl}</div></div>'
        for lbl, val in cards_data
    )

    # Timing cards
    ps = summary["pipeline_stats"]
    ocr_s = ps.get("elapsed_ocr_s", 0)
    llm_s = ps.get("elapsed_llm_s", 0)
    total_s = ps.get("elapsed_total_s", 0)
    total_imgs = ps.get("total", 1)
    timing_html = "".join(
        f'<div class="tcard"><div class="tcard-val">{val}</div><div class="tcard-lbl">{lbl}</div></div>'
        for lbl, val in [
            ("ocr stage", f"{ocr_s:.1f}s"),
            ("llm stage", f"{llm_s/60:.1f}min"),
            ("total wall time", f"{total_s/60:.1f}min"),
            ("receipts/min", f"{total_imgs/(total_s/60):.1f}" if total_s > 0 else "—"),
        ]
    )

    # ── Confidence section ────────────────────────────────────────────────────
    oc = summary["ocr_confidence"]
    overall_avg = oc.get("overall_avg")
    conf_dist = oc.get("distribution", {})
    total_with_conf = oc.get("receipts_with_data", 0)

    conf_dist_html = ""
    dist_colors = {
        "high (>=0.95)":    "#3B6D11",
        "medium (0.80-0.95)": "#BA7517",
        "low (<0.80)":      "#A32D2D",
    }
    for label, count in conf_dist.items():
        pct = count / total_with_conf * 100 if total_with_conf else 0
        color = dist_colors.get(label, "#888")
        conf_dist_html += (
            f'<div class="dist-row">'
            f'<span class="dist-label" style="width:160px">{label}</span>'
            f'{bar(pct, color)}'
            f'<span class="dist-count">{count}</span>'
            f'</div>'
        )

    low_conf_list_html = ""
    for f in oc.get("low_confidence_files", []):
        avg_v = oc["per_receipt"].get(f)
        pct_str = f"{avg_v:.2%}" if avg_v is not None else "?"
        low_conf_list_html += f'<li><span class="conf-fname">{f}</span><span class="conf-val-low">{pct_str}</span></li>'
    if not low_conf_list_html:
        low_conf_list_html = '<li class="empty">None — all receipts above 80%</li>'

    # Per-receipt confidence detail table (sorted worst→best)
    per_receipt_sorted = sorted(
        oc.get("per_receipt", {}).items(), key=lambda x: x[1]
    )
    per_receipt_rows = ""
    for s, avg_v in per_receipt_sorted:
        tier = confidence_tier(avg_v)
        tier_cls = {"high": "badge-conf-high", "medium": "badge-conf-med", "low": "badge-conf-low"}[tier]
        # Pull line count from conf_data if available
        line_count = conf_data.get(s, {}).get("line_count", "—")
        worst_lines = sorted(
            conf_data.get(s, {}).get("lines", []),
            key=lambda x: x.get("confidence", 1),
        )[:3]
        snippets = "; ".join(
            f'"{ln["text"][:28]}" ({ln["confidence"]:.0%})' for ln in worst_lines
        )
        per_receipt_rows += f"""<tr>
          <td class="td-file">{s}</td>
          <td class="td-num">{line_count}</td>
          <td><span class="badge {tier_cls}">{avg_v:.2%}</span></td>
          <td class="td-snippets">{snippets}</td>
        </tr>"""
    if not per_receipt_rows:
        per_receipt_rows = '<tr><td colspan="4" class="empty">No confidence data found</td></tr>'

    conf_summary_cards = f"""
    <div class="conf-cards">
      <div class="conf-card">
        <div class="card-val">{f"{overall_avg:.2%}" if overall_avg is not None else "—"}</div>
        <div class="card-lbl">overall avg confidence</div>
      </div>
      <div class="conf-card">
        <div class="card-val">{f"{oc['min']:.2%}" if oc.get('min') is not None else "—"}</div>
        <div class="card-lbl">lowest receipt avg</div>
      </div>
      <div class="conf-card">
        <div class="card-val">{f"{oc['max']:.2%}" if oc.get('max') is not None else "—"}</div>
        <div class="card-lbl">highest receipt avg</div>
      </div>
      <div class="conf-card">
        <div class="card-val">{oc.get('receipts_with_data', 0)}</div>
        <div class="card-lbl">receipts with data</div>
      </div>
    </div>"""

    # Store table (top 20)
    store_rows = ""
    stores = list(summary["spend_per_store"].items())[:20]
    max_store_amt = stores[0][1] if stores else 1
    for store, amt in stores:
        pct = amt / summary["total_spend"] * 100 if summary["total_spend"] else 0
        w = amt / max_store_amt * 100
        store_rows += f"""<tr>
          <td class="td-store">{store}</td>
          <td class="td-num">{amt:,.2f}</td>
          <td class="td-pct">{pct:.1f}%</td>
          <td class="td-bar">{bar(w, "#3B6D11")}</td>
        </tr>"""
    if not store_rows:
        store_rows = '<tr><td colspan="4" class="empty">No data</td></tr>'

    # Monthly table
    monthly_rows = ""
    months = list(summary["spend_per_month"].items())
    max_month = max((v for _, v in months), default=1)
    for month, amt in months:
        w = amt / max_month * 100
        monthly_rows += f"""<tr>
          <td class="td-store">{month}</td>
          <td class="td-num">{amt:,.2f}</td>
          <td class="td-bar" style="width:200px">{bar(w, "#185FA5")}</td>
        </tr>"""
    if not monthly_rows:
        monthly_rows = '<tr><td colspan="3" class="empty">No data</td></tr>'

    # Payment / currency side by side
    pm_rows = ""
    for pm, cnt in summary["payment_method_distribution"].items():
        pct = cnt / summary["total_transactions"] * 100 if summary["total_transactions"] else 0
        pm_rows += f'<div class="dist-row"><span class="dist-label">{pm}</span>{bar(pct, "#534AB7")}<span class="dist-count">{cnt}</span></div>'
    if not pm_rows:
        pm_rows = '<p class="empty">No data</p>'

    cur_rows = ""
    for cur, cnt in summary["currency_distribution"].items():
        pct = cnt / summary["total_transactions"] * 100 if summary["total_transactions"] else 0
        cur_rows += f'<div class="dist-row"><span class="dist-label">{cur}</span>{bar(pct, "#0F6E56")}<span class="dist-count">{cnt}</span></div>'
    if not cur_rows:
        cur_rows = '<p class="empty">No data</p>'

    # Top items
    top_items_html = ""
    max_item_count = summary["top_items"][0]["count"] if summary["top_items"] else 1
    for item in summary["top_items"]:
        w = item["count"] / max_item_count * 100
        top_items_html += f'<div class="dist-row"><span class="dist-label" style="width:220px">{item["name"]}</span>{bar(w, "#BA7517")}<span class="dist-count">{item["count"]}</span></div>'
    if not top_items_html:
        top_items_html = '<p class="empty">No data</p>'

    # Receipt table (all receipts) — now includes a confidence column
    per_receipt_map = oc.get("per_receipt", {})
    receipt_rows = ""
    for r in receipts:
        meta = r.get("_meta", {})
        src = stem(meta.get("source", ""))
        avg_c = per_receipt_map.get(src)
        if r.get("error"):
            receipt_rows += f"""<tr class="tr-err">
              <td class="td-file">{src}</td>
              <td>—</td><td>—</td><td>—</td><td>—</td>
              <td><span class="badge badge-err">failed</span></td>
              <td>—</td>
              <td>{conf_badge(avg_c)}</td>
            </tr>"""
        else:
            store = r.get("store") or "—"
            date = r.get("date") or "—"
            total = r.get("total") or "—"
            currency = r.get("currency") or ""
            pm = r.get("payment_method") or "—"
            n_items = len(r.get("items", []))
            receipt_rows += f"""<tr>
              <td class="td-file">{src}</td>
              <td>{store}</td>
              <td>{date}</td>
              <td class="td-num">{currency} {total}</td>
              <td class="td-num">{n_items}</td>
              <td><span class="badge badge-ok">ok</span></td>
              <td>{pm}</td>
              <td>{conf_badge(avg_c)}</td>
            </tr>"""

    failed_section = ""
    if summary["failed_files"]:
        failed_list = "".join(f"<li>{f}</li>" for f in summary["failed_files"])
        failed_section = f"""
        <div class="section">
          <div class="section-title">failed files ({len(summary['failed_files'])})</div>
          <ul class="failed-list">{failed_list}</ul>
        </div>"""

    gen_time = summary["generated_at"][:19].replace("T", " at ")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Receipt Pipeline Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Georgia', serif;
    background: #faf9f6;
    color: #1c1c1a;
    font-size: 15px;
    line-height: 1.6;
  }}

  header {{
    background: #1c1c1a;
    color: #faf9f6;
    padding: 2.5rem 3rem 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
  }}
  .header-left h1 {{
    font-size: 1.9rem;
    font-weight: normal;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
  }}
  .header-left p {{
    font-size: 0.82rem;
    color: #888;
    font-family: 'Courier New', monospace;
  }}
  .header-right {{
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #666;
    text-align: right;
    line-height: 1.8;
  }}

  .container {{
    max-width: 1160px;
    margin: 0 auto;
    padding: 2.5rem 2rem;
  }}

  .cards {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: #e8e6e0;
    border: 1px solid #e8e6e0;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2rem;
  }}
  .card {{
    background: #faf9f6;
    padding: 1.25rem 1rem;
    text-align: center;
  }}
  .card-val {{
    font-size: 1.8rem;
    font-weight: normal;
    letter-spacing: -0.03em;
    color: #1c1c1a;
    font-family: 'Courier New', monospace;
  }}
  .card-lbl {{
    font-size: 0.72rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }}

  .timing {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #e8e6e0;
    border: 1px solid #e8e6e0;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2rem;
  }}
  .tcard {{
    background: #f2f0eb;
    padding: 1rem 1.25rem;
  }}
  .tcard-val {{
    font-size: 1.3rem;
    font-family: 'Courier New', monospace;
    color: #1c1c1a;
  }}
  .tcard-lbl {{
    font-size: 0.72rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 3px;
  }}

  /* Confidence summary cards */
  .conf-cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #e8e6e0;
    border: 1px solid #e8e6e0;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1.25rem;
  }}
  .conf-card {{
    background: #f7f5ee;
    padding: 1rem 1.25rem;
    text-align: center;
  }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin-bottom: 1.25rem;
  }}
  .grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1.25rem;
    margin-bottom: 1.25rem;
  }}

  .section {{
    background: white;
    border: 1px solid #e8e6e0;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
  }}
  .section-title {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
    margin-bottom: 1.1rem;
    font-family: 'Courier New', monospace;
  }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  th {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    font-weight: normal;
    padding: 0 0 0.6rem;
    text-align: left;
    border-bottom: 1px solid #e8e6e0;
    font-family: 'Courier New', monospace;
  }}
  td {{ padding: 0.5rem 0; border-bottom: 1px solid #f0eee8; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #faf9f6; }}
  .td-store {{ color: #1c1c1a; font-weight: normal; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .td-num {{ font-family: 'Courier New', monospace; font-size: 0.85rem; text-align: right; padding-right: 1rem; }}
  .td-pct {{ font-family: 'Courier New', monospace; font-size: 0.78rem; color: #888; width: 52px; text-align: right; padding-right: 0.75rem; }}
  .td-bar {{ width: 120px; }}
  .td-file {{ font-family: 'Courier New', monospace; font-size: 0.8rem; color: #666; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .td-err {{ font-size: 0.78rem; color: #A32D2D; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .td-snippets {{ font-family: 'Courier New', monospace; font-size: 0.72rem; color: #888; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  tr.tr-err td {{ background: #fef9f9; }}

  .badge {{
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.04em;
  }}
  .badge-ok  {{ background: #EAF3DE; color: #27500A; }}
  .badge-err {{ background: #FCEBEB; color: #791F1F; }}
  .badge-na  {{ background: #f0eee8; color: #aaa; }}
  .badge-conf-high {{ background: #EAF3DE; color: #27500A; }}
  .badge-conf-med  {{ background: #FAEEDA; color: #633806; }}
  .badge-conf-low  {{ background: #FCEBEB; color: #791F1F; }}

  .dist-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 0.85rem;
  }}
  .dist-label {{
    width: 130px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #444;
    font-size: 0.82rem;
  }}
  .dist-count {{
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #888;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
  }}

  /* Low-confidence file list */
  .conf-fname {{
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    color: #666;
    flex: 1;
  }}
  .conf-val-low {{
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    color: #A32D2D;
    margin-left: 10px;
  }}
  ul.conf-low-list {{
    list-style: none;
    font-size: 0.85rem;
  }}
  ul.conf-low-list li {{
    display: flex;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid #f5e8e8;
  }}
  ul.conf-low-list li:last-child {{ border-bottom: none; }}
  ul.conf-low-list li.empty {{ color: #3B6D11; font-family: 'Courier New', monospace; font-size: 0.8rem; }}

  .overflow-wrap {{ overflow-x: auto; }}
  .empty {{ color: #aaa; font-size: 0.85rem; padding: 0.5rem 0; }}
  .failed-list {{ list-style: none; font-family: 'Courier New', monospace; font-size: 0.8rem; color: #A32D2D; }}
  .failed-list li {{ padding: 3px 0; border-bottom: 1px solid #f5e8e8; }}
  .failed-list li:last-child {{ border-bottom: none; }}

  .receipts-search {{
    width: 100%;
    padding: 7px 12px;
    border: 1px solid #e8e6e0;
    border-radius: 6px;
    font-size: 0.85rem;
    font-family: 'Georgia', serif;
    background: #faf9f6;
    color: #1c1c1a;
    margin-bottom: 1rem;
    outline: none;
  }}
  .receipts-search:focus {{ border-color: #aaa; }}

  footer {{
    text-align: center;
    padding: 2rem;
    font-size: 0.75rem;
    color: #bbb;
    font-family: 'Courier New', monospace;
    border-top: 1px solid #e8e6e0;
    margin-top: 1rem;
  }}

  @media (max-width: 900px) {{
    .cards {{ grid-template-columns: repeat(3, 1fr); }}
    .timing {{ grid-template-columns: repeat(2, 1fr); }}
    .conf-cards {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <h1>Receipt OCR Report</h1>
    <p>generated {gen_time}</p>
  </div>
  <div class="header-right">
    {ps.get('total', 0)} images processed<br>
    {ps.get('success', 0)} succeeded &middot; {ps.get('failed', 0)} failed
  </div>
</header>

<div class="container">

  <div class="cards">
    {cards_html}
  </div>

  <div class="timing">
    {timing_html}
  </div>

  <!-- ── OCR Confidence ─────────────────────────────────────────────────── -->
  <div class="section">
    <div class="section-title">ocr confidence</div>
    {conf_summary_cards}
    <div class="grid-2" style="margin-bottom:0">
      <div>
        <div class="section-title" style="margin-bottom:0.6rem">confidence distribution</div>
        {conf_dist_html if conf_dist_html else '<p class="empty">No confidence data found</p>'}
      </div>
      <div>
        <div class="section-title" style="margin-bottom:0.6rem">low confidence receipts (&lt;80%)</div>
        <ul class="conf-low-list">{low_conf_list_html}</ul>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">confidence by receipt — worst first</div>
    <div class="overflow-wrap">
      <table>
        <thead>
          <tr>
            <th>file</th>
            <th style="text-align:right;padding-right:1rem">lines</th>
            <th>avg confidence</th>
            <th>3 lowest-confidence lines</th>
          </tr>
        </thead>
        <tbody>{per_receipt_rows}</tbody>
      </table>
    </div>
  </div>
  <!-- ───────────────────────────────────────────────────────────────────── -->

  <div class="section">
    <div class="section-title">spend by store — top 20</div>
    <div class="overflow-wrap">
      <table>
        <thead><tr><th>store</th><th style="text-align:right;padding-right:1rem">spend</th><th style="text-align:right;padding-right:0.75rem">%</th><th></th></tr></thead>
        <tbody>{store_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="section">
      <div class="section-title">monthly spend</div>
      <div class="overflow-wrap">
        <table>
          <thead><tr><th>month</th><th style="text-align:right;padding-right:1rem">spend</th><th></th></tr></thead>
          <tbody>{monthly_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <div class="section-title">top items by frequency</div>
      {top_items_html}
    </div>
  </div>

  <div class="grid-2">
    <div class="section">
      <div class="section-title">payment methods</div>
      {pm_rows}
    </div>
    <div class="section">
      <div class="section-title">currencies</div>
      {cur_rows}
    </div>
  </div>

  {failed_section}

  <div class="section">
    <div class="section-title">all receipts</div>
    <input class="receipts-search" type="text" id="search" placeholder="filter by store, date, file..." oninput="filterTable()" />
    <div class="overflow-wrap">
      <table id="receipts-table">
        <thead>
          <tr>
            <th>file</th>
            <th>store</th>
            <th>date</th>
            <th style="text-align:right;padding-right:1rem">total</th>
            <th style="text-align:right;padding-right:1rem">items</th>
            <th>status</th>
            <th>payment</th>
            <th>ocr conf</th>
          </tr>
        </thead>
        <tbody id="receipts-body">
          {receipt_rows}
        </tbody>
      </table>
    </div>
  </div>

</div>

<footer>
  receipt ocr pipeline &mdash; summary_report.py
</footer>

<script>
  function filterTable() {{
    var q = document.getElementById('search').value.toLowerCase();
    var rows = document.getElementById('receipts-body').getElementsByTagName('tr');
    for (var i = 0; i < rows.length; i++) {{
      var text = rows[i].textContent.toLowerCase();
      rows[i].style.display = text.includes(q) ? '' : 'none';
    }}
  }}
</script>

</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate financial summary and HTML report from pipeline summary.json"
    )
    parser.add_argument(
        "--results", default="./output/summary.json",
        help="Path to summary.json produced by pipeline.py (default: ./output/summary.json)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write outputs. Defaults to the same folder as --results."
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"File not found: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Support both the full summary.json ({"stats": ..., "receipts": [...]})
    # and a bare list of receipt records.
    if isinstance(raw, dict) and "receipts" in raw:
        receipts = raw["receipts"]
        stats = raw.get("stats", {})
    elif isinstance(raw, list):
        receipts = raw
        stats = {}
    else:
        raise SystemExit("Unrecognised format: expected summary.json from pipeline.py")

    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load confidence data from sibling confidence/ folder (graceful if absent)
    conf_data = load_confidence_data(results_path)
    if conf_data:
        print(f"Loaded confidence data for {len(conf_data)} receipt(s)")
    else:
        print("No confidence data found (confidence/ folder missing or empty — scores will show as n/a)")

    summary = build_summary(receipts, stats, conf_data)

    summary_path = output_dir / "financial_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary JSON  ->  {summary_path}")

    html = render_html(summary, receipts, conf_data)
    html_path = output_dir / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report   ->  {html_path}")
