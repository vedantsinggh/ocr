"""
summary_report.py
-----------------
Reads _all_results.json and generates:
  - _financial_summary.json (structured)
  - report.html (visual dashboard)

Usage:
    python summary_report.py --results outputs/_all_results.json
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_amount(value_str) -> float:
    if not value_str:
        return 0.0
    nums = re.findall(r'\d+\.\d{2}', str(value_str))
    return float(nums[0]) if nums else 0.0


def build_summary(all_results: list[dict]) -> dict:
    total_spend = 0.0
    transactions = 0
    store_spend = defaultdict(float)
    monthly_spend = defaultdict(float)
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    failed = []
    items_all = []

    for r in all_results:
        if r["status"] != "ok" or not r["data"]:
            failed.append(r["file"])
            continue

        data = r["data"]
        amt = parse_amount(data.get("total_amount", {}).get("value"))
        conf = data.get("overall_confidence", 0)

        transactions += 1
        total_spend += amt

        store = data.get("store_name", {}).get("value") or "Unknown"
        store_spend[store] += amt

        date_val = data.get("date", {}).get("value")
        if date_val:
            try:
                month = datetime.strptime(date_val, "%Y-%m-%d").strftime("%Y-%m")
                monthly_spend[month] += amt
            except:
                pass

        if conf >= 0.8:
            confidence_dist["high"] += 1
        elif conf >= 0.6:
            confidence_dist["medium"] += 1
        else:
            confidence_dist["low"] += 1

        for item in data.get("items", []):
            items_all.append({
                "store": store,
                "name": item.get("name", ""),
                "price": parse_amount(item.get("price"))
            })

    return {
        "generated_at": datetime.now().isoformat(),
        "total_spend": round(total_spend, 2),
        "total_transactions": transactions,
        "failed_images": len(failed),
        "average_transaction_value": round(total_spend / transactions, 2) if transactions else 0,
        "spend_per_store": dict(sorted(store_spend.items(), key=lambda x: -x[1])),
        "spend_per_month": dict(sorted(monthly_spend.items())),
        "confidence_distribution": confidence_dist,
        "total_items_detected": len(items_all),
        "failed_files": failed,
    }


def render_html(summary: dict, all_results: list[dict]) -> str:
    store_rows = "".join(
        f"<tr><td>{store}</td><td>{amt:.2f}</td>"
        f"<td>{amt/summary['total_spend']*100:.1f}%</td></tr>"
        for store, amt in list(summary["spend_per_store"].items())[:15]
    ) or "<tr><td colspan='3'>No data</td></tr>"

    monthly_rows = "".join(
        f"<tr><td>{month}</td><td>{amt:.2f}</td></tr>"
        for month, amt in summary["spend_per_month"].items()
    ) or "<tr><td colspan='2'>No data</td></tr>"

    cd = summary["confidence_distribution"]
    total_conf = max(sum(cd.values()), 1)

    receipt_rows = ""
    for r in all_results[:50]:  # show first 50
        status = r["status"]
        badge = f'<span class="badge badge-{"ok" if status=="ok" else "err"}">{status}</span>'
        if r["data"]:
            d = r["data"]
            store = d.get("store_name", {}).get("value", "-") or "-"
            date = d.get("date", {}).get("value", "-") or "-"
            total = d.get("total_amount", {}).get("value", "-") or "-"
            conf = d.get("overall_confidence", 0)
            flags = ", ".join(d.get("flagged_fields", [])) or "None"
            conf_color = "#2ecc71" if conf >= 0.8 else "#f39c12" if conf >= 0.6 else "#e74c3c"
            receipt_rows += f"""
            <tr>
                <td>{r['file']}</td>
                <td>{store}</td>
                <td>{date}</td>
                <td>{total}</td>
                <td><span style="color:{conf_color};font-weight:bold">{conf:.2f}</span></td>
                <td style="font-size:0.8em;color:#e74c3c">{flags}</td>
                <td>{badge}</td>
            </tr>"""
        else:
            receipt_rows += f"<tr><td>{r['file']}</td><td colspan='5'>—</td><td>{badge}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Receipt OCR — Financial Summary</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #333; }}
  header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white;
           padding: 2rem; text-align: center; }}
  header h1 {{ font-size: 2rem; }} header p {{ opacity: 0.7; margin-top: 0.3rem; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: white; border-radius: 12px; padding: 1.5rem;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .card .val {{ font-size: 2rem; font-weight: 700; color: #1a1a2e; }}
  .card .lbl {{ font-size: 0.85rem; color: #888; margin-top: 0.3rem; }}
  .section {{ background: white; border-radius: 12px; padding: 1.5rem;
             box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #1a1a2e;
               border-bottom: 2px solid #f0f2f5; padding-bottom: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #f8f9fa; padding: 0.6rem 0.8rem; text-align: left;
       font-weight: 600; color: #555; }}
  td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #f0f2f5; }}
  tr:hover td {{ background: #fafbfc; }}
  .conf-bar {{ display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0; }}
  .bar-bg {{ flex: 1; height: 18px; background: #f0f2f5; border-radius: 9px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 9px; display: flex;
              align-items: center; padding: 0 6px; font-size: 0.75rem; color: white; }}
  .badge {{ padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }}
  .badge-ok {{ background: #d4edda; color: #155724; }}
  .badge-err {{ background: #f8d7da; color: #721c24; }}
  .overflow-table {{ overflow-x: auto; }}
</style>
</head>
<body>
<header>
  <h1>🧾 Receipt OCR Dashboard</h1>
  <p>Generated {summary['generated_at'][:19]}</p>
</header>
<div class="container">
  <div class="cards">
    <div class="card"><div class="val">{summary['total_transactions']}</div><div class="lbl">Transactions</div></div>
    <div class="card"><div class="val">{summary['total_spend']:.2f}</div><div class="lbl">Total Spend</div></div>
    <div class="card"><div class="val">{summary['average_transaction_value']:.2f}</div><div class="lbl">Avg Transaction</div></div>
    <div class="card"><div class="val">{len(summary['spend_per_store'])}</div><div class="lbl">Unique Stores</div></div>
    <div class="card"><div class="val">{summary['failed_images']}</div><div class="lbl">Failed / Unreadable</div></div>
    <div class="card"><div class="val">{summary['total_items_detected']}</div><div class="lbl">Items Detected</div></div>
  </div>

  <div class="section">
    <h2>🎯 OCR Confidence Distribution</h2>
    <div class="conf-bar"><span style="width:80px;font-size:0.85rem">High (≥0.8)</span>
      <div class="bar-bg"><div class="bar-fill" style="width:{cd['high']/total_conf*100:.0f}%;background:#2ecc71">
        {cd['high']}</div></div></div>
    <div class="conf-bar"><span style="width:80px;font-size:0.85rem">Medium</span>
      <div class="bar-bg"><div class="bar-fill" style="width:{cd['medium']/total_conf*100:.0f}%;background:#f39c12">
        {cd['medium']}</div></div></div>
    <div class="conf-bar"><span style="width:80px;font-size:0.85rem">Low (&lt;0.6)</span>
      <div class="bar-bg"><div class="bar-fill" style="width:{cd['low']/total_conf*100:.0f}%;background:#e74c3c">
        {cd['low']}</div></div></div>
  </div>

  <div class="section">
    <h2>🏪 Spend by Store (Top 15)</h2>
    <div class="overflow-table"><table>
      <tr><th>Store</th><th>Total Spend</th><th>% of Total</th></tr>
      {store_rows}
    </table></div>
  </div>

  <div class="section">
    <h2>📅 Monthly Spend</h2>
    <div class="overflow-table"><table>
      <tr><th>Month</th><th>Total Spend</th></tr>
      {monthly_rows}
    </table></div>
  </div>

  <div class="section">
    <h2>📋 Receipt Details (first 50)</h2>
    <div class="overflow-table"><table>
      <tr><th>File</th><th>Store</th><th>Date</th><th>Total</th>
          <th>Confidence</th><th>Flagged Fields</th><th>Status</th></tr>
      {receipt_rows}
    </table></div>
  </div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='outputs/_all_results.json')
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()

    with open(args.results, 'r') as f:
        all_results = json.load(f)

    summary = build_summary(all_results)

    summary_path = Path(args.output_dir) / '_financial_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary JSON: {summary_path}")

    html = render_html(summary, all_results)
    html_path = Path(args.output_dir) / 'report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ HTML Report : {html_path}")
