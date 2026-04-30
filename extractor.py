"""
extractor.py
------------
Item extractor for Donut's CORD v2 menu output.

Responsibility: given the raw `menu` list from Donut, return only
genuine purchasable line items with cleaned prices.

Store name and date are handled separately by postprocessor.py using
Tesseract OCR — we do NOT attempt to classify those here.

NOTE: This module is a thin wrapper kept for backward compatibility.
      The heavy lifting (validation, price cleaning) lives in postprocessor.py.
      You can call postprocessor.validate_items() directly if you prefer.
"""

from postprocessor import validate_items, clean_price


def extract_items(donut_output: dict) -> list:
    """
    Extract and validate menu items from a raw Donut CORD prediction dict.

    Parameters
    ----------
    donut_output : dict
        Raw output from DonutModel.inference()["predictions"][0]

    Returns
    -------
    list of dicts:
        [{"name": str, "price": str, "quantity": str, "confidence": float}, ...]

    Items are filtered to remove:
      - Store name / address / phone number lines
      - Footer / thank-you lines
      - Summary / total / tax lines
      - Lines without a valid numeric price
    Prices are cleaned of currency symbols and stray characters.
    """
    menu = donut_output.get("menu", [])
    return validate_items(menu)


def extract_total(donut_output: dict) -> dict:
    """
    Extract the total amount from Donut's total block.

    Returns {"value": str|None, "confidence": float}
    """
    total_block = donut_output.get("total", {})
    sub_block   = donut_output.get("sub_total", {})

    raw = (total_block.get("total_price")
           or total_block.get("total_etc")
           or sub_block.get("subtotal_price")
           or "")

    price = clean_price(raw)
    return {
        "value":      price,
        "confidence": 0.96 if price else 0.0,
        "_source":    "donut.total",
    }
