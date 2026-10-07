"""
Invoice extraction.

GeminiExtractor  sends the invoice image/PDF to Google Gemini (vision) and asks for a strict
                 JSON structure. Free API key: https://aistudio.google.com/apikey
DemoExtractor    offline mode: replays saved Gemini readings of the real sample documents,
                 so the app can be demonstrated without an API key.

Whatever the AI returns is then NORMALISED (numbers, dates, GSTINs) before validation.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path


MODEL_CANDIDATES = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-flash-latest", "gemini-flash-lite-latest",
                                "gemini-3.5-flash", "gemini-2.5-flash"] if m]
RETRYABLE = ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand",
             "DEADLINE", "timed out", "INTERNAL", "10053", "10054", "aborted", "reset by peer",
             "Connection", "EOF occurred", "RemoteProtocolError")
NOT_FOUND = ("404", "not found", "NOT_FOUND", "is not supported")


def call_gemini(call, preferred: str | None, attempts_per_model: int = 3):
    """Run call(model_name) with resilience:
    - busy / rate-limited (503, 429): wait 2s, then 4s, and retry, then try the next model
    - model not found (404): move straight to the next model name
    - anything else (e.g. invalid API key): stop and report it
    Returns (result, model_that_worked)."""
    order = ([preferred] if preferred else []) + [m for m in MODEL_CANDIDATES if m != preferred]
    last = None
    for model in order:
        for attempt in range(attempts_per_model):
            try:
                return call(model), model
            except Exception as e:
                last, msg = e, str(e)
                if any(s in msg for s in NOT_FOUND):
                    break                                   # try the next model
                if any(s.lower() in msg.lower() for s in RETRYABLE):
                    if attempt < attempts_per_model - 1:
                        time.sleep(2 ** (attempt + 1))      # 2s, 4s
                    continue
                raise                                       # e.g. bad API key: don't hide it
    raise RuntimeError(f"All Gemini models are busy or unavailable right now. Last error: {last}")

PROMPT = """You are an expert Indian GST invoice data-entry assistant.
The document is a tax invoice or a shop receipt. Read it carefully and return ONLY a JSON object with exactly these keys:

{
  "invoice_number": string,
  "invoice_date": string (YYYY-MM-DD),
  "seller_name": string, "seller_address": string or null, "seller_gstin": string (15 chars, no spaces),
  "buyer_name": string, "buyer_gstin": string or null,
  "place_of_supply": string (state name) or null,
  "line_items": [ {"description": string, "hsn": string, "quantity": number,
                   "unit_price": number, "taxable_value": number, "gst_rate": number (percent, e.g. 18)} ],
  "taxable_total": number, "cgst": number, "sgst": number, "igst": number,
  "other_charges": number (cess + TCS + freight + non-taxable fees + round-off, i.e. anything else added to
                   reach the grand total; negative if something is deducted; 0 if none),
  "grand_total": number
}

Rules:
- Copy values EXACTLY as printed. Do NOT correct, recalculate or "fix" any number, even if it looks wrong.
- Numbers: plain numbers without commas or currency symbols (Rs 1,23,456.50 -> 123456.5).
- Use 0 for a tax that is not charged. Use null for anything not present on the invoice.
- Labels vary: "Assessable Value"/"Sub Total"/"Taxable Value" = taxable_total;
  "Total Invoice Value"/"Amount Payable"/"Grand Total" = grand_total; "Bill No"/"Inv #" = invoice_number.
- seller_name / seller_address: the business name and address exactly as printed at the top (keep suffixes
  like PVT LTD or SDN BHD; join address lines with ", ").
- invoice_number is the TAX INVOICE number. Do NOT use the PO number, order number, customer number, PNR or
  reference number, even if they look similar.
- invoice_date is the date of the invoice itself - NOT the order date, PO date, due date or delivery date.
- Invoices often show TWO GSTINs: the seller's (at the top / "From") and the buyer's (in "Bill To" / "Buyer").
  Fill both. Read GSTINs character by character: digits and letters look alike (0/O, 1/I, 5/S, 8/B).
- place_of_supply: copy exactly what is printed next to "Place of Supply" (it may differ from the buyer's state).
- If a line has no quantity or rate printed (common for services), use null for them, not 0.
Return the JSON only."""


class ExtractionError(RuntimeError):
    pass


def mime_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
            ".pdf": "application/pdf"}.get(ext, "application/octet-stream")


class GeminiExtractor:
    name = "gemini-vision"

    def __init__(self, api_key: str | None = None):
        from google import genai  # pip install google-genai
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ExtractionError("No Gemini API key found. Set GEMINI_API_KEY.")
        self.client = genai.Client(api_key=key)
        self.model = None

    def extract(self, data: bytes, filename: str) -> dict:
        from google.genai import types
        parts = [types.Part.from_bytes(data=data, mime_type=mime_type(filename)), PROMPT]
        config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
        try:
            resp, self.model = call_gemini(
                lambda m: self.client.models.generate_content(model=m, contents=parts, config=config),
                self.model)
        except Exception as e:
            raise ExtractionError(f"Gemini call failed: {e}") from e
        return normalise(parse_json(resp.text or ""))


class DemoExtractor:
    """Offline mode: shows the SAVED Gemini readings of the real sample documents (from the last
    evaluation run: eval/real_results.json and eval/sroie_results.json), so the app can be demonstrated
    without an API key. It cannot read new documents."""
    name = "demo (saved AI readings)"

    def __init__(self, result_files: list[Path] | None = None):
        root = Path(__file__).resolve().parent.parent / "eval"
        self.saved = {}
        for path in result_files or [root / "real_results.json", root / "sroie_results.json"]:
            if path.exists():
                for d in json.loads(path.read_text(encoding="utf-8")).get("details", []):
                    if d.get("ai_extracted"):
                        self.saved[d["file"]] = d["ai_extracted"]

    def extract(self, data: bytes, filename: str) -> dict:
        name = Path(filename).name
        if name not in self.saved:
            raise ExtractionError("Offline demo mode only has saved readings for the real sample documents "
                                  "(run the evaluations first). Add a Gemini API key to read new invoices.")
        return normalise(json.loads(json.dumps(self.saved[name])))


def get_extractor(api_key: str | None = None):
    try:
        return GeminiExtractor(api_key)
    except Exception:
        return DemoExtractor()


# ------------------------------------------------------------------ normalisation
def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ExtractionError("The AI did not return JSON.")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Invalid JSON from AI: {e}") from e


def to_number(x) -> float | None:
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"(?i)(rs\.?|inr|rm|myr|usd|\$|₹|%|\s)", "", str(x)).replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y",
                "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%d/%b/%Y", "%d-%b-%y",
                "%b %d, %Y", "%B %d, %Y", "%d %b, %Y", "%d %B, %Y")


def to_iso_date(x) -> str | None:
    """Understands Indian invoice date styles: 25-09-2020, 25/09/2020, 5th June 2019, 04-JULY-2022, 24-nov-2022."""
    if not x:
        return None
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", str(x).strip(), flags=re.I)   # 5th -> 5
    s = re.sub(r"\s+", " ", s)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s   # leave as-is; the validator will flag it


# Positions in a GSTIN that must be digits vs letters (see src/gstin.py for the layout).
_DIGIT_POS = {0, 1, 7, 8, 9, 10}
_LETTER_POS = {2, 3, 4, 5, 6, 11}
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


def repair_gstin(g: str | None) -> str | None:
    """Fix classic OCR look-alikes (I<->1, O<->0, S<->5, B<->8) in positions where the GSTIN format
    requires a digit or a letter. The repair is ONLY accepted if the mod-36 check digit then validates,
    so a real typo on the invoice is never 'fixed' into a different valid-looking number."""
    from .gstin import validate_gstin
    if not g or len(g) != 15 or validate_gstin(g)[0]:
        return g
    chars = list(g)
    for i in _DIGIT_POS:
        chars[i] = _TO_DIGIT.get(chars[i], chars[i])
    for i in _LETTER_POS:
        chars[i] = _TO_LETTER.get(chars[i], chars[i])
    candidate = "".join(chars)
    return candidate if validate_gstin(candidate)[0] else g


def normalise(inv: dict) -> dict:
    out = dict(inv)
    for k in ("taxable_total", "cgst", "sgst", "igst", "grand_total", "other_charges"):
        out[k] = to_number(inv.get(k)) or 0.0
    out["total_tax"] = round(out["cgst"] + out["sgst"] + out["igst"], 2)
    out["invoice_date"] = to_iso_date(inv.get("invoice_date"))
    out["gstin_corrections"] = list(inv.get("gstin_corrections") or [])   # keep repairs from an earlier pass
    for k in ("seller_gstin", "buyer_gstin"):
        raw = re.sub(r"[\s\-]", "", str(inv.get(k))).upper() if inv.get(k) else None
        fixed = repair_gstin(raw)
        if raw and fixed != raw:
            out["gstin_corrections"].append(f"{k}: {raw} -> {fixed} (OCR look-alike characters, check digit verified)")
        out[k] = fixed
    out["invoice_number"] = str(inv.get("invoice_number") or "").strip() or None
    for k in ("seller_name", "seller_address", "buyer_name"):
        out[k] = " ".join(str(inv.get(k)).split()) if inv.get(k) else None
    items = []
    for it in inv.get("line_items") or []:
        items.append({
            "description": str(it.get("description") or "").strip(),
            "hsn": re.sub(r"\D", "", str(it.get("hsn") or "")) or None,
            "quantity": to_number(it.get("quantity")),        # None = not printed on the invoice
            "unit_price": to_number(it.get("unit_price")),
            "taxable_value": to_number(it.get("taxable_value")) or 0.0,
            "gst_rate": to_number(it.get("gst_rate")) or 0.0,
        })
    out["line_items"] = items
    return out
