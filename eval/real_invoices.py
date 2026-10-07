"""
Test InvoiceAI on REAL Indian GST invoices.

Data: the public Hugging Face dataset "AjitRawat/invoice" (22 real Indian GST invoice images with
hand-made labels). https://huggingface.co/datasets/AjitRawat/invoice
The images are downloaded to data/real_invoices/ for local testing only and are NOT committed to git
(no licence is stated on the dataset page, so we don't redistribute them).

The labels are not complete for every invoice, so each field is only scored when a label exists:
    invoice number, invoice date, GSTIN, CGST, SGST, IGST, grand total

Label sanity checks (they look ONLY at the label, never at the AI's answer):
    - a labelled GSTIN that fails the GSTIN format / check-digit rules is not scored
    - a labelled grand total below Rs 10, or smaller than another amount in the same label, is not scored

Human-verified corrections: eval/label_corrections.json lets a person who LOOKED AT THE IMAGE fix a wrong
label, with a note saying what is printed. Both scores are reported: original labels and corrected labels.

Usage:
    python -m eval.real_invoices fetch          # download images + labels (needs internet)
    python -m eval.real_invoices evaluate       # run Gemini on them (needs GEMINI_API_KEY)
    python -m eval.real_invoices rescore        # re-score the last run with the current code (no API calls)
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from src.extract import GeminiExtractor, normalise, to_iso_date, to_number
from src.gstin import validate_gstin
from src.validate import overall_status, validate_invoice

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "real_invoices"
LABELS = OUT / "labels.json"
RESULTS = ROOT / "eval" / "real_results.json"
CORRECTIONS = ROOT / "eval" / "label_corrections.json"   # human-verified fixes to dataset labels
API = ("https://datasets-server.huggingface.co/rows?dataset=AjitRawat/invoice"
       "&config=default&split=train&offset={offset}&length=100")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "InvoiceAI-eval"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, offset = [], 0
    while True:
        rows = json.loads(_get(API.format(offset=offset))).get("rows", [])
        if not rows:
            break
        for r in rows:
            idx, row = r["row_idx"], r["row"]
            name = f"real_{idx:02d}.jpg"
            (OUT / name).write_bytes(_get(row["image"]["src"]))
            gt = row["ground_truth"]
            gt = json.loads(gt) if isinstance(gt, str) else gt
            labels.append({"file": name, "label": gt.get("gt_parse", gt)})
            print(f"downloaded {name}")
        offset += len(rows)
    LABELS.write_text(json.dumps(labels, indent=1, ensure_ascii=False))
    print(f"\nSaved {len(labels)} real invoices + labels to {OUT}")


# ---------------------------------------------------------------- scoring helpers
def label_fields(label: dict) -> dict:
    """Map the dataset's label keys onto InvoiceAI's field names (only fields that are present)."""
    h = {k.lower(): v for k, v in (label.get("header") or {}).items()}
    h.update({k.lower(): v for k, v in (label.get("summary") or {}).items()})
    pick = lambda *keys: next((h[k] for k in keys if h.get(k) not in (None, "")), None)
    out = {
        "invoice_number": pick("invoice_no", "invoice_number", "inv_no", "bill_no"),
        "invoice_date": pick("invoice_date", "date"),
        "gstin": pick("gstin_no", "gstin", "seller_gstin"),
        "cgst": pick("cgst"), "sgst": pick("sgst"), "igst": pick("igst"),
        "grand_total": pick("grand_total", "total", "total_amount"),
    }
    return {k: v for k, v in out.items() if v is not None}


def label_problems(fields: dict, label: dict) -> list[tuple[str, str]]:
    """Reasons NOT to score some labelled fields. Looks only at the label itself."""
    probs = []
    if "gstin" in fields and not validate_gstin(fields["gstin"])[0]:
        probs.append(("gstin", f"labelled GSTIN {fields['gstin']!r} is itself invalid"))
    if "grand_total" in fields:
        gt = to_number(fields["grand_total"]) or 0
        amounts = [to_number(v) for k, v in (label.get("header") or {}).items()
                   if k.lower() in ("amount", "taxable_value", "total_value", "total_price") and to_number(v)]
        if gt < 10:
            probs.append(("grand_total", f"labelled grand total {fields['grand_total']!r} is not a plausible total"))
        elif amounts and gt < max(amounts):
            probs.append(("grand_total", f"labelled grand total {fields['grand_total']!r} is smaller than another "
                                         f"amount in the same label ({max(amounts):g})"))
    return probs


def field_correct(field: str, label, got: dict) -> bool:
    if field == "gstin":   # the label doesn't say whose GSTIN it is -> accept seller or buyer
        clean = re.sub(r"\s", "", str(label)).upper()
        return clean in {str(got.get("seller_gstin") or "").upper(), str(got.get("buyer_gstin") or "").upper()}
    if field == "invoice_date":
        return to_iso_date(label) == got.get("invoice_date")
    if field == "invoice_number":
        norm = lambda s: re.sub(r"[\s\-/]", "", str(s)).upper()
        return norm(label) == norm(got.get("invoice_number") or "")
    lv, gv = to_number(label), to_number(got.get(field))
    return lv is not None and gv is not None and abs(lv - gv) <= 1.0


def load_corrections() -> dict:
    if not CORRECTIONS.exists():
        return {}
    data = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def score(records: list[dict], extracted: dict, corrections: dict | None = None) -> dict:
    """Compare AI output with labels. `extracted` maps file name -> normalised AI output.
    `corrections` = {file: {field: verified_value, "note": ...}} from a human who checked the image."""
    per_field, skipped, statuses, rows, applied = {}, [], {}, [], []
    for rec in records:
        got = extracted.get(rec["file"])
        if got is None:
            continue
        fields = label_fields(rec["label"])
        for f, v in (corrections or {}).get(rec["file"], {}).items():
            if f != "note" and v not in (None, ""):
                applied.append(f"{rec['file']} {f}: {fields.get(f)!r} -> {v!r}")
                fields[f] = v
        for f, why in label_problems(fields, rec["label"]):
            skipped.append(f"{rec['file']}: {why} - not scored")
            fields.pop(f, None)
        checks = validate_invoice(got)
        status = overall_status(checks)
        statuses[status] = statuses.get(status, 0) + 1
        right, mismatches = [], []
        for f, v in fields.items():
            ok = field_correct(f, v, got)
            per_field.setdefault(f, [0, 0])
            per_field[f][0] += ok
            per_field[f][1] += 1
            right.append(ok)
            if not ok:
                shown = {k: got.get(k) for k in ("seller_gstin", "buyer_gstin")} if f == "gstin" else got.get(f)
                mismatches.append({"field": f, "label": v, "ai_read": shown})
        rows.append({"file": rec["file"], "fields_scored": len(right), "correct": sum(right), "status": status,
                     "mismatches": mismatches, "failed_checks": [c for c in checks if c["status"] != "pass"],
                     "ai_extracted": got})
    total_right = sum(v[0] for v in per_field.values())
    total = sum(v[1] for v in per_field.values())
    return {"summary": {
        "invoices_processed": len(rows),
        "labelled_field_accuracy": f"{total_right / max(total, 1):.1%} ({total_right}/{total})",
        "by_field": {f: f"{a}/{n}" for f, (a, n) in sorted(per_field.items())},
        "validator_status_on_real_invoices": statuses,
        "labels_skipped_by_sanity_checks": skipped,
        **({"human_verified_corrections": applied} if corrections else {}),
    }, "details": rows}


def _save(result: dict, model: str, records: list[dict] | None = None, extracted: dict | None = None):
    result["summary"] = {"model": model, **result["summary"]}
    corr = load_corrections()
    if corr and records is not None:
        verified = score(records, extracted, corr)["summary"]
        result["summary"]["with_human_verified_labels"] = {
            k: verified[k] for k in ("labelled_field_accuracy", "by_field", "human_verified_corrections")}
    print("\n" + json.dumps(result["summary"], indent=2, ensure_ascii=False))
    RESULTS.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved eval/real_results.json")


def evaluate():
    if not LABELS.exists():
        sys.exit("Run 'python -m eval.real_invoices fetch' first.")
    records = json.loads(LABELS.read_text())
    ex = GeminiExtractor()
    extracted = {}
    for i, rec in enumerate(records, 1):
        got, err = None, None
        for _ in range(2):                             # one retry for network hiccups
            try:
                got = ex.extract((OUT / rec["file"]).read_bytes(), rec["file"])
                break
            except Exception as e:
                err = e
                time.sleep(10)
        if got is None:
            print(f"[{i:2}/{len(records)}] {rec['file']}: ERROR {err}")
            continue
        extracted[rec["file"]] = got
        one = score([rec], {rec["file"]: got})["details"][0]
        print(f"[{i:2}/{len(records)}] {rec['file']}: {one['correct']}/{one['fields_scored']} labelled fields "
              f"correct | validator: {one['status']}")
        time.sleep(3)
    _save(score(records, extracted), ex.model, records, extracted)


def rescore():
    """Re-score the previous run's AI outputs with the current scoring + validation code (no API calls)."""
    prev = json.loads(RESULTS.read_text())
    records = json.loads(LABELS.read_text())
    extracted = {d["file"]: normalise(d["ai_extracted"]) for d in prev["details"]}
    model = prev["summary"].get("model", "?").replace(" (rescored)", "")
    _save(score(records, extracted), model + " (rescored)", records, extracted)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"fetch": fetch, "evaluate": evaluate, "rescore": rescore}.get(cmd, lambda: sys.exit(__doc__))()
