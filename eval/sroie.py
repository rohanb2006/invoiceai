"""
Benchmark on SROIE - 361 REAL scanned receipts with human-labelled fields.

Data: ICDAR 2019 "Scanned Receipts OCR and Information Extraction" (SROIE) test set, from the Hugging Face
copy "jsdnrs/ICDAR2019-SROIE" (licence: CC-BY-4.0). https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE
These are real shop receipts from Malaysia (2017-2018). Each one has 4 labelled fields:
    company, address, date, total
It's a well-known public benchmark, so the numbers can be compared with other work.

InvoiceAI uses exactly the same prompt as for Indian GST invoices - nothing is tuned for receipts.
Field mapping: company -> seller_name, address -> seller_address, date -> invoice_date, total -> grand_total

Scoring (strict, like the SROIE competition): a field counts as correct only if it matches exactly after
normalising case, spaces and punctuation (dates compared as calendar dates, totals as numbers).

Usage:
    python -m eval.sroie fetch                  # download the 361 test receipts (needs internet)
    python -m eval.sroie evaluate               # run Gemini on them (needs GEMINI_API_KEY). RESUMABLE:
                                                # if you stop it or hit the daily limit, run it again later
                                                # and it continues where it stopped
    python -m eval.sroie evaluate --limit 100   # only do the next 100 (to split it over days)
    python -m eval.sroie rescore                # re-score saved AI output with the current code (no API)
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from src.extract import GeminiExtractor, normalise, to_iso_date

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sroie"
LABELS = OUT / "labels.json"
RESULTS = ROOT / "eval" / "sroie_results.json"
API = ("https://datasets-server.huggingface.co/rows?dataset=jsdnrs/ICDAR2019-SROIE"
       "&config=default&split=test&offset={offset}&length=100")
FIELDS = {"company": "seller_name", "address": "seller_address", "date": "invoice_date", "total": "grand_total"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "InvoiceAI-eval"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, offset = [], 0
    while True:
        page = json.loads(_get(API.format(offset=offset)))
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            name = f"sroie_{row['key']}.jpg"
            if not (OUT / name).exists():
                (OUT / name).write_bytes(_get(row["image"]["src"]))
            labels.append({"file": name, "label": row["entities"]})
        offset += len(rows)
        print(f"downloaded {offset}/{page.get('num_rows_total', '?')}")
    LABELS.write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(labels)} real receipts + labels to {OUT}")


# ---------------------------------------------------------------- scoring
def _text(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _amount(s):
    s = re.sub(r"[^0-9.\-]", "", str(s if s is not None else ""))
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def field_correct(field: str, label, got: dict) -> bool:
    value = got.get(FIELDS[field])
    if field == "date":
        return bool(value) and to_iso_date(label) == value
    if field == "total":
        return _amount(label) is not None and _amount(label) == _amount(value)
    return bool(_text(label)) and _text(label) == _text(value)


REG_NO = re.compile(r"\(\s*[0-9A-Z]{3,}[\-\s]?[A-Z0-9]?\s*\)")   # e.g. "(519537-X)" company registration number


def company_without_reg_no(label, got: dict) -> bool:
    """Secondary check: SROIE labels leave out the company registration number printed after the name,
    e.g. label "99 SPEED MART S/B" vs printed "99 SPEED MART S/B (519537-X)"."""
    return bool(_text(label)) and _text(label) == _text(REG_NO.sub("", str(got.get("seller_name") or "").upper()))


def score(records: list[dict], extracted: dict) -> dict:
    per_field = {f: [0, 0] for f in FIELDS}
    company_relaxed = 0
    rows = []
    for rec in records:
        got = extracted.get(rec["file"])
        if got is None:
            continue
        mismatches, right = [], 0
        for f in FIELDS:
            label = rec["label"].get(f)
            if label in (None, ""):
                continue
            ok = field_correct(f, label, got)
            per_field[f][0] += ok
            per_field[f][1] += 1
            right += ok
            if f == "company":
                company_relaxed += ok or company_without_reg_no(label, got)
            if not ok:
                mismatches.append({"field": f, "label": label, "ai_read": got.get(FIELDS[f])})
        rows.append({"file": rec["file"], "correct": right, "mismatches": mismatches, "ai_extracted": got})
    total_right = sum(a for a, _ in per_field.values())
    total = sum(n for _, n in per_field.values())
    perfect = sum(1 for r in rows if not r["mismatches"])
    return {"summary": {
        "dataset": "SROIE 2019 test set (real scanned receipts)",
        "receipts_processed": f"{len(rows)}/{len(records)}",
        "field_accuracy": f"{total_right / max(total, 1):.1%} ({total_right}/{total})",
        "by_field": {f: f"{a}/{n} ({a / max(n, 1):.1%})" for f, (a, n) in per_field.items()},
        "receipts_with_all_4_fields_right": f"{perfect}/{len(rows)}",
        "company_ignoring_registration_number": f"{company_relaxed}/{per_field['company'][1]} "
                                                f"({company_relaxed / max(per_field['company'][1], 1):.1%})",
    }, "details": rows}


def _load_previous() -> dict:
    if not RESULTS.exists():
        return {}
    prev = json.loads(RESULTS.read_text(encoding="utf-8"))
    return {d["file"]: d["ai_extracted"] for d in prev.get("details", [])}


def _save(records, extracted, model):
    result = score(records, extracted)
    result["summary"] = {"model": model, **result["summary"]}
    RESULTS.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def evaluate(limit: int | None = None):
    if not LABELS.exists():
        sys.exit("Run 'python -m eval.sroie fetch' first.")
    records = json.loads(LABELS.read_text(encoding="utf-8"))
    extracted = _load_previous()
    todo = [r for r in records if r["file"] not in extracted][:limit]
    print(f"{len(extracted)} receipts already done, {len(todo)} to do now")
    ex = GeminiExtractor()
    errors_in_a_row = 0
    for i, rec in enumerate(todo, 1):
        try:
            extracted[rec["file"]] = ex.extract((OUT / rec["file"]).read_bytes(), rec["file"])
            errors_in_a_row = 0
        except Exception as e:
            errors_in_a_row += 1
            print(f"[{i}/{len(todo)}] {rec['file']}: ERROR {str(e)[:150]}")
            if errors_in_a_row >= 3:
                print("\n3 errors in a row - probably the free daily limit. Progress is saved; "
                      "run the same command again later and it will continue.")
                break
            time.sleep(10)
            continue
        one = score([rec], {rec["file"]: extracted[rec["file"]]})["details"][0]
        print(f"[{i}/{len(todo)}] {rec['file']}: {one['correct']}/4 fields correct")
        if i % 10 == 0:
            _save(records, extracted, ex.model)      # save progress regularly
        time.sleep(4)                                # stay under the free-tier requests-per-minute limit
    result = _save(records, extracted, ex.model or "?")
    print("\n" + json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print("Saved eval/sroie_results.json")


def rescore():
    records = json.loads(LABELS.read_text(encoding="utf-8"))
    prev = json.loads(RESULTS.read_text(encoding="utf-8"))
    extracted = {d["file"]: normalise(d["ai_extracted"]) for d in prev["details"]}
    result = _save(records, extracted, prev["summary"].get("model", "?"))
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "evaluate":
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        evaluate(lim)
    else:
        {"fetch": fetch, "rescore": rescore}.get(cmd, lambda: sys.exit(__doc__))()
