"""
Error-injection test for the checker - on REAL documents.

How do you test an invoice checker? You need invoices where you KNOW there is a mistake. Real invoices
don't come with that information, so I take the AI's reading of each real document (22 Indian GST
invoices + the SROIE receipts), plant ONE known mistake into a copy of it, and check whether the checker
now flags the right check.

Only fair cases are counted: a mistake is only planted if the check that should catch it was NOT already
flagging that document before the change (otherwise we couldn't tell what caused the flag), and if the
document actually has the data the mistake needs (e.g. no "CGST = SGST" mistake on an IGST invoice).
Everything that was skipped is counted and reported with the reason.

The same real documents WITHOUT any planted mistake are the false-alarm test: see
"validator_status_on_real_invoices" in eval/real_results.json.

Usage (after running eval.real_invoices evaluate and/or eval.sroie evaluate - no API calls here):
    python -m eval.error_injection
"""
import copy
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from src.extract import normalise
from src.gstin import check_digit, validate_gstin
from src.validate import find_duplicates, validate_invoice

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"Indian GST invoices": ROOT / "eval" / "real_results.json",
           "SROIE receipts": ROOT / "eval" / "sroie_results.json"}
OUT = ROOT / "eval" / "error_injection_results.json"
TODAY = date.today()


def flagged(inv: dict) -> set[str]:
    return {c["check"] for c in validate_invoice(inv, today=TODAY) if c["status"] in ("fail", "warn")}


def _num(x) -> float:
    return float(x or 0)


# Each mutation: (name, check that should catch it, function(inv) -> mutated copy, or a reason string
# if the document doesn't have the data this mistake needs)
def m_grand_total(inv):
    if not _num(inv.get("grand_total")):
        return "no grand total"
    inv["grand_total"] = round(_num(inv["grand_total"]) * 1.1 + 100, 2)
    return inv


def m_small_total(inv):
    if not _num(inv.get("grand_total")):
        return "no grand total"
    inv["grand_total"] = round(_num(inv["grand_total"]) + 5, 2)   # a small Rs 5 slip, not an obvious one
    return inv


def m_gstin_typo(inv):
    g = inv.get("seller_gstin")
    if not g or not validate_gstin(g)[0]:
        return "no valid seller GSTIN to change"
    for c in "XYZWQ":
        typo = g[:5] + c + g[6:]                       # change one letter of the PAN part
        if c != g[5] and check_digit(typo[:14]) != typo[14]:
            inv["seller_gstin"] = typo
            return inv
    return "could not make a typo"


def m_wrong_tax_type(inv):
    c, s, i = _num(inv.get("cgst")), _num(inv.get("sgst")), _num(inv.get("igst"))
    if c + s > 0 and i == 0:
        inv.update(igst=round(c + s, 2), cgst=0.0, sgst=0.0)
    elif i > 0 and c + s == 0:
        inv.update(cgst=round(i / 2, 2), sgst=round(i / 2, 2), igst=0.0)
    else:
        return "no CGST/SGST or IGST charged"
    return inv


def m_cgst_ne_sgst(inv):
    c, s = _num(inv.get("cgst")), _num(inv.get("sgst"))
    if not (c > 0 and s > 0):
        return "not a CGST+SGST invoice"
    inv["sgst"] = round(s + 50, 2)
    inv["grand_total"] = round(_num(inv.get("grand_total")) + 50, 2)
    return inv


def m_taxable_total(inv):
    if not any(_num(it.get("taxable_value")) for it in inv.get("line_items") or []):
        return "line items have no amounts"
    cut = round(_num(inv.get("taxable_total")) * 0.1 + 100, 2)
    inv["taxable_total"] = round(_num(inv.get("taxable_total")) - cut, 2)
    inv["grand_total"] = round(_num(inv.get("grand_total")) - cut, 2)   # keep the grand total consistent
    return inv


def m_future_date(inv):
    if not inv.get("invoice_date"):
        return "no invoice date"
    inv["invoice_date"] = (TODAY + timedelta(days=30)).isoformat()
    return inv


def m_tax_amount(inv):
    tax = _num(inv.get("cgst")) + _num(inv.get("sgst")) + _num(inv.get("igst"))
    if not tax or not any(_num(it.get("taxable_value")) for it in inv.get("line_items") or []):
        return "no tax or no line amounts"
    extra = round(tax * 0.2 + 20, 2)
    if _num(inv.get("igst")):
        inv["igst"] = round(_num(inv["igst"]) + extra, 2)
    else:
        inv["cgst"] = round(_num(inv.get("cgst")) + extra / 2, 2)
        inv["sgst"] = round(_num(inv.get("sgst")) + extra / 2, 2)
    inv["grand_total"] = round(_num(inv.get("grand_total")) + extra, 2)
    return inv


def m_line_math(inv):
    for it in inv.get("line_items") or []:
        if it.get("quantity") and it.get("unit_price"):
            it["unit_price"] = round(_num(it["unit_price"]) * 0.8, 2)   # amount now > qty x rate
            return inv
    return "no line with quantity and rate"


def m_bad_rate(inv):
    for it in inv.get("line_items") or []:
        if _num(it.get("taxable_value")) > 0:
            it["gst_rate"] = 13.0                      # not a GST slab in any year
            return inv
    return "no line with an amount"


def m_missing_number(inv):
    if not inv.get("invoice_number"):
        return "no invoice number"
    inv["invoice_number"] = None
    return inv


MUTATIONS = [
    ("wrong grand total", "Grand total", m_grand_total),
    ("grand total off by Rs 5", "Grand total", m_small_total),
    ("seller GSTIN typo", "Seller GSTIN", m_gstin_typo),
    ("wrong tax type", "Tax type", m_wrong_tax_type),
    ("CGST != SGST", "CGST = SGST", m_cgst_ne_sgst),
    ("taxable total wrong", "Taxable total", m_taxable_total),
    ("future date", "Invoice date", m_future_date),
    ("tax amount inflated", "Tax amount", m_tax_amount),
    ("line qty x rate wrong", "Line arithmetic", m_line_math),
    ("invalid GST rate", "GST rate slab", m_bad_rate),
    ("invoice number missing", "Required fields", m_missing_number),
]


def load_documents() -> list[tuple[str, str, dict]]:
    docs = []
    for source, path in SOURCES.items():
        if path.exists():
            for d in json.loads(path.read_text(encoding="utf-8")).get("details", []):
                if d.get("ai_extracted"):
                    docs.append((source, d["file"], normalise(d["ai_extracted"])))
    return docs


def run(docs: list[tuple[str, str, dict]]) -> dict:
    per_mut = {name: {"caught": 0, "tested": 0, "missed": [], "skipped": Counter()} for name, _, _ in MUTATIONS}
    per_source = {}
    for source, file, inv in docs:
        before = flagged(inv)
        src = per_source.setdefault(source, [0, 0])
        for name, check, fn in MUTATIONS:
            if check in before:
                per_mut[name]["skipped"]["check already flagging this document"] += 1
                continue
            mutated = fn(copy.deepcopy(inv))
            if isinstance(mutated, str):
                per_mut[name]["skipped"][mutated] += 1
                continue
            hit = check in flagged(mutated)
            per_mut[name]["tested"] += 1
            per_mut[name]["caught"] += hit
            src[0] += hit
            src[1] += 1
            if not hit:
                per_mut[name]["missed"].append(file)

    # duplicate test: add a second copy of each document to the batch
    dup = {"caught": 0, "tested": 0, "skipped": Counter()}
    for source, file, inv in docs:
        if not (inv.get("seller_gstin") and inv.get("invoice_number")):
            dup["skipped"]["no seller GSTIN or invoice number"] += 1
            continue
        hit = 1 in find_duplicates([inv, copy.deepcopy(inv)])
        dup["tested"] += 1
        dup["caught"] += hit
        per_source[source][0] += hit
        per_source[source][1] += 1

    caught = sum(v["caught"] for v in per_mut.values()) + dup["caught"]
    tested = sum(v["tested"] for v in per_mut.values()) + dup["tested"]
    return {
        "documents": dict(Counter(s for s, _, _ in docs)),
        "planted_errors_caught": f"{caught}/{tested} ({caught / max(tested, 1):.1%})",
        "by_source": {s: f"{a}/{n}" for s, (a, n) in per_source.items()},
        "by_mistake": {
            **{name: {"caught": f"{v['caught']}/{v['tested']}", "missed_on": v["missed"],
                      "not_tested": dict(v["skipped"])} for name, v in per_mut.items()},
            "duplicate invoice": {"caught": f"{dup['caught']}/{dup['tested']}", "not_tested": dict(dup["skipped"])},
        },
    }


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit("No saved AI readings found. Run 'python -m eval.real_invoices evaluate' "
                         "and/or 'python -m eval.sroie evaluate' first.")
    result = run(docs)
    print(json.dumps({k: v for k, v in result.items() if k != "by_mistake"}, indent=2))
    for name, v in result["by_mistake"].items():
        print(f"  {name:<24} caught {v['caught']:<9} not tested: {v['not_tested']}")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSaved eval/error_injection_results.json")


if __name__ == "__main__":
    main()
