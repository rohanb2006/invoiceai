"""
Rule-based invoice validation.

The AI *reads* the invoice; this module *checks* it with plain, deterministic code.
Nothing here uses AI, so every check is explainable and 100% repeatable.

Each check returns: {"check", "status": pass|warn|fail, "message"}
Overall status: FAIL if any check fails, REVIEW if any warning, else OK.
"""
from __future__ import annotations

from datetime import date

from .gstin import state_from_text, state_of, validate_gstin

TOLERANCE = 1.0   # rupees; allows normal rounding on printed invoices

# GST 2.0 (effective 22 Sep 2025) replaced the 12% and 28% slabs with 5%, 18% and 40%.
GST2_DATE = date(2025, 9, 22)
SLABS_BEFORE = {0, 0.1, 0.25, 1.5, 3, 5, 12, 18, 28}
SLABS_AFTER = {0, 0.1, 0.25, 1.5, 3, 5, 18, 40}

REQUIRED = ["invoice_number", "invoice_date", "seller_name", "seller_gstin", "grand_total"]


def _num(x) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def _close(a, b, tol=TOLERANCE) -> bool:
    return abs(_num(a) - _num(b)) <= tol


def _rs(x) -> str:
    return f"Rs {_num(x):,.2f}"


def validate_invoice(inv: dict, today: date | None = None) -> list[dict]:
    today = today or date.today()
    out = []
    add = lambda check, status, msg: out.append({"check": check, "status": status, "message": msg})
    items = inv.get("line_items") or []

    # 1. Required fields
    missing = [f for f in REQUIRED if not inv.get(f)]
    add("Required fields", "fail" if missing else "pass",
        f"Missing: {', '.join(missing)}" if missing else "All key fields present")

    # 2-3. GSTINs
    ok, why = validate_gstin(inv.get("seller_gstin"))
    add("Seller GSTIN", "pass" if ok else "fail", why)
    if inv.get("buyer_gstin"):
        ok, why = validate_gstin(inv.get("buyer_gstin"))
        add("Buyer GSTIN", "pass" if ok else "fail", why)
    else:
        add("Buyer GSTIN", "warn", "No buyer GSTIN (B2C invoice) - input tax credit cannot be claimed")

    # 4. Date
    inv_date = None
    try:
        inv_date = date.fromisoformat(str(inv.get("invoice_date")))
        if inv_date > today:
            add("Invoice date", "fail", f"Date {inv_date} is in the future")
        else:
            add("Invoice date", "pass", str(inv_date))
    except ValueError:
        add("Invoice date", "fail", f"Unreadable date: {inv.get('invoice_date')!r}")

    # 5. Line arithmetic: quantity x unit price = taxable value.
    #    Real invoices often print no quantity/rate for services -> those lines are skipped.
    #    A taxable value LOWER than qty x rate is usually a discount -> REVIEW, not FAIL.
    priced = [(i, it) for i, it in enumerate(items, 1) if it.get("quantity") and it.get("unit_price")]
    over, under = [], []
    for i, it in priced:
        expected = _num(it["quantity"]) * _num(it["unit_price"])
        actual = _num(it.get("taxable_value"))
        # A rate printed to the paisa can be off by at most half a paisa per unit, so the allowed gap grows
        # with the quantity. (A flat Rs 1 was too loose: on small items it hid errors of 20% or more.)
        if _close(expected, actual, max(0.05, 0.005 * abs(_num(it["quantity"])))):
            continue
        msg = f"line {i}: {_num(it['quantity']):g} x {_rs(it['unit_price'])} = {_rs(expected)}, taxable {_rs(actual)}"
        (under if actual < expected else over).append(msg)
    if not items:
        add("Line items", "warn", "No line items found")
    elif not priced:
        add("Line arithmetic", "pass", "No quantity/rate printed (service invoice) - skipped")
    elif over:
        add("Line arithmetic", "fail", "; ".join(over))
    elif under:
        add("Line arithmetic", "warn", "Taxable value below qty x rate (discount?): " + "; ".join(under))
    else:
        add("Line arithmetic", "pass", f"{len(priced)} line(s) add up")

    # 6. Lines add up to the taxable total
    line_sum = sum(_num(it.get("taxable_value")) for it in items)
    # some pages list items without amounts (e.g. a separate annexure). Lines can be negative (returns).
    itemised = any(_num(it.get("taxable_value")) for it in items)
    if items and not itemised:
        add("Taxable total", "warn", "Line items carry no amounts - cannot cross-check the taxable total")
    elif items:
        add("Taxable total", "pass" if _close(line_sum, inv.get("taxable_total")) else "fail",
            f"Lines sum to {_rs(line_sum)}; invoice says {_rs(inv.get('taxable_total'))}")

    # 7. GST slab valid for the invoice date
    slabs = SLABS_AFTER if inv_date and inv_date >= GST2_DATE else SLABS_BEFORE
    rates = {_num(it.get("gst_rate")) for it in items if _num(it.get("taxable_value")) > 0}
    wrong = sorted(r for r in rates if r not in slabs)
    if rates:
        era = "GST 2.0 (from 22 Sep 2025)" if slabs is SLABS_AFTER else "pre-GST 2.0"
        add("GST rate slab", "warn" if wrong else "pass",
            f"Rate(s) {wrong}% not valid under {era} slabs" if wrong else f"Rates {sorted(rates)}% valid ({era})")

    # 8. Intra-state -> CGST + SGST ; inter-state -> IGST
    seller_state = state_of(inv.get("seller_gstin"))
    cgst, sgst, igst = _num(inv.get("cgst")), _num(inv.get("sgst")), _num(inv.get("igst"))
    # Two possible sources for the place of supply: what is PRINTED on the invoice, and the buyer's GSTIN state.
    # For most goods/services they agree; for some services (e.g. work on a building) the printed place of
    # supply is the site's state. If they disagree, the tax type is flagged for REVIEW instead of FAIL.
    printed = state_from_text(inv.get("place_of_supply"))
    buyer_state = state_of(inv.get("buyer_gstin"))
    candidates = [s_ for s_ in dict.fromkeys([printed, buyer_state]) if s_]

    def consistent(state):
        intra = seller_state.lower() == state.lower()
        return (intra and igst == 0) or (not intra and cgst == 0 and sgst == 0)

    if seller_state and candidates:
        ok_with = [c for c in candidates if consistent(c)]
        kind = "CGST+SGST" if igst == 0 else "IGST"
        if len(ok_with) == len(candidates):
            intra = seller_state.lower() == candidates[0].lower()
            add("Tax type", "pass", ("Intra-state (CGST+SGST)" if intra else "Inter-state (IGST)") +
                f": {seller_state} -> {candidates[0]}")
        elif ok_with:
            add("Tax type", "warn", f"{kind} matches place of supply {ok_with[0]}, but not "
                f"{[c for c in candidates if c not in ok_with][0]} - verify which place of supply applies")
        else:
            should = "CGST+SGST" if seller_state.lower() == candidates[0].lower() else "IGST"
            add("Tax type", "fail", f"{seller_state} -> {candidates[0]}: should charge {should}, invoice charges {kind}")
    else:
        add("Tax type", "warn", "Could not determine place of supply")

    # CGST and SGST are always two equal halves - this holds even when the states can't be worked out
    if (cgst or sgst) and not _close(cgst, sgst):
        add("CGST = SGST", "fail", f"CGST {_rs(cgst)} and SGST {_rs(sgst)} should be equal")

    # 9. Tax amount = sum(taxable value x rate)
    expected_tax = sum(_num(it.get("taxable_value")) * _num(it.get("gst_rate")) / 100 for it in items)
    charged_tax = cgst + sgst + igst
    if items and itemised:
        add("Tax amount", "pass" if _close(expected_tax, charged_tax, max(TOLERANCE, 0.005 * expected_tax)) else "fail",
            f"Expected {_rs(expected_tax)} from line rates; charged {_rs(charged_tax)}")

    # 10. Grand total = taxable total + GST + other charges (cess, TCS, freight, fees, round-off)
    other = _num(inv.get("other_charges"))
    expected_total = _num(inv.get("taxable_total")) + charged_tax + other
    extra = f" + other charges {_rs(other)}" if other else ""
    add("Grand total", "pass" if _close(expected_total, inv.get("grand_total")) else "fail",
        f"{_rs(inv.get('taxable_total'))} + tax {_rs(charged_tax)}{extra} = {_rs(expected_total)}; "
        f"invoice says {_rs(inv.get('grand_total'))}")
    for note in inv.get("gstin_corrections") or []:
        add("GSTIN auto-repair", "warn", note + " - please confirm")
    return out


def overall_status(checks: list[dict]) -> str:
    statuses = {c["status"] for c in checks}
    return "FAIL" if "fail" in statuses else "REVIEW" if "warn" in statuses else "OK"


def find_duplicates(invoices: list[dict]) -> dict[int, str]:
    """Same seller GSTIN + same invoice number = possible duplicate (double payment risk)."""
    seen, dupes = {}, {}
    for i, inv in enumerate(invoices):
        key = (str(inv.get("seller_gstin", "")).upper(), str(inv.get("invoice_number", "")).strip().upper())
        if not all(key):
            continue
        if key in seen:
            dupes[i] = f"Duplicate of invoice #{seen[key] + 1} (same seller GSTIN and invoice number)"
        else:
            seen[key] = i
    return dupes
