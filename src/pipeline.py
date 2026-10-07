"""Process a batch of invoices: extract -> normalise -> validate -> duplicates -> tables for export."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .validate import find_duplicates, overall_status, validate_invoice


@dataclass
class InvoiceResult:
    filename: str
    data: dict | None = None
    checks: list[dict] = field(default_factory=list)
    status: str = "ERROR"          # OK | REVIEW | FAIL | ERROR
    error: str | None = None
    seconds: float = 0.0


def process_batch(files: list[tuple[str, bytes]], extractor) -> list[InvoiceResult]:
    results = []
    for name, data in files:
        t0 = time.time()
        r = InvoiceResult(filename=name)
        try:
            r.data = extractor.extract(data, name)
            r.checks = validate_invoice(r.data)
            r.status = overall_status(r.checks)
        except Exception as e:
            r.error = str(e)
        r.seconds = round(time.time() - t0, 2)
        results.append(r)

    ok = [r for r in results if r.data]
    for idx, msg in find_duplicates([r.data for r in ok]).items():
        ok[idx].checks.append({"check": "Duplicate", "status": "fail", "message": msg})
        ok[idx].status = "FAIL"
    return results


def summary_table(results: list[InvoiceResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = r.data or {}
        problems = [c["message"] for c in r.checks if c["status"] == "fail"]
        warnings = [c["message"] for c in r.checks if c["status"] == "warn"]
        rows.append({
            "file": r.filename, "status": r.status,
            "invoice_number": d.get("invoice_number"), "invoice_date": d.get("invoice_date"),
            "seller_name": d.get("seller_name"), "seller_gstin": d.get("seller_gstin"),
            "buyer_name": d.get("buyer_name"), "buyer_gstin": d.get("buyer_gstin"),
            "taxable_total": d.get("taxable_total"), "cgst": d.get("cgst"), "sgst": d.get("sgst"),
            "igst": d.get("igst"), "other_charges": d.get("other_charges"), "grand_total": d.get("grand_total"),
            "issues": " | ".join(problems + warnings) or (r.error or ""),
        })
    return pd.DataFrame(rows)


def line_items_table(results: list[InvoiceResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        if not r.data:
            continue
        for it in r.data.get("line_items", []):
            rows.append({"file": r.filename, "invoice_number": r.data.get("invoice_number"),
                         "seller_name": r.data.get("seller_name"), **it})
    return pd.DataFrame(rows)
