"""Run with:  python -m pytest -q     (no API key needed)"""
import copy
import json
from datetime import date
from pathlib import Path

import pytest

from src.extract import DemoExtractor, normalise, parse_json, to_iso_date, to_number
from src.gstin import make_gstin, validate_gstin
from src.pipeline import process_batch
from src.validate import find_duplicates, overall_status, validate_invoice

ROOT = Path(__file__).resolve().parent.parent
SELLER = make_gstin("36", "AABCS1234K")    # Telangana
BUYER_TS = make_gstin("36", "AAFCN5678P")  # Telangana (same state)
BUYER_KA = make_gstin("29", "AAFCN5678P")  # Karnataka (other state)


def good_invoice(**overrides):
    inv = {"invoice_number": "SLT/25-26/101", "invoice_date": "2025-11-10",
           "seller_name": "Test Traders", "seller_gstin": SELLER,
           "buyer_name": "Buyer Co", "buyer_gstin": BUYER_TS, "place_of_supply": "Telangana",
           "line_items": [{"description": "Laptop", "hsn": "8471", "quantity": 2, "unit_price": 50000,
                           "taxable_value": 100000, "gst_rate": 18}],
           "taxable_total": 100000, "cgst": 9000, "sgst": 9000, "igst": 0, "grand_total": 118000}
    inv.update(overrides)
    return inv


def failed(inv):
    return {c["check"] for c in validate_invoice(inv, today=date(2026, 9, 1)) if c["status"] == "fail"}


# ---------- GSTIN ----------
def test_real_world_gstin_examples_pass_checksum():
    assert validate_gstin("27AAPFU0939F1ZV")[0] and validate_gstin("29AAGCB7383J1Z4")[0]


def test_gstin_typo_is_caught():
    typo = SELLER[:5] + ("X" if SELLER[5] != "X" else "Y") + SELLER[6:]
    ok, why = validate_gstin(typo)
    assert not ok and "Check digit" in why


def test_gstin_bad_format_and_state():
    assert not validate_gstin("36ABC")[0]
    assert not validate_gstin("88" + SELLER[2:])[0]


# ---------- validator ----------
def test_clean_invoice_passes():
    checks = validate_invoice(good_invoice(), today=date(2026, 9, 1))
    assert overall_status(checks) == "OK", checks


@pytest.mark.parametrize("change, expected", [
    ({"grand_total": 128000}, "Grand total"),
    ({"cgst": 0, "sgst": 0, "igst": 18000}, "Tax type"),                      # IGST inside one state
    ({"buyer_gstin": BUYER_KA, "place_of_supply": "Karnataka"}, "Tax type"),  # CGST+SGST across states
    ({"sgst": 9500, "grand_total": 118500}, "CGST = SGST"),
    ({"taxable_total": 90000, "grand_total": 108000}, "Taxable total"),
    ({"invoice_date": "2030-01-01"}, "Invoice date"),
    ({"cgst": 11000, "sgst": 11000, "grand_total": 122000}, "Tax amount"),
])
def test_validator_catches_errors(change, expected):
    assert expected in failed(good_invoice(**change))


def test_line_math_error():
    inv = good_invoice()
    inv["line_items"][0]["taxable_value"] = 110000
    assert "Line arithmetic" in failed(inv)


def test_gst2_slab_rule_depends_on_date():
    inv = good_invoice()
    inv["line_items"][0]["gst_rate"] = 12
    inv.update(cgst=6000, sgst=6000, grand_total=112000)
    before = copy.deepcopy(inv)
    before["invoice_date"] = "2025-08-01"
    warn = lambda i: {c["check"] for c in validate_invoice(i, today=date(2026, 9, 1)) if c["status"] == "warn"}
    assert "GST rate slab" in warn(inv)          # 12% no longer exists after 22 Sep 2025
    assert "GST rate slab" not in warn(before)   # but was valid before


def test_duplicates_detected():
    a = good_invoice()
    assert find_duplicates([a, good_invoice(invoice_number="X/1"), dict(a)]) == {
        2: "Duplicate of invoice #1 (same seller GSTIN and invoice number)"}


# ---------- extraction helpers ----------
def test_normalisation_of_indian_formats():
    assert to_number("Rs 1,23,456.50") == 123456.5 and to_number("18%") == 18
    assert to_iso_date("15-09-2025") == "2025-09-15" and to_iso_date("14 May 2026") == "2026-05-14"
    inv = normalise({"seller_gstin": "36 aabcs 1234k1zh", "grand_total": "1,18,000.00", "line_items": []})
    assert inv["seller_gstin"] == "36AABCS1234K1ZH" and inv["grand_total"] == 118000


def test_parse_json_from_messy_reply():
    assert parse_json('Here you go:\n```json\n{"a": 1}\n```')["a"] == 1


# ---------- pipeline, error injection and scoring (offline) ----------
def test_pipeline_with_saved_readings(tmp_path):
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps({"details": [{"file": "a.jpg", "ai_extracted": good_invoice()},
                                             {"file": "b.jpg", "ai_extracted": good_invoice()}]}))
    results = process_batch([("a.jpg", b""), ("b.jpg", b""), ("c.jpg", b"")], DemoExtractor([saved]))
    assert results[0].status == "OK"
    assert results[1].status == "FAIL"                 # same seller + number as a.jpg -> duplicate
    assert results[2].status == "ERROR"                # no saved reading for c.jpg


def test_error_injection_catches_every_planted_mistake():
    from eval.error_injection import run
    inv = normalise(good_invoice())
    result = run([("test", "x.jpg", inv)])
    caught, tested = result["planted_errors_caught"].split(" ")[0].split("/")
    assert caught == tested and int(tested) >= 9


def test_error_injection_skips_unfair_cases():
    from eval.error_injection import run
    inv = normalise(good_invoice(grand_total=999))     # grand total already wrong on the "real" document
    result = run([("test", "x.jpg", inv)])
    assert result["by_mistake"]["wrong grand total"]["caught"] == "0/0"
    assert result["by_mistake"]["wrong grand total"]["not_tested"] == {"check already flagging this document": 1}


def test_sroie_scoring():
    from eval.sroie import field_correct
    got = normalise({"seller_name": "OJC  Marketing Sdn. Bhd.", "invoice_date": "15/01/2019", "grand_total": "RM 193.00",
                     "seller_address": "NO 2 & 4, JALAN BAYU 4"})
    assert field_correct("company", "OJC MARKETING SDN BHD", got)
    assert field_correct("date", "15/01/2019", got) and field_correct("total", "193.00", got)
    assert field_correct("address", "NO 2 & 4 ,JALAN BAYU 4", got)
    assert not field_correct("total", "139.00", got)


# ---------- real-world rules learned from the 22 real invoices ----------
def test_indian_date_styles():
    assert to_iso_date("5th June 2019") == "2019-06-05"
    assert to_iso_date("04-JULY-2022") == "2022-07-04"
    assert to_iso_date("24-nov-2022") == "2022-11-24"


def test_gstin_ocr_repair_only_when_check_digit_confirms():
    from src.extract import repair_gstin
    assert repair_gstin("27AAACHI201R1ZN") == "27AAACH1201R1ZN"   # I read instead of 1 (seen on a real invoice)
    assert repair_gstin("23AM2359012D1ZW") == "23AM2359012D1ZW"   # genuinely invalid GSTIN is NOT 'fixed'


def test_service_lines_without_qty_are_not_flagged():
    inv = good_invoice()
    inv["line_items"][0].update(quantity=None, unit_price=None)
    assert "Line arithmetic" not in failed(inv)


def test_discount_is_review_not_fail():
    inv = good_invoice()
    inv["line_items"][0].update(quantity=1, unit_price=110000)   # taxable 100000 < 110000 -> discount
    checks = validate_invoice(inv, today=date(2026, 9, 1))
    assert "Line arithmetic" not in failed(inv)
    assert any(c["check"] == "Line arithmetic" and c["status"] == "warn" for c in checks)


def test_other_charges_count_towards_grand_total():
    inv = good_invoice(other_charges=1180, grand_total=119180)   # e.g. TCS / freight
    assert "Grand total" not in failed(inv)


def test_conflicting_place_of_supply_is_review():
    # seller & buyer in Telangana, but printed place of supply is Karnataka and IGST is charged
    inv = good_invoice(cgst=0, sgst=0, igst=18000, place_of_supply="Karnataka")
    checks = validate_invoice(inv, today=date(2026, 9, 1))
    assert any(c["check"] == "Tax type" and c["status"] == "warn" for c in checks)


# ---------- gaps found by the error-injection test on 383 real documents ----------
def test_small_line_errors_are_caught():
    # 1 x Rs 1.88 printed as Rs 1.50 used to pass because of the flat Rs 1 tolerance
    inv = good_invoice(taxable_total=1.5, cgst=0.14, sgst=0.14, grand_total=1.78)
    inv["line_items"] = [{"description": "pen", "quantity": 1, "unit_price": 2.0, "taxable_value": 2.4, "gst_rate": 18}]
    assert "Line arithmetic" in failed(inv)


def test_large_quantity_rounding_is_allowed():
    inv = good_invoice()
    inv["line_items"][0].update(quantity=1000, unit_price=100.003, taxable_value=100003)
    inv.update(taxable_total=100003, cgst=9000.27, sgst=9000.27, grand_total=118003.54)
    assert "Line arithmetic" not in failed(inv)


def test_cgst_sgst_equal_even_without_gstin():
    inv = good_invoice(seller_gstin=None, buyer_gstin=None, place_of_supply=None, sgst=9050, grand_total=118050)
    assert "CGST = SGST" in failed(inv)


def test_tax_checked_when_lines_include_a_return():
    inv = good_invoice(taxable_total=0, cgst=900, sgst=900, grand_total=1800)
    inv["line_items"] = [{"description": "item", "quantity": 1, "unit_price": 5000, "taxable_value": 5000, "gst_rate": 18},
                         {"description": "returned", "quantity": -1, "unit_price": 5000, "taxable_value": -5000,
                          "gst_rate": 18}]
    assert "Tax amount" in failed(inv)


def test_sroie_company_registration_number():
    from eval.sroie import company_without_reg_no
    assert company_without_reg_no("99 SPEED MART S/B", {"seller_name": "99 SPEED MART S/B (519537-X)"})
