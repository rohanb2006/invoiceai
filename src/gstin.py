"""
GSTIN (GST Identification Number) validation.

A GSTIN has 15 characters, e.g. 36AABCS1234K1ZH:
  36          state code (36 = Telangana)
  AABCS1234K  PAN of the business
  1           entity number (1-9, A-Z)
  Z           fixed letter
  H           CHECK DIGIT computed from the first 14 characters (mod-36 algorithm)

The check digit means a typo or a made-up GSTIN is detected almost every time,
without calling any government API.
"""
from __future__ import annotations

import re

STATE_CODES = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra and Nagar Haveli and Daman and Diu", "27": "Maharashtra", "29": "Karnataka",
    "30": "Goa", "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman and Nicobar Islands", "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
    "97": "Other Territory", "99": "Centre Jurisdiction",
}
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def check_digit(first14: str) -> str:
    """Mod-36 checksum used by the GST network."""
    total = 0
    for i, ch in enumerate(first14):
        product = CHARSET.index(ch) * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return CHARSET[(36 - total % 36) % 36]


def clean(gstin: str | None) -> str:
    return re.sub(r"[\s\-]", "", (gstin or "")).upper()


def validate_gstin(gstin: str | None) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    g = clean(gstin)
    if not g:
        return False, "GSTIN missing"
    if len(g) != 15:
        return False, f"GSTIN must have 15 characters (found {len(g)})"
    if not PATTERN.match(g):
        return False, "GSTIN format is invalid"
    if g[:2] not in STATE_CODES:
        return False, f"Unknown state code {g[:2]}"
    expected = check_digit(g[:14])
    if g[14] != expected:
        return False, f"Check digit mismatch (expected {expected}, found {g[14]}) - likely a typo or fake GSTIN"
    return True, f"Valid ({STATE_CODES[g[:2]]})"


def state_of(gstin: str | None) -> str | None:
    g = clean(gstin)
    return STATE_CODES.get(g[:2]) if len(g) >= 2 else None


def make_gstin(state_code: str, pan: str, entity: str = "1") -> str:
    """Build a GSTIN with a correct check digit (used to create test invoices)."""
    body = f"{state_code}{pan}{entity}Z"
    return body + check_digit(body)


def state_from_text(text) -> str | None:
    """Map a printed 'Place of Supply' to a state name: 'MAHARASHTRA', 'Maharashtra (27)', '27-Maharashtra', '27'."""
    if text is None:
        return None
    t = str(text).strip()
    m = re.search(r"\b(\d{2})\b", t)
    if m and m.group(1) in STATE_CODES:
        return STATE_CODES[m.group(1)]
    low = t.lower()
    for name in sorted(STATE_CODES.values(), key=len, reverse=True):   # longest first ("Andhra Pradesh" before "Pradesh")
        if name.lower() in low:
            return name
    return None
