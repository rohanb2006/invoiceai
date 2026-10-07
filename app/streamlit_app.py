"""
InvoiceAI web app.   Run:  streamlit run app/streamlit_app.py
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.extract import DemoExtractor, GeminiExtractor, normalise  # noqa: E402
from src.pipeline import line_items_table, process_batch, summary_table  # noqa: E402
from src.validate import overall_status, validate_invoice  # noqa: E402

st.set_page_config(page_title="InvoiceAI", page_icon="🧾", layout="wide")
# Real sample documents, downloaded by the evaluation scripts (not stored in git):
#   data/real_invoices/  - 22 real Indian GST invoices   (python -m eval.real_invoices fetch)
#   data/sroie/          - 361 real scanned receipts      (python -m eval.sroie fetch)
SAMPLE_DIRS = [ROOT / "data" / "real_invoices", ROOT / "data" / "sroie"]


def sample_files():
    return [p for d in SAMPLE_DIRS if d.exists() for p in sorted(d.glob("*.jpg"))]
ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
BADGE = {"OK": "🟢 OK", "REVIEW": "🟡 REVIEW", "FAIL": "🔴 FAIL", "ERROR": "⚫ ERROR"}


def get_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    key = st.text_input("Gemini API key", value=get_key(), type="password",
                        help="Free key: aistudio.google.com/apikey. Without a key, the app replays the "
                             "saved AI readings of the real sample documents (offline demo mode).")
    try:
        extractor = GeminiExtractor(key) if key else DemoExtractor()
    except Exception:
        extractor = DemoExtractor()
    st.caption(f"Reader: **{extractor.name}**")
    st.divider()
    st.markdown("**How it works**\n1. AI reads the invoice\n2. Code checks every number\n"
                "3. You review anything flagged")

st.title("🧾 InvoiceAI — GST invoice reader & auditor")
st.caption("Upload invoice photos or PDFs → AI extracts the data → automatic GST checks → export to Excel")
if isinstance(extractor, DemoExtractor):
    st.info("Offline demo mode: add a Gemini API key in the sidebar to read your own invoices. "
            "Without a key you can still try the real sample documents (it shows the AI readings saved "
            "during the last evaluation run).")

# ---------------- input ----------------
c1, c2 = st.columns([3, 1])
uploads = c1.file_uploader("Upload invoices (PNG, JPG, PDF)", type=["png", "jpg", "jpeg", "pdf", "webp"],
                           accept_multiple_files=True)
samples = sample_files()
indian = [p for p in samples if p.name.startswith("real_")]
use_samples = c2.button("📂 Try real sample documents", disabled=not samples,
                        help=None if samples else "Run 'python -m eval.real_invoices fetch' first")
n_samples = c2.slider("How many", 3, max(4, len(samples)), min(12, max(4, len(samples)))) \
    if samples and not uploads else None

files = None
if uploads and c1.button("🔍 Process uploaded invoices", type="primary"):
    files = [(u.name, u.getvalue()) for u in uploads]
if use_samples:
    chosen = (indian + [p for p in samples if p not in indian])[: n_samples or len(samples)]  # Indian ones first
    files = [(p.name, p.read_bytes()) for p in chosen]

if files:
    bar = st.progress(0.0, text="Reading invoices…")
    results = []
    for i, f in enumerate(files, 1):
        results += process_batch([f], extractor)
        bar.progress(i / len(files), text=f"Read {i}/{len(files)}: {f[0]}")
    # duplicates need the whole batch, so re-run that step on the full set
    from src.validate import find_duplicates
    ok = [r for r in results if r.data]
    for idx, msg in find_duplicates([r.data for r in ok]).items():
        ok[idx].checks.append({"check": "Duplicate", "status": "fail", "message": msg})
        ok[idx].status = "FAIL"
    bar.empty()
    st.session_state.results = results
    st.session_state.images = dict(files)

results = st.session_state.get("results")
if not results:
    st.stop()

# ---------------- dashboard ----------------
df = summary_table(results)
counts = df.status.value_counts()
gst_total = (df.cgst.fillna(0) + df.sgst.fillna(0) + df.igst.fillna(0))
at_risk = gst_total[df.status == "FAIL"].sum()
k = st.columns(5)
k[0].metric("Invoices", len(df))
k[1].metric("🟢 OK", int(counts.get("OK", 0)))
k[2].metric("🟡 Review", int(counts.get("REVIEW", 0)))
k[3].metric("🔴 Fail", int(counts.get("FAIL", 0)) + int(counts.get("ERROR", 0)))
k[4].metric("GST credit at risk", f"Rs {at_risk:,.0f}", help="GST on invoices that failed checks")

show = df.copy()
show["status"] = show.status.map(BADGE)
st.dataframe(show[["file", "status", "invoice_number", "invoice_date", "seller_name", "grand_total", "issues"]], hide_index=True)

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xl:
    df.to_excel(xl, sheet_name="Invoices", index=False)
    line_items_table(results).to_excel(xl, sheet_name="Line items", index=False)
d1, d2 = st.columns(2)
d1.download_button("⬇️ Download Excel (invoices + line items)", buf.getvalue(), "invoiceai_export.xlsx")
d2.download_button("⬇️ Download CSV", df.to_csv(index=False), "invoiceai_export.csv")

with st.expander("📊 GST by seller"):
    by_seller = df.assign(gst=gst_total).groupby("seller_name")["gst"].sum().sort_values(ascending=False)
    st.bar_chart(by_seller)

# ---------------- single invoice review ----------------
st.divider()
st.subheader("🔎 Review an invoice")
pick = st.selectbox("Invoice", [f"{r.filename} — {BADGE[r.status]}" for r in results])
r = results[[f"{x.filename} — {BADGE[x.status]}" for x in results].index(pick)]
left, right = st.columns([1, 1])
with left:
    raw = st.session_state.images.get(r.filename)
    if raw and not r.filename.lower().endswith(".pdf"):
        st.image(raw)
    else:
        st.caption("PDF preview not shown")
with right:
    if r.error:
        st.error(r.error)
        st.stop()
    st.markdown(f"**Status: {BADGE[r.status]}**")
    for c in r.checks:
        st.markdown(f"{ICON[c['status']]} **{c['check']}** — {c['message']}")

    st.markdown("**Extracted data** (edit any value the AI misread, then re-check)")
    header_keys = ["invoice_number", "invoice_date", "seller_name", "seller_address", "seller_gstin", "buyer_name", "buyer_gstin",
                   "place_of_supply", "taxable_total", "cgst", "sgst", "igst", "other_charges", "grand_total"]
    header = pd.DataFrame({"field": header_keys, "value": [str(r.data.get(k_) if r.data.get(k_) is not None
                                                               else "") for k_ in header_keys]})
    edited_header = st.data_editor(header, disabled=["field"], hide_index=True, key=f"h_{r.filename}")
    edited_items = st.data_editor(pd.DataFrame(r.data.get("line_items", [])), num_rows="dynamic",
                                  hide_index=True, key=f"l_{r.filename}")
    if st.button("🔁 Re-check with my corrections"):
        fixed = {f: (v if str(v).strip() not in ("", "None", "nan") else None)
                 for f, v in zip(edited_header.field, edited_header.value)}
        fixed["line_items"] = edited_items.to_dict("records")
        r.data = normalise(fixed)
        r.checks = validate_invoice(r.data)
        r.status = overall_status(r.checks)
        st.rerun()
