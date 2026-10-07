# InvoiceAI - Reading and checking GST invoices with AI

This is my second project for my 5th semester (B.E. CSE - AI & ML). You upload a photo or PDF of an Indian
GST invoice, Google Gemini reads it and turns it into structured data (seller, GSTIN, items, taxes, total),
and then my own code checks every number and GST rule before anyone trusts it.

## Why I built this

Small businesses in India get a lot of supplier invoices and someone usually types them into Tally or Excel
by hand. It's slow, and mistakes are expensive - a wrong GSTIN or the wrong tax type (IGST vs CGST+SGST)
can mean losing input tax credit, and duplicate invoices can mean paying twice.

The main idea of the project is: **the AI reads, plain code checks.** LLMs are good at reading messy
documents but they can misread numbers, so nothing the AI extracts is trusted until it passes the checks.
Anything suspicious goes to a person to review.

## What the checks do

- **GSTIN check** - format, valid state code, and the check digit (the last character of a GSTIN is
  calculated from the first 14 using a mod-36 formula, so typos and fake GSTINs can be caught offline)
- **GSTIN auto-repair** - if the AI mixes up look-alike characters (I/1, O/0, S/5, B/8), the code tries
  swapping them in positions where the format needs a digit or a letter. The repair is only accepted if the
  check digit then passes, and it's still shown to the user to confirm.
- **Line maths** - quantity x rate = amount (skipped for service lines that don't print a quantity; a lower
  amount is treated as a possible discount and sent for review instead of failing)
- **Totals** - lines add up to the taxable total; taxable + GST + other charges (cess, TCS, freight,
  round-off) = grand total
- **Tax type** - same state should be CGST+SGST, different states should be IGST. If the printed place of
  supply and the buyer's state disagree, it's marked for review because the rule depends on the kind of service.
- **GST rate** - checks the rate is a valid slab for the invoice date (GST 2.0 changed the slabs to
  5% / 18% / 40% from 22 Sep 2025)
- **Tax amount, CGST = SGST, dates, required fields, duplicate invoices**

Every invoice ends up as OK, REVIEW or FAIL with the reasons listed. In the app you can edit any value the AI
got wrong and re-run the checks, and export everything to Excel.

## Testing - only real documents

Everything is tested on real documents. There is no generated or fake invoice data in the evaluation (the unit tests use one hand-written
example invoice to test each rule, like any unit test).
(My first version tested on 30 invoices I generated myself - I removed that, because a checker that only
ever saw my own neat invoices told me nothing about real ones. See "What I learned".)

| Test | Real data used | What it measures |
|---|---|---|
| 1. Reading accuracy - receipts | 361 scanned receipts (SROIE 2019 benchmark) | Did the AI read company, address, date, total correctly? |
| 2. Reading accuracy - GST invoices | 22 real Indian GST invoices | Did the AI read invoice no., date, GSTIN, CGST/SGST/IGST, total? |
| 3. Checker - planted mistakes | the same real documents, one mistake planted at a time | Does the checker catch the mistake? |
| 4. Checker - false alarms | the 22 real GST invoices as they are | How many real invoices get flagged, and why? |

### 1. SROIE - 361 real scanned receipts
[SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) is the test set of the ICDAR 2019
"Scanned Receipts OCR and Information Extraction" competition (CC-BY-4.0). They are real shop receipts
from Malaysia - faded thermal paper, crumpled, different layouts - and every one has 4 labelled fields.
I used exactly the same prompt as for Indian invoices, nothing tuned for receipts, and strict scoring
(a field only counts if it matches exactly, ignoring case, spaces and punctuation).

Result (gemini-flash-lite-latest): **89.8% of fields correct (1297/1444)** on all 361 receipts.

| Field | Correct |
|---|---|
| Total | 357/361 (98.9%) |
| Date | 341/361 (94.5%) |
| Address | 306/361 (84.8%) |
| Company | 293/361 (81.2%) |

236/361 receipts had all 4 fields exactly right. The numbers that matter most for money (total, date) are
the most accurate.

I looked at all 68 company-name "mistakes" to see what was going on:
- **51** were the AI copying the company registration number printed after the name, e.g. label
  "99 SPEED MART S/B" vs AI "99 SPEED MART S/B (519537-X)". The SROIE labels leave that number out.
  Ignoring it, company accuracy is **90.6% (327/361)** - I report both, the strict one is the main number.
- **4** were 1-3 characters different - and some of these are typos **in the labels**
  ("INTERANTIONAL", "TED HENG" where the AI read "TEO HENG").
- **13** were a genuinely different name, e.g. the AI took a brand name ("Burger King") instead of the
  company that runs the shop, or the AI picked a different line on a cluttered receipt.

Receipts are not GST invoices, so they are only used to measure reading accuracy (and for the
arithmetic checks in test 3) - the GST checks don't apply to them.

### 2. Real Indian GST invoices
A public dataset of 22 real GST invoices with labels from Hugging Face
([AjitRawat/invoice](https://huggingface.co/datasets/AjitRawat/invoice)). The images are not included in
this repo because the dataset has no licence listed - `python -m eval.real_invoices fetch` downloads them.

**First run (gemini-flash-lite-latest): 72.6% of labelled fields correct, and the checker marked 18 of 22
real invoices as FAIL.** That was very different from what I expected, so I went through every invoice one
by one. What I found:

- **Real invoices are messier than I assumed.** Service invoices don't print a quantity, prices include
  discounts, and totals include things like compensation cess, TCS and airport fees. My checks assumed a
  perfect invoice, so I changed them (see the list above).
- **The place of supply isn't always the buyer's state.** One invoice was for interior work at a site in
  Odisha for a Mumbai company, so IGST was correct even though both GSTINs were from Maharashtra.
- **Some "errors" were in the dataset labels, not the AI.** 6 labelled GSTINs fail the check digit (for
  example one is 14 characters long), one "grand total" label was Rs 4, and another was actually the tax
  amount. The evaluation now skips labels that fail these sanity checks - they only look at the label,
  never at the AI's answer.
- **My scoring didn't understand some date formats** like "5th June 2019" or "04-JULY-2022".
- **Some were real AI mistakes** - it picked the PO number instead of the invoice number once, misread a
  "1" as "I" inside a GSTIN (now caught by the auto-repair), and missed the buyer's GSTIN on one invoice.
  I made the prompt more specific about these.

Re-scoring the same AI output with the fixed code gives **80.0% (48/60)**. Tax amounts (CGST, SGST, IGST)
were **100% correct** on every invoice where they were labelled.

For the remaining disagreements I can't tell who is right without looking at the image, so there's a
`eval/label_corrections.json` file where a person who checked the image writes what is actually printed.
The script reports the score with the original labels and with the human-checked labels separately.

Latest full run with the improved prompt: **78.3% (47/60)** - one field fewer than the 80.0% rescore. The
differences are on dates where the AI and the dataset still disagree (several of the ExpoHub invoices), so
the prompt change didn't really move the number either way. Tax amounts are still **100%** (16/16).

### 3. Checker - mistakes planted into real documents
To test a checker you need invoices where you *know* there's a mistake, and real invoices don't come with
that. So `eval/error_injection.py` takes the AI's reading of each real document, plants ONE known mistake
into a copy (wrong grand total, a Rs 5 slip, GSTIN typo, IGST instead of CGST+SGST, CGST != SGST, wrong
taxable total, inflated tax, qty x rate wrong, invalid GST rate, future date, missing invoice number,
duplicate invoice) and checks that the right check flags it.

To keep it fair, a mistake is only planted if the check that should catch it wasn't already flagging that
document, and if the document has the data the mistake needs (you can't plant "CGST != SGST" on an IGST
invoice). Skipped cases are counted and listed with the reason.

**First result: 2,342 / 2,451 caught (95.6%).** All 168 on the Indian invoices were caught, but 109
were missed on the receipts - and every miss was a real weakness in my checker:

| Missed | Why | Fix |
|---|---|---|
| qty x rate wrong: 206/302 | I allowed a flat Rs 1 difference. Receipt items cost 1-5, so a 20% error on a 1.88 item is only 0.38 and slipped through. | The allowed gap is now half a paisa per unit (the most a rate rounded to paise can be off), minimum 5 paise. |
| CGST != SGST: 7/19 | This check only ran when the states could be worked out from GSTINs. | CGST and SGST are always equal halves, so it now always runs. |
| tax amount inflated: 131/132 | A receipt with a returned item had lines adding up to zero, so the tax check was skipped. | Negative lines (returns) now count. |

**After the fixes: 2,410 / 2,410 caught (100%)**, and the 22 real GST invoices got exactly the same
OK/REVIEW/FAIL results as before - so the stricter rules didn't add false alarms there. (The number of
tests went down from 2,451 because the stricter rules flag more receipts already, and a mistake isn't
planted where the check is already flagging.)

On the receipts the stricter line rule does flag 42 more receipts, mostly as REVIEW. I looked at them: the
AI read the price *including* 6% tax but the line amount *excluding* it (6.36 vs 6.00), so the numbers on
the extracted data really don't match and a person should look. That's what REVIEW is for.

Honestly, the 100% itself isn't the interesting part - the checks are plain code, so once they apply they
catch the mistake. The useful part was finding where they *didn't* apply, and the "not tested" list, e.g.
qty x rate could only be tested on about half of the real GST invoices because the rest are service invoices
that don't print a quantity.

### 4. False alarms on real invoices
On the 22 real GST invoices as they are, the checker gives: **2 OK / 3 REVIEW / 17 FAIL**.
I went through all 17 FAILs:
- **10 are real problems with the document itself**: 4 ExpoHub invoices have no GSTIN at all (not valid GST
  tax invoices), 3 invoices have GSTINs with an impossible format, and 3 images are only a continuation page
  of a multi-page invoice (no invoice number or total on that page).
- **5 are the AI misreading something**: GSTINs read with a character missing (14 instead of 15), a missed
  invoice number and date, and tax rates read as 0%. This is exactly what the checker is for - a person
  looks, fixes the value in the app and re-checks.
- **1 is a limitation of my checker**: the place of supply is printed as a city ("Bhubaneshwar"), and my code
  only understands state names, so it couldn't see that IGST was correct (the work site was in Odisha).
- **1 I can't explain without the image**: tax is Rs 192 where 5% of the amount would be Rs 190.

So apart from the city name, FAIL didn't mean the checker was wrong - in a real accounts team, these all needed a human to look.

## How to run it

You need Python 3.10+ and a free Gemini API key from https://aistudio.google.com/apikey

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                       (31 tests, no API key needed)
python -m eval.real_invoices fetch        (downloads the 22 real GST invoices)
python -m eval.sroie fetch                (downloads the 361 real receipts)
set GEMINI_API_KEY=your_key_here
python -m eval.real_invoices evaluate     (test 2)
python -m eval.sroie evaluate             (test 1 - takes about 45 min, and it can be stopped and resumed)
python -m eval.error_injection            (test 3 - no API calls)
streamlit run app/streamlit_app.py        (paste the key in the sidebar, or try the real samples)
```
`rescore` re-scores a saved run with the current code without calling the API
(`python -m eval.real_invoices rescore`, `python -m eval.sroie rescore`).

## Folder structure

```
invoiceai/
  app/streamlit_app.py      - web app (upload, dashboard, review/edit, Excel export)
  src/extract.py            - Gemini prompt, JSON parsing, cleaning numbers/dates, GSTIN auto-repair
  src/validate.py           - all the checks
  src/gstin.py              - GSTIN format, state codes, check digit
  src/pipeline.py           - processes a batch of invoices and builds the export tables
  eval/sroie.py             - SROIE receipt benchmark (fetch / evaluate / rescore)
  eval/real_invoices.py     - real GST invoice test (fetch / evaluate / rescore)
  eval/error_injection.py   - plants mistakes into the real documents to test the checker
  eval/label_corrections.json - human-checked label fixes
  tests/                    - 31 unit tests
```

## Problems I ran into

- Gemini sometimes returned 503 "high demand" or the connection dropped - added retries and backup models.
- The free Gemini tier has a daily limit, so the 361-receipt run had to be resumable (it saves progress
  and continues where it stopped).
- The checker was way too strict on real invoices (18/22 FAIL). Most of the work after the first real test
  was understanding real invoice formats and GST rules better.
- Deciding what to do when two rules disagree (place of supply) - I chose "review" instead of guessing.

## What I learned

- Testing only on data you generated yourself hides big problems. My checker scored 10/10 with 0 false
  alarms on my own generated invoices, then marked 18 of 22 real invoices as FAIL. So I threw the generated
  data away and now test only on real documents.
- A test that scores 100% isn't automatically good. My error-injection test only became useful when I ran
  it on enough real documents (383) to find the cases my rules didn't cover.
- Datasets have mistakes too, so you have to check the labels before trusting the accuracy number.
- How GST works in practice: GSTIN structure, CGST/SGST vs IGST, place of supply, cess and TCS.
- How to use a vision LLM with structured JSON output, and how to put checks around it.

## Things I want to add later

- Explain errors in Telugu/Hindi for shop owners
- Handle multi-page invoices as one document
- Look up the correct GST rate from the HSN code
- Understand city names in the place of supply (map cities to states)
- Save the clean data to a database so it can be queried (like my other project, AskDB)
