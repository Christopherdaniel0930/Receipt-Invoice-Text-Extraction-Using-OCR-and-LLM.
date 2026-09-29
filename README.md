# Receipt / Invoice Information Extraction

A simple receipt and invoice extraction pipeline using **RapidOCR + Ollama/gpt-oss + Pydantic + TF-IDF/Logistic Regression**.

## Architecture

```text
IMAGE
  ↓
RapidOCR (PP-OCRv6 ONNX)
  ↓
Raw OCR Text
  ↓
OCR Text Normalization
  ↓
Currency Recovery (symbol / lookalike / locale / glyph crop)
  ↓
Symbol restored into the amounts
  ↓
Document gate (hard-counter veto only, on the restored text)
  ↓
Ollama + gpt-oss
  ↓
Structured ReceiptData
  ↓
Pydantic + deterministic validation
  ↓
TF-IDF + Logistic Regression
  ↓
Expense Category
  ↓
Final JSON
```

## Responsibilities

### RapidOCR
Only performs OCR: image → text.

### OCR normalization
Removes unnecessary whitespace, blank lines, and consecutive duplicate lines without changing meaning.

### Ollama + gpt-oss
Converts messy OCR text into structured fields:

- vendor
- invoice number
- date
- tax
- total
- currency
- line items

gpt-oss is **not** used for expense classification.

### Document gate
`app/doc_classifier.py` scores the OCR text against receipt/invoice rules and
rejects the document before any LLM call is made — but **only** on its hard
counters: identity documents, personal letters, CVs, legal contracts, news
articles, and business cards. Those can never be a receipt.

A low score does **not** reject. UPI payment screens, card-payment screenshots,
bus tickets and bank confirmations score poorly because they do not look like a
till receipt, and on the sample corpus a full `is_receipt_invoice` gate rejected
7 of 22 genuine money documents. Bank statements are a soft counter and are not
vetoed either, because a statement that is really a payment confirmation still
has a total worth extracting. The verdict is reported as `document_type` and
`document_confidence` in the output JSON.

`business_card` is a combined rule rather than a keyword: a page needs two or
more contact markers (phone, fax, email, website, or a company suffix) **and** no
money at all. A phone number alone proves nothing, since real receipts print
those too. The money half of the test uses the structural `has_money` signal
rather than the raw amount count, because an invoice can print its figures as
bare integers (`10 1000 10000`) that carry no symbol or decimal point and so are
never counted by `AMOUNT_RE`. On the corpus this vetoes the four letterheads and
business cards and nothing else.

Anything that survives the gate is reported at no less than
`MIN_FORWARDED_CONFIDENCE` (0.50) in `document_confidence`, so a consumer that
filters on confidence keeps exactly the same set of documents the gate keeps.
A document that scores well keeps its own higher confidence, and the raw
`is_receipt_invoice` boolean is left untouched.

### Currency recovery
OCR routinely drops currency symbols. Before gpt-oss runs, `app/currency_recovery.py`
scores the surviving evidence — a symbol or ISO code in the text, a letter that a
symbol may have been flattened into, regional markers, and a re-read of the ink
cropped from just left of each amount — and writes the recovered symbol back into
the amounts that lost it. gpt-oss then reads the patched text, and the recovered ISO
code is only used as a fallback when gpt-oss returns no currency at all. The layer
never fails an extraction: if it errors, the pipeline continues without it.

When a page contains no money-shaped number at all — a UPI screen whose only
figure is a bare `190` — a standalone short figure is marked as well, because
such a receipt otherwise reads as a document with no total in it and is
classified as `unknown`. The rule is deliberately narrow: it only fires when the
page has no decimal or grouped amount, and only on a line that is nothing but a
short integer, so a 12-digit UPI transaction id or a bank account is never
marked. This happens before the document gate runs, so the classifier judges the
text with the recovered symbol already in place.

Inspect a single receipt with:

```bash
python -m app.currency_recovery data/test/image6.jpeg --evidence
```

The pixel layer is skipped entirely when the OCR text still carries a currency
symbol, since a symbol that survived has nothing to recover. It also runs on its
own RapidOCR instance, because a `use_det=False` call corrupts whichever engine
it runs on: sharing one engine with the full-image pass makes every later
receipt in a batch OCR as empty.

### Pydantic
Enforces the structured application schema.

### Validation
Performs deterministic checks such as negative amounts and tax greater than total.

It also reconciles the total against the page, because a toll ticket prints its
fare once with no `Total` label and the model correctly returns `total_amount:
null` rather than promoting an unlabelled number. `reconcile_totals()` in
`app/validator.py` then takes the total from the single line item, from the line
item sum, or from the only currency-anchored amount on the page.

A line item whose description is mostly OCR damage is dropped first. A ticket
can read back as `ADU4T (S)343)-R`, and the amount beside it is part of the
same damage: the model returned `5729.00` as both the line price and the total
when the real fare was `129.00`. When a total matches a price that was just
rejected as damage, it is cleared and re-derived, because an amount whose
currency symbol only exists because recovery patched one onto a garbled line is
not evidence of a total. Every change is recorded in `reconciliation_notes`, so a
filled-in total is distinguishable from one the model stated.

### Expense classifier
A trainable **TF-IDF + Logistic Regression** model predicts:

- Food
- Travel
- Fuel
- Electronics
- Office Supplies
- Accommodation
- Healthcare
- Utilities
- Other

## Setup

Python 3.11 or 3.12 is recommended.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure your Ollama endpoint, API key, and gpt-oss model.

### Ollama Cloud/API

```env
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_API_KEY=your_key
OLLAMA_MODEL=your-model-name
```

### Local Ollama

```env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=qwen3:8b
```

Never commit `.env` to Git.

## Train the expense classifier

```bash
python train.py
```

This reads `data/train/categories.csv` and writes:

```text
models/category_model.joblib
```

## Run on an image

```bash
python -m app.main data/test/image.jpg
```

The program runs:

```text
image → RapidOCR → gpt-oss → validation → category model → JSON
```

## Run with OCR text

```bash
python -m app.main data/test/sample_receipt.txt
```

## Tests

```bash
pytest -v
