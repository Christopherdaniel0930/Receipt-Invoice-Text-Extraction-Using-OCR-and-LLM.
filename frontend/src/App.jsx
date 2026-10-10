import React, { useEffect, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const TERMINAL_STATES = new Set(["COMPLETED", "FAILED", "REJECTED"]);
const POLL_INTERVAL_MS = 1500;
const MAX_POLL_ATTEMPTS = 300;

function displayValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function formatAmount(value, currency) {
  if (value === null || value === undefined || value === "") return "—";
  const amount = Number(value);
  if (!Number.isFinite(amount)) return String(value);
  return currency ? `${currency} ${amount.toFixed(2)}` : amount.toFixed(2);
}

function readableStatus(status, documentType) {
  const kind = documentType === "business_card" ? "business card" : "document";
  return {
    QUEUED: "In the queue",
    PROCESSING: `Reading your ${kind}`,
    COMPLETED: "Extraction complete",
    FAILED: `Could not process ${kind}`,
    REJECTED: "Document not supported",
  }[status] || status;
}

function responseMessage(data, fallback) {
  if (typeof data?.error === "string") return data.error;
  return data?.error?.message || data?.message || fallback;
}

function resultData(statusData) {
  const envelope = statusData.result;
  if (envelope && typeof envelope === "object" && "result" in envelope) {
    return envelope.result;
  }
  return envelope;
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [documentType, setDocumentType] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const requestId = useRef(0);
  const timerRef = useRef(null);
  const controllerRef = useRef(null);

  useEffect(() => () => {
    requestId.current += 1;
    window.clearTimeout(timerRef.current);
    controllerRef.current?.abort();
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  function stopRequest() {
    window.clearTimeout(timerRef.current);
    controllerRef.current?.abort();
    controllerRef.current = null;
  }

  function clearResults() {
    requestId.current += 1;
    stopRequest();
    setBusy(false);
    setJob(null);
    setResult(null);
    setDocumentType("");
    setError("");
    setFile(null);
    setPreview("");
    if (inputRef.current) inputRef.current.value = "";
  }

  function selectFile(candidate) {
    if (!candidate) return;
    if (inputRef.current) inputRef.current.value = "";
    requestId.current += 1;
    stopRequest();
    setBusy(false);
    setJob(null);
    setResult(null);
    setDocumentType("");
    setError("");
    setPreview("");
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setFile(null);
      setError("Choose a JPEG, PNG, or WEBP image.");
      return;
    }
    if (candidate.size > MAX_FILE_SIZE) {
      setFile(null);
      setError("The image must be 10 MB or smaller.");
      return;
    }
    setFile(candidate);
    setPreview(URL.createObjectURL(candidate));
  }

  async function submitDocument() {
    if (!file || busy) return;
    const currentRequest = ++requestId.current;
    setBusy(true);
    setError("");
    setResult(null);
    setDocumentType("");
    setJob({ status: "QUEUED", jobId: "" });

    try {
      const form = new FormData();
      form.append("file", file);
      const uploadController = new AbortController();
      controllerRef.current = uploadController;
      const uploadResponse = await fetch(`${API_BASE}/api/v1/documents`, {
        method: "POST",
        body: form,
        signal: uploadController.signal,
      });
      const upload = await uploadResponse.json();
      if (!uploadResponse.ok) throw new Error(responseMessage(upload, "Upload failed. Please try again."));
      if (currentRequest !== requestId.current) return;
      if (!upload.job_id) throw new Error("The server did not return a job ID.");

      setJob({ status: "QUEUED", jobId: upload.job_id });
      let attempts = 0;

      const poll = async () => {
        if (currentRequest !== requestId.current) return;
        attempts += 1;
        if (attempts > MAX_POLL_ATTEMPTS) {
          setError("Processing is taking longer than expected. You can check this job again later.");
          setBusy(false);
          return;
        }

        try {
          const controller = new AbortController();
          controllerRef.current = controller;
          const statusResponse = await fetch(`${API_BASE}/api/v1/jobs/${upload.job_id}`, {
            signal: controller.signal,
          });
          const statusData = await statusResponse.json();
          if (currentRequest !== requestId.current) return;
          if (!statusResponse.ok) throw new Error(responseMessage(statusData, "Could not check job status."));

          const status = String(statusData.status || "").toUpperCase();
          const type = statusData.document_type || statusData.result?.document_type || "";
          setDocumentType(type);
          setJob({ status, jobId: upload.job_id, error: statusData.error });

          if (status === "COMPLETED") {
            setResult(resultData(statusData));
            setBusy(false);
            return;
          }
          if (status === "REJECTED" || status === "FAILED") {
            setError(responseMessage(statusData, readableStatus(status, type)));
            setBusy(false);
            return;
          }

          timerRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
        } catch (requestError) {
          if (requestError.name === "AbortError" || currentRequest !== requestId.current) return;
          setError(requestError.message || "Could not check job status. Please try again.");
          setBusy(false);
        }
      };

      timerRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
    } catch (requestError) {
      if (requestError.name === "AbortError" || currentRequest !== requestId.current) return;
      setError(requestError.message || "Something went wrong. Please try again.");
      setBusy(false);
    }
  }

  const completed = job?.status === "COMPLETED" && result;
  const isTerminal = job && TERMINAL_STATES.has(job.status);

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Document Extractor home">
          <span className="brand-mark"><span /></span>
          <span>Document <span className="brand-light">Extractor</span></span>
        </a>
        <span className="topbar-note"><span className="status-dot" /> Receipt, invoice & business card processing</span>
      </header>

      <main id="top" className="main-content">
        <section className="intro">
          <div className="eyebrow"><span className="sparkle">✳</span> DOCUMENTS, MADE SIMPLE</div>
          <h1>Every detail,<br /><span>already taken care of.</span></h1>
          <p>Upload a receipt, invoice, or business card. We’ll turn it into tidy, useful data.</p>
        </section>

        <section className="workspace" aria-label="Document extraction">
          <div className="upload-card">
            <div className="card-heading">
              <div>
                <span className="step-label">01 <span>·</span> UPLOAD</span>
                <h2>Your document</h2>
              </div>
              <span className="file-limit">JPEG, PNG, WEBP <i /> 10 MB max</span>
            </div>

            {!file ? (
              <div
                className={`dropzone ${dragging ? "dropzone-active" : ""}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  selectFile(event.dataTransfer.files[0]);
                }}
              >
                <input
                  ref={inputRef}
                  className="visually-hidden"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(event) => selectFile(event.target.files[0])}
                />
                <div className="upload-icon" aria-hidden="true">
                  <svg viewBox="0 0 48 48" fill="none"><path d="M24 31V9m0 0-8 8m8-8 8 8M10 29v8a3 3 0 0 0 3 3h22a3 3 0 0 0 3-3v-8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </div>
                <strong>Drop your image here</strong>
                <span className="drop-subtitle">or browse from your device</span>
                <button className="button button-secondary" type="button" onClick={() => inputRef.current?.click()}>
                  Choose an image <span aria-hidden="true">↗</span>
                </button>
              </div>
            ) : (
              <div className="selected-file">
                <div className="preview-frame"><img src={preview} alt="Selected document preview" /></div>
                <div className="file-details">
                  <span className="file-kicker">READY TO EXTRACT</span>
                  <strong title={file.name}>{file.name}</strong>
                  <span>{(file.size / 1024 / 1024).toFixed(2)} MB · {file.type.split("/")[1].toUpperCase()}</span>
                  <button className="text-button" type="button" onClick={() => inputRef.current?.click()}>Choose a different image</button>
                  <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => selectFile(event.target.files[0])} />
                </div>
                <span className="checkmark" aria-label="Image ready">✓</span>
              </div>
            )}

            {error && <div className="error-message" role="alert"><span>!</span>{error}</div>}

            <div className="upload-footer">
              <span className="privacy-note"><span aria-hidden="true">◇</span> Image uploaded for extraction</span>
              <div className="upload-actions">
                {(file || job || error) && <button className="button button-secondary reset-button" type="button" onClick={clearResults}>Clear / Reset</button>}
                <button className="button button-primary" type="button" disabled={!file || busy} onClick={submitDocument}>
                  {busy ? <><span className="button-spinner" /> Processing</> : <>Upload / Extract <span aria-hidden="true">→</span></>}
                </button>
              </div>
            </div>
          </div>

          <div className="result-column">
            {job ? (
              <section className={`result-card ${completed ? "result-complete" : ""}`} aria-live="polite">
                <div className="card-heading result-heading">
                  <div>
                    <span className="step-label">02 <span>·</span> EXTRACTION</span>
                    <h2>{completed ? "Looking good." : isTerminal ? "Needs attention." : "Working on it."}</h2>
                  </div>
                  {completed && <span className="success-badge">✓ Done</span>}
                </div>

                {!isTerminal ? (
                  <div className="job-progress">
                    <span className="progress-icon"><span className="progress-spinner" /></span>
                    <div><strong>{readableStatus(job.status, documentType)}</strong><span>This usually takes a few moments.</span></div>
                    <span className={`state-pill state-${job.status.toLowerCase()}`}>{job.status}</span>
                  </div>
                ) : completed ? (
                  <div className="job-progress">
                    <span className="progress-icon progress-done">✓</span>
                    <div><strong>{readableStatus(job.status, documentType)}</strong><span>Your extracted fields are ready below.</span></div>
                    <span className={`state-pill state-${job.status.toLowerCase()}`}>{job.status}</span>
                  </div>
                ) : (
                  <div className="terminal-state">
                    <span className="terminal-icon">!</span>
                    <div><strong>{readableStatus(job.status, documentType)}</strong><span>{error || "Please try another image."}</span></div>
                    <span className={`state-pill state-${job.status.toLowerCase()}`}>{job.status}</span>
                  </div>
                )}

                {completed && documentType === "business_card" && <BusinessCardResult data={result} />}
                {completed && documentType === "receipt_invoice" && <ReceiptResult data={result} />}
                {completed && !["business_card", "receipt_invoice"].includes(documentType) && <DynamicResult data={result} />}
                {job.jobId && <div className="job-reference">JOB ID <code>{job.jobId}</code></div>}
              </section>
            ) : (
              <section className="placeholder-card">
                <div className="placeholder-art" aria-hidden="true">
                  <div className="art-paper"><span /><span /><span /><b /></div>
                  <div className="art-spark art-spark-one">✳</div><div className="art-spark art-spark-two">✳</div>
                </div>
                <span className="step-label">02 <span>·</span> YOUR RESULTS</span>
                <h2>Clarity, at a glance.</h2>
                <p>Contact details, merchants, totals, and line items—organized and ready to use.</p>
                <div className="result-tags"><span>Business cards</span><span>Receipts</span><span>Invoices</span></div>
              </section>
            )}
          </div>
        </section>

        <footer className="page-footer"><span>Made for the little things that add up.</span><span>Document Extractor <b>·</b> Receipt, invoice & business card extraction</span></footer>
      </main>
    </div>
  );
}

function BusinessCardResult({ data = {} }) {
  const fields = [
    ["Name", data.name],
    ["Designation", data.designation],
    ["Company", data.company_name],
    ["Phone", data.phone],
    ["Fax", data.fax],
    ["Email", data.email],
    ["Website", data.website],
    ["Address", data.address],
  ];
  return (
    <div className="business-card-result">
      <h3>Business card details</h3>
      <dl className="field-grid">
        {fields.map(([label, value]) => (
          <div className={`field-item ${label === "Address" ? "field-wide" : ""}`} key={label}>
            <dt>{label}</dt><dd>{displayValue(value)}</dd>
          </div>
        ))}
      </dl>
      <RawFields data={data} />
    </div>
  );
}

function ReceiptResult({ data = {} }) {
  const lineItems = Array.isArray(data.line_items) ? data.line_items : [];
  return (
    <div className="receipt-result">
      <div className="summary-grid">
        <SummaryItem label="MERCHANT" value={data.vendor_name || "Not identified"} wide />
        <SummaryItem label="TOTAL" value={formatAmount(data.total_amount, data.currency)} total />
        <SummaryItem label="DATE" value={data.invoice_date} />
        <SummaryItem label="INVOICE NUMBER" value={data.invoice_number} />
        <SummaryItem label="CATEGORY" value={data.expense_category} />
        <SummaryItem label="TAX" value={formatAmount(data.tax_amount, data.currency)} />
        <SummaryItem label="CURRENCY" value={data.currency} />
      </div>
      {lineItems.length > 0 && (
        <div className="items-section">
          <div className="items-heading"><strong>Line items</strong><span>{lineItems.length} found</span></div>
          <div className="line-items">
            {lineItems.map((item, index) => (
              <div className="line-item" key={`${item.description || "item"}-${index}`}>
                <span>{item.description || "Item"}{item.quantity ? <small> × {item.quantity}</small> : null}</span>
                <strong>{formatAmount(item.amount, data.currency)}</strong>
              </div>
            ))}
          </div>
        </div>
      )}
      <RawFields data={data} />
    </div>
  );
}

function RawFields({ data }) {
  return (
    <details className="raw-details">
      <summary>View all extracted fields</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

function SummaryItem({ label, value, wide = false, total = false }) {
  return (
    <div className={`summary-item ${wide ? "summary-wide" : ""}`}>
      <span>{label}</span><strong className={total ? "total-value" : ""}>{displayValue(value)}</strong>
    </div>
  );
}

function DynamicResult({ data }) {
  if (!data || typeof data !== "object") return <p className="empty-result">No structured fields were returned.</p>;
  return <DynamicFields entries={Object.entries(data)} title="Extracted fields" />;
}

function DynamicFields({ entries, title }) {
  return (
    <div className="dynamic-fields">
      <h3>{title}</h3>
      <dl className="field-grid">
        {entries.map(([key, value]) => (
          <div className="field-item" key={key}>
            <dt>{key.replaceAll("_", " ")}</dt><dd>{displayValue(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default App;
