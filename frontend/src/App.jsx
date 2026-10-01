import React, { useEffect, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const TERMINAL_STATES = new Set(["COMPLETED", "FAILED", "REJECTED"]);

function formatAmount(value, currency) {
  if (value === null || value === undefined || value === "") return "—";
  const amount = Number(value);
  if (!Number.isFinite(amount)) return String(value);
  return currency ? `${currency} ${amount.toFixed(2)}` : amount.toFixed(2);
}

function readableStatus(status) {
  return {
    QUEUED: "In the queue",
    PROCESSING: "Reading your receipt",
    COMPLETED: "Extraction complete",
    FAILED: "Could not process receipt",
    REJECTED: "Not recognized as a receipt",
  }[status] || status;
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [job, setJob] = useState(null);
  const [receipt, setReceipt] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const requestId = useRef(0);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  function selectFile(candidate) {
    if (!candidate) return;
    requestId.current += 1;
    setBusy(false);
    setJob(null);
    setReceipt(null);
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

  async function submitReceipt() {
    if (!file || busy) return;
    const currentRequest = ++requestId.current;
    setBusy(true);
    setError("");
    setReceipt(null);
    setJob({ status: "QUEUED", jobId: "" });

    try {
      const form = new FormData();
      form.append("file", file);
      const uploadResponse = await fetch(`${API_BASE}/api/v1/receipts`, {
        method: "POST",
        body: form,
      });
      const upload = await uploadResponse.json();
      if (!uploadResponse.ok) throw new Error(upload.message || "Upload failed. Please try again.");
      if (currentRequest !== requestId.current) return;
      setJob({ status: "QUEUED", jobId: upload.job_id });

      for (let attempt = 0; attempt < 300; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (currentRequest !== requestId.current) return;
        const statusResponse = await fetch(`${API_BASE}/api/v1/jobs/${upload.job_id}`);
        const statusData = await statusResponse.json();
        if (currentRequest !== requestId.current) return;
        if (!statusResponse.ok) throw new Error(statusData.message || "Could not check job status.");
        setJob({ status: statusData.status, jobId: upload.job_id, error: statusData.error });
        if (statusData.status === "COMPLETED") {
          setReceipt(statusData.result);
          break;
        }
        if (statusData.status === "FAILED" || statusData.status === "REJECTED") {
          setError(statusData.error?.message || readableStatus(statusData.status));
          break;
        }
      }
      if (currentRequest === requestId.current) setBusy(false);
    } catch (requestError) {
      if (currentRequest === requestId.current) {
        setError(requestError.message || "Something went wrong. Please try again.");
        setBusy(false);
      }
    }
  }

  const completed = job?.status === "COMPLETED" && receipt;
  const isTerminal = job && TERMINAL_STATES.has(job.status);

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Receipt Extractor home">
          <span className="brand-mark"><span /></span>
          <span>Receipt <span className="brand-light">Extractor</span></span>
        </a>
        <span className="topbar-note"><span className="status-dot" /> Receipt processing</span>
      </header>

      <main id="top" className="main-content">
        <section className="intro">
          <div className="eyebrow"><span className="sparkle">✳</span> RECEIPTS, MADE SIMPLE</div>
          <h1>Every detail,<br /><span>already taken care of.</span></h1>
          <p>Upload a receipt or invoice. We’ll turn it into tidy, useful data in seconds.</p>
        </section>

        <section className="workspace" aria-label="Receipt extraction">
          <div className="upload-card">
            <div className="card-heading">
              <div>
                <span className="step-label">01 <span>·</span> UPLOAD</span>
                <h2>Your receipt</h2>
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
                <div className="preview-frame"><img src={preview} alt="Selected receipt preview" /></div>
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
              <button className="button button-primary" type="button" disabled={!file || busy} onClick={submitReceipt}>
                {busy ? <><span className="button-spinner" /> Processing</> : <>Extract details <span aria-hidden="true">→</span></>}
              </button>
            </div>
          </div>

          <div className="result-column">
            {job ? (
              <section className={`result-card ${completed ? "result-complete" : ""}`} aria-live="polite">
                <div className="card-heading result-heading">
                  <div>
                    <span className="step-label">02 <span>·</span> EXTRACTION</span>
                    <h2>{completed ? "Looking good." : "Working on it."}</h2>
                  </div>
                  {completed && <span className="success-badge">✓ Done</span>}
                </div>

                {!isTerminal || completed ? (
                  <div className="job-progress">
                    <span className={`progress-icon ${completed ? "progress-done" : ""}`}>
                      {completed ? "✓" : <span className="progress-spinner" />}
                    </span>
                    <div><strong>{completed ? readableStatus(job.status) : readableStatus(job.status)}</strong><span>{completed ? "Your receipt details are ready below." : "This usually takes a few moments."}</span></div>
                    <span className={`state-pill state-${job.status.toLowerCase()}`}>{job.status}</span>
                  </div>
                ) : (
                  <div className="terminal-state">
                    <span className="terminal-icon">!</span>
                    <div><strong>{readableStatus(job.status)}</strong><span>{error || "Please try another image."}</span></div>
                  </div>
                )}

                {completed && <ReceiptResult receipt={receipt} />}
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
                <p>Merchant, totals, tax, and line items—organized and ready to use.</p>
                <div className="result-tags"><span>Merchant</span><span>Totals</span><span>Line items</span></div>
              </section>
            )}
          </div>
        </section>

        <footer className="page-footer"><span>Made for the little things that add up.</span><span>Receipt Extractor <b>·</b> Receipt & invoice extraction</span></footer>
      </main>
    </div>
  );
}

function ReceiptResult({ receipt }) {
  const lineItems = receipt.line_items || [];
  return (
    <div className="receipt-result">
      <div className="summary-grid">
        <div className="summary-item summary-wide"><span>MERCHANT</span><strong>{receipt.vendor_name || "Not identified"}</strong></div>
        <div className="summary-item"><span>TOTAL</span><strong className="total-value">{formatAmount(receipt.total_amount, receipt.currency)}</strong></div>
        <div className="summary-item"><span>DATE</span><strong>{receipt.invoice_date || "—"}</strong></div>
        <div className="summary-item"><span>CATEGORY</span><strong>{receipt.expense_category || "—"}</strong></div>
        <div className="summary-item"><span>TAX</span><strong>{formatAmount(receipt.tax_amount, receipt.currency)}</strong></div>
      </div>
      {lineItems.length > 0 && (
        <div className="items-section">
          <div className="items-heading"><strong>Line items</strong><span>{lineItems.length} found</span></div>
          <div className="line-items">
            {lineItems.map((item, index) => (
              <div className="line-item" key={`${item.description}-${index}`}>
                <span>{item.description || "Item"}{item.quantity ? <small> × {item.quantity}</small> : null}</span>
                <strong>{formatAmount(item.amount, receipt.currency)}</strong>
              </div>
            ))}
          </div>
        </div>
      )}
      <details className="raw-details">
        <summary>View all extracted fields</summary>
        <pre>{JSON.stringify(receipt, null, 2)}</pre>
      </details>
    </div>
  );
}

export default App;
