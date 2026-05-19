import { useState, useRef, useCallback, useEffect } from 'react';
import React from 'react';

const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000';

function parseMarkdown(text) {
  if (!text) return null;
  
  const lines = text.split('\n');
  
  return lines.map((line, idx) => {
    let isListItem = false;
    let listContent = line;
    let listType = null;
    
    const olMatch = line.match(/^\s*(\d+)\.\s+(.*)/);
    const ulMatch = line.match(/^\s*([*\-–])\s+(.*)/);
    
    if (olMatch) {
      isListItem = true;
      listType = 'ol';
      listContent = olMatch[2];
    } else if (ulMatch) {
      isListItem = true;
      listType = 'ul';
      listContent = ulMatch[2];
    }
    
    const parts = [];
    let currentText = isListItem ? listContent : line;
    const boldRegex = /\*\*([^*]+)\*\*/g;
    let match;
    let lastIndex = 0;
    
    while ((match = boldRegex.exec(currentText)) !== null) {
      if (match.index > lastIndex) {
        parts.push(currentText.substring(lastIndex, match.index));
      }
      parts.push(<strong key={match.index}>{match[1]}</strong>);
      lastIndex = boldRegex.lastIndex;
    }
    
    if (lastIndex < currentText.length) {
      parts.push(currentText.substring(lastIndex));
    }
    
    if (isListItem) {
      return (
        <div key={idx} className={`md-list-item md-${listType}`}>
          <span className="md-list-bullet">
            {olMatch ? `${olMatch[1]}.` : '•'}
          </span>
          <span className="md-list-text">{parts}</span>
        </div>
      );
    }
    
    if (line.trim() === '') {
      return <div key={idx} className="md-spacer" />;
    }
    
    return <p key={idx}>{parts}</p>;
  });
}

const UploadIcon = () => (
  <svg className="drop-icon" viewBox="0 0 48 48" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 32s-6 0-6-8c0-4.4 3.2-8 8-8 .4 0 .8 0 1.2.1C20.8 12.5 25 9 30 9c6.6 0 12 5.4 12 12 0 .3 0 .6-.1.9C44.8 23 46 25.4 46 28c0 4.4-3.6 8-8 8H16z" />
    <path d="M20 36l4-5 4 5M24 31v13" />
  </svg>
);

const FileIcon = () => (
  <svg className="file-icon" width="18" height="18" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="16" y1="13" x2="8" y2="13"/>
    <line x1="16" y1="17" x2="8" y2="17"/>
    <polyline points="10 9 9 9 8 9"/>
  </svg>
);

const InfoIcon = () => (
  <svg className="tooltip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="8" x2="12" y2="8.5"/>
    <line x1="12" y1="12" x2="12" y2="16"/>
  </svg>
);

const AlertIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

const CopyIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>
);

const CheckIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

function HowItWorks({ onClose }) {
  React.useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const steps = [
    { title: 'Upload a PDF', desc: 'Your document is saved on the server and Mistral OCR extracts markdown from every page.' },
    { title: 'Tree Indexing', desc: 'A heading-based tree is built over the OCR output — titles, page ranges, and summaries — stored as JSON.' },
    { title: 'Agentic Retrieval', desc: 'A LangGraph pipeline reads the tree and decides which pages are most relevant to your question.' },
    { title: 'Grounded Answer', desc: 'Mistral generates an answer strictly from the retrieved pages — no hallucinations.' },
    { title: 'Optional Verification', desc: 'A second LLM pass checks the answer for accuracy and returns structured feedback.' },
  ];

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-label="How PageMind works">
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">How PageMind works</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="steps">
          {steps.map((s, i) => (
            <div className="step" key={i}>
              <div className="step-num">{i + 1}</div>
              <div className="step-content">
                <strong>{s.title}</strong>
                <span>{s.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatusIndicator({ status }) {
  const labels = { idle: 'Idle', indexing: 'Indexing…', ready: 'Ready', error: 'Error' };
  return (
    <span className={`status-indicator status-${status}`}>
      <span className="status-dot" />
      {labels[status] ?? status}
    </span>
  );
}

export default function App() {
  const [showHowItWorks, setShowHowItWorks] = useState(false);

  // Upload state
  const [file, setFile] = useState(null);
  const [docId, setDocId] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [uploadError, setUploadError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Chat state
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [verification, setVerification] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryError, setQueryError] = useState(null);

  const fileInputRef = useRef(null);
  const chatHistoryRef = useRef(null);

  // Auto-scroll chat
  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages, isQuerying, queryError]);

  const handleCopy = useCallback(async (text, setCopiedFn) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedFn(true);
      setTimeout(() => setCopiedFn(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  }, []);

  const handleFileChange = useCallback((f) => {
    if (!f) return;
    if (!f.name.endsWith('.pdf')) {
      setUploadError('Only PDF files are supported.');
      return;
    }
    setFile(f);
    setUploadError(null);
    setUploadStatus('idle');
    setDocId(null);
    setMessages([]);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    handleFileChange(dropped);
  }, [handleFileChange]);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus('indexing');
    setUploadError(null);
    setMessages([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed.' }));
        throw new Error(err.detail ?? 'Upload failed.');
      }
      const data = await res.json();
      setDocId(data.doc_id);
      setUploadStatus('ready');
    } catch (err) {
      setUploadStatus('error');
      setUploadError(err.message);
    }
  };

  const handleQuery = async () => {
    if (!query.trim() || !docId) return;
    
    const userText = query.trim();
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    
    setIsQuerying(true);
    setQueryError(null);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId, user_query: userText, verification }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Query failed.' }));
        throw new Error(err.detail ?? 'Query failed.');
      }

      const data = await res.json();
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.llm_answer, 
        feedback: data.llm_feedback 
      }]);
    } catch (err) {
      setQueryError(err.message);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleClear = async () => {
    if (docId) {
      try { await fetch(`${API_BASE}/cleanup/${docId}`, { method: 'DELETE' }); } 
      catch (_) { /* best-effort */ }
    }
    setFile(null);
    setDocId(null);
    setUploadStatus('idle');
    setUploadError(null);
    setQuery('');
    setMessages([]);
    setQueryError(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  // Helper for message copy
  const MessageCopyButton = ({ text }) => {
    const [copied, setCopied] = useState(false);
    return (
      <button className={`btn-copy-icon ${copied ? 'copied' : ''}`} onClick={() => handleCopy(text, setCopied)} title="Copy markdown">
        {copied ? <CheckIcon /> : <CopyIcon />}
      </button>
    );
  };

  return (
    <div className="app-container">
      {showHowItWorks && <HowItWorks onClose={() => setShowHowItWorks(false)} />}

      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-block">
            <span className="logo-name">PageMind</span>
            <span className="logo-sub">Agentic RAG</span>
          </div>
        </div>

        <div className="sidebar-content">
          <button className="btn-new-chat" onClick={handleClear}>
            <span className="new-chat-icon">+</span> New Chat
          </button>

          {file && (
            <div className="sidebar-doc">
              <div className="sidebar-doc-icon"><FileIcon /></div>
              <div className="sidebar-doc-info">
                <span className="sidebar-file-name" title={file.name}>{file.name}</span>
                <StatusIndicator status={uploadStatus} />
              </div>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <button className="nav-link" onClick={() => setShowHowItWorks(v => !v)}>
            How it works
          </button>
          <div className="footer-credit">
            PageMind · Vectorless Agentic RAG<br/>Built with LangGraph + Mistral
          </div>
        </div>
      </aside>

      <main className="chat-area">
        <div className="chat-history" ref={chatHistoryRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <section className="hero">
                <h1 className="hero-title">Ask your documents <em>anything</em></h1>
                <p className="hero-sub">Upload a PDF and get precise, grounded answers — powered by reasoning-based retrieval</p>
                <div className="hero-divider" />
              </section>

              <section className="upload-section">
                <p className="section-label">01 — Upload document</p>
                <div
                  className={`drop-zone ${isDragging ? 'dragging' : ''} ${file ? 'file-selected' : ''}`}
                  onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}
                  onClick={() => fileInputRef.current?.click()}
                  role="button" tabIndex={0}
                  onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFileChange(e.target.files?.[0])}
                    tabIndex={-1}
                  />
                  {file ? (
                    <>
                      <FileIcon />
                      <p className="drop-title drop-title--file">{file.name}</p>
                      <p className="drop-hint">Click to change file</p>
                    </>
                  ) : (
                    <>
                      <UploadIcon />
                      <p className="drop-title">Drop your PDF here</p>
                      <p className="drop-hint">or click to browse — PDF files only</p>
                    </>
                  )}
                </div>

                {uploadError && (
                  <div className="error-toast" style={{ marginTop: '1rem' }} role="alert">
                    <AlertIcon /> {uploadError}
                  </div>
                )}

                <div className="upload-actions">
                  <button className="btn-amber" onClick={handleUpload} disabled={!file || uploadStatus === 'indexing'}>
                    {uploadStatus === 'indexing' ? <><span className="spinner" /> Indexing…</> : 'Upload & Index'}
                  </button>
                </div>
              </section>
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((m, i) => (
                <div key={i} className={`message ${m.role}`}>
                  <div className="message-inner">
                    {m.role === 'assistant' && <div className="message-avatar">PM</div>}
                    <div className="message-content">
                      {m.role === 'user' ? (
                        <div className="user-text">{m.content}</div>
                      ) : (
                        <div className="assistant-card">
                          <div className="assistant-header">
                            <span className="answer-label">PageMind</span>
                            <MessageCopyButton text={m.content} />
                          </div>
                          <div className="answer-body">{parseMarkdown(m.content)}</div>
                          {m.feedback && (
                            <div className="verification-feedback">
                              <p className="feedback-label">Verification feedback</p>
                              <div className="feedback-body">{parseMarkdown(m.feedback)}</div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {isQuerying && (
                <div className="message assistant thinking">
                  <div className="message-inner">
                    <div className="message-avatar">PM</div>
                    <div className="message-content">
                      <div className="assistant-card"><span className="spinner" /> Thinking…</div>
                    </div>
                  </div>
                </div>
              )}
              {queryError && (
                <div className="message assistant error">
                  <div className="message-inner">
                    <div className="message-avatar">PM</div>
                    <div className="message-content">
                      <div className="error-toast" role="alert"><AlertIcon /> {queryError}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="chat-input-area">
          {uploadStatus === 'ready' ? (
            <div className="input-container">
              <div className="input-box">
                <textarea
                  className="query-input"
                  placeholder="Ask a question about your document..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                />
                <button className="btn-send" onClick={handleQuery} disabled={!query.trim() || isQuerying}>
                  {isQuerying ? <span className="spinner small" /> : <SendIcon />}
                </button>
              </div>
              <div className="input-footer">
                <label className="verification-toggle">
                  <input type="checkbox" className="toggle-checkbox" checked={verification} onChange={(e) => setVerification(e.target.checked)} />
                  <span className="toggle-label">Enable answer verification</span>
                  <span className="tooltip-wrap">
                    <InfoIcon />
                    <span className="tooltip-box" role="tooltip">
                      Runs a second LLM pass to check the answer against the source pages and returns structured accuracy feedback.
                    </span>
                  </span>
                </label>
              </div>
            </div>
          ) : (
            <div className="input-placeholder">
              Upload and index a document to start chatting.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
