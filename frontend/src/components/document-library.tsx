"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { useApiKey } from "@/components/api-key-provider";
import {
  ApiError,
  deleteDocument,
  listDocuments,
  uploadDocument,
} from "@/lib/api";
import { formatBytes } from "@/lib/format";
import type { DocumentDetail, DocumentSummary } from "@/lib/types";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorNotice,
  SectionHeading,
  Spinner,
} from "@/components/ui";

const MAX_CLIENT_BYTES = 10 * 1024 * 1024;

export function DocumentLibrary() {
  const { apiKey, ready } = useApiKey();
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  const [file, setFile] = useState<File | null>(null);
  const [collectionId, setCollectionId] = useState("default");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<DocumentDetail | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!ready || !apiKey) {
      setDocuments([]);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    listDocuments(apiKey, controller.signal)
      .then(setDocuments)
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(cause instanceof ApiError ? cause.message : "Could not load documents.");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [apiKey, ready, refreshToken]);

  const refresh = useCallback(() => setRefreshToken((value) => value + 1), []);

  async function handleUpload(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiKey) return;
    if (!file) {
      setUploadError("Choose a PDF file first.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only .pdf files are accepted.");
      return;
    }
    if (file.size > MAX_CLIENT_BYTES) {
      setUploadError("This file is larger than the 10 MB default upload limit.");
      return;
    }
    if (!collectionId.trim()) {
      setUploadError("Collection id must not be empty.");
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploaded(null);
    try {
      const result = await uploadDocument(apiKey, file, collectionId.trim());
      setUploaded(result);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      refresh();
    } catch (cause) {
      setUploadError(cause instanceof ApiError ? cause.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(document: DocumentSummary) {
    if (!apiKey) return;
    const confirmed = window.confirm(
      `Delete "${document.filename}"? Its pages, chunks, and embeddings will be removed.`,
    );
    if (!confirmed) return;
    try {
      await deleteDocument(apiKey, document.id);
      refresh();
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "Delete failed.");
    }
  }

  if (!ready) {
    return (
      <Card>
        <Spinner label="Loading workspace" />
      </Card>
    );
  }

  if (!apiKey) {
    return (
      <Card>
        <SectionHeading
          title="Document library"
          description="Prism needs an API key before it can list or process documents."
        />
        <EmptyState
          title="API key required"
          description="Open “Set API key” in the header and enter one of the keys configured in PRISM_API_KEYS (for local development this is often dev-key)."
        />
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <SectionHeading
          title="Upload a PDF"
          description="Text is extracted page by page, chunked with page metadata, embedded locally, and indexed for retrieval. Scanned PDFs without a text layer are reported rather than silently accepted."
        />
        <form onSubmit={handleUpload} className="grid gap-4 sm:grid-cols-[1fr_14rem_auto] sm:items-end">
          <div>
            <label htmlFor="pdf-file" className="block text-sm font-medium text-slate-700">
              PDF file
            </label>
            <input
              ref={fileInputRef}
              id="pdf-file"
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null);
                setUploadError(null);
              }}
              className="mt-1.5 block w-full cursor-pointer rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 file:mr-3 file:rounded-md file:border-0 file:bg-prism-50 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-prism-700 focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
            />
          </div>
          <div>
            <label htmlFor="collection-id" className="block text-sm font-medium text-slate-700">
              Collection
            </label>
            <input
              id="collection-id"
              value={collectionId}
              onChange={(event) => setCollectionId(event.target.value)}
              maxLength={128}
              className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
            />
          </div>
          <Button type="submit" disabled={uploading}>
            {uploading ? "Processing…" : "Upload"}
          </Button>
        </form>

        {uploading ? (
          <p className="mt-3 text-sm text-slate-600" aria-live="polite">
            Extracting text, generating embeddings, and indexing. The first upload also downloads the local embedding model.
          </p>
        ) : null}
        {uploadError ? (
          <div className="mt-4">
            <ErrorNotice title="Upload failed" message={uploadError} />
          </div>
        ) : null}
        {uploaded ? (
          <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900" role="status">
            <p className="font-semibold">Uploaded and indexed {uploaded.filename}</p>
            <p className="mt-1">
              {uploaded.page_count} pages · {uploaded.extracted_char_count.toLocaleString()} extracted characters ·{" "}
              {uploaded.chunk_count} chunks.
            </p>
            {uploaded.warnings.length > 0 ? (
              <ul className="mt-2 list-disc space-y-1 pl-5">
                {uploaded.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            ) : null}
            <Link href={`/documents/${uploaded.id}`} className="mt-2 inline-block font-medium text-emerald-800 underline">
              Inspect extracted pages
            </Link>
          </div>
        ) : null}
      </Card>

      <Card>
        <SectionHeading
          title="Document library"
          description="Each document is scoped to the owner mapped from your API key."
          action={
            <Button variant="secondary" onClick={refresh} disabled={loading}>
              Refresh
            </Button>
          }
        />
        {error ? <ErrorNotice title="Could not load documents" message={error} /> : null}
        {loading && documents.length === 0 ? (
          <Spinner label="Loading documents" />
        ) : documents.length === 0 && !error ? (
          <EmptyState
            title="No documents yet"
            description="Upload a text-based PDF to build your first searchable collection."
          />
        ) : documents.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full border-separate border-spacing-y-2 text-left text-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide text-slate-500">
                  <th scope="col" className="px-3 py-1 font-medium">Document</th>
                  <th scope="col" className="px-3 py-1 font-medium">Pages</th>
                  <th scope="col" className="px-3 py-1 font-medium">Chunks</th>
                  <th scope="col" className="px-3 py-1 font-medium">Size</th>
                  <th scope="col" className="px-3 py-1 font-medium">Status</th>
                  <th scope="col" className="px-3 py-1 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((document) => (
                  <tr key={document.id} className="rounded-xl bg-slate-50 text-slate-700">
                    <td className="rounded-l-xl px-3 py-3">
                      <Link href={`/documents/${document.id}`} className="font-medium text-ink hover:text-prism-700">
                        {document.filename}
                      </Link>
                      <p className="mt-0.5 font-mono text-[11px] text-slate-400">{document.id}</p>
                    </td>
                    <td className="px-3 py-3">{document.page_count}</td>
                    <td className="px-3 py-3">{document.chunk_count}</td>
                    <td className="px-3 py-3">{formatBytes(document.size_bytes)}</td>
                    <td className="px-3 py-3">
                      <Badge tone={document.status === "processed" ? "success" : "warning"}>{document.status}</Badge>
                    </td>
                    <td className="rounded-r-xl px-3 py-3 text-right">
                      <div className="inline-flex gap-2">
                        <Link
                          href={`/documents/${document.id}`}
                          className="rounded-md px-2 py-1 text-sm font-medium text-prism-700 hover:bg-prism-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-prism-600"
                        >
                          Open
                        </Link>
                        <Button variant="ghost" onClick={() => handleDelete(document)}>
                          Delete
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
