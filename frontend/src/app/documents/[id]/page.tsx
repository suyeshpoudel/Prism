"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { useApiKey } from "@/components/api-key-provider";
import { ApiError, getDocument } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import type { DocumentDetail } from "@/lib/types";
import {
  Badge,
  Card,
  EmptyState,
  ErrorNotice,
  SectionHeading,
  Spinner,
  Stat,
  cx,
} from "@/components/ui";

export default function DocumentPage() {
  const params = useParams<{ id: string }>();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  const { apiKey, ready } = useApiKey();
  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPage, setSelectedPage] = useState(1);

  useEffect(() => {
    if (!ready || !apiKey || !id) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    getDocument(apiKey, id, controller.signal)
      .then((result) => {
        setDocument(result);
        setSelectedPage(result.pages[0]?.page_number ?? 1);
      })
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(cause instanceof ApiError ? cause.message : "Could not load this document.");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [apiKey, ready, id]);

  if (!ready) {
    return (
      <Card>
        <Spinner label="Loading document" />
      </Card>
    );
  }

  if (!apiKey) {
    return (
      <Card>
        <EmptyState
          title="API key required"
          description="Open “Set API key” in the header to view documents owned by your key."
        />
      </Card>
    );
  }

  if (loading && !document) {
    return (
      <Card>
        <Spinner label="Loading document" />
      </Card>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Link href="/" className="text-sm font-medium text-prism-700 hover:underline">
          ← Back to library
        </Link>
        <ErrorNotice title="Document unavailable" message={error} />
      </div>
    );
  }

  if (!document) return null;

  const page = document.pages.find((item) => item.page_number === selectedPage) ?? document.pages[0];
  const pageChunks = document.chunks.filter((chunk) => chunk.page_number === page?.page_number);

  return (
    <div className="space-y-6">
      <div>
        <Link href="/" className="text-sm font-medium text-prism-700 hover:underline">
          ← Back to library
        </Link>
        <div className="mt-3 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{document.filename}</h1>
            <p className="mt-1 font-mono text-xs text-slate-500">{document.id}</p>
          </div>
          <Badge tone={document.status === "processed" ? "success" : "warning"}>{document.status}</Badge>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Pages" value={document.page_count} />
        <Stat label="Chunks" value={document.chunk_count} />
        <Stat label="Characters" value={document.extracted_char_count.toLocaleString()} />
        <Stat label="File size" value={formatBytes(document.size_bytes)} />
      </div>

      {document.warnings.length > 0 ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900" role="status">
          <p className="font-semibold">Extraction warnings</p>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {document.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <Card>
        <SectionHeading
          title="Extracted pages"
          description="Each chunk is an exact slice of one page, so citations always resolve to a real page and character range."
        />
        {document.pages.length === 0 ? (
          <EmptyState title="No pages" description="This document produced no extractable pages." />
        ) : (
          <div className="space-y-4">
            <div className="flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="Pages">
              {document.pages.map((item) => (
                <button
                  key={item.page_number}
                  type="button"
                  role="tab"
                  aria-selected={item.page_number === page?.page_number}
                  onClick={() => setSelectedPage(item.page_number)}
                  className={cx(
                    "shrink-0 rounded-lg border px-3 py-1.5 text-sm font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-prism-600",
                    item.page_number === page?.page_number
                      ? "border-prism-200 bg-prism-50 text-prism-700"
                      : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50",
                  )}
                >
                  Page {item.page_number}
                  {item.is_empty ? <span className="ml-1 text-xs text-amber-700">(empty)</span> : null}
                </button>
              ))}
            </div>

            {page ? (
              <div role="tabpanel" className="space-y-3">
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                  <Badge tone={page.is_empty ? "warning" : "neutral"}>
                    {page.is_empty ? "Empty page" : `${page.char_count.toLocaleString()} characters`}
                  </Badge>
                  <span>{pageChunks.length} chunk(s) on this page</span>
                </div>
                <pre className="max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-xl border border-slate-200 bg-slate-50 p-4 font-sans text-sm leading-relaxed text-slate-800">
                  {page.text || "(no extractable text)"}
                </pre>
                {pageChunks.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-left text-xs text-slate-600">
                      <thead>
                        <tr className="uppercase tracking-wide text-slate-500">
                          <th scope="col" className="px-2 py-1 font-medium">Chunk</th>
                          <th scope="col" className="px-2 py-1 font-medium">Offsets</th>
                          <th scope="col" className="px-2 py-1 font-medium">Characters</th>
                          <th scope="col" className="px-2 py-1 font-medium">Id</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pageChunks.map((chunk) => (
                          <tr key={chunk.id} className="border-t border-slate-100">
                            <td className="px-2 py-2">{chunk.chunk_index}</td>
                            <td className="px-2 py-2 font-mono">{chunk.start_offset}–{chunk.end_offset}</td>
                            <td className="px-2 py-2">{chunk.char_count}</td>
                            <td className="px-2 py-2 font-mono text-[11px]">{chunk.id}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        )}
      </Card>
    </div>
  );
}
