"use client";

import { useState } from "react";

import { useApiKey } from "@/components/api-key-provider";
import { ApiError, answer, search } from "@/lib/api";
import { formatMetric } from "@/lib/format";
import type { AnswerResponse, SearchResponse } from "@/lib/types";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorNotice,
  SectionHeading,
  Spinner,
  cx,
} from "@/components/ui";

type Mode = "answer" | "search";
type Result = { kind: "answer"; data: AnswerResponse } | { kind: "search"; data: SearchResponse };

const TOP_K_OPTIONS = [1, 3, 5, 10, 20];

function AnswerText({ text, validMarkers }: { text: string; validMarkers: Set<number> }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
      {parts.map((part, index) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match) return <span key={index}>{part}</span>;
        const marker = Number(match[1]);
        if (!validMarkers.has(marker)) {
          return (
            <span key={index} className="rounded bg-rose-100 px-1 text-xs font-medium text-rose-700">
              {part}
            </span>
          );
        }
        return (
          <a
            key={index}
            href={`#citation-${marker}`}
            className="rounded bg-prism-50 px-1 text-xs font-semibold text-prism-700 hover:bg-prism-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-prism-600"
            aria-label={`Jump to citation ${marker}`}
          >
            {part}
          </a>
        );
      })}
    </p>
  );
}

export function AskPanel() {
  const { apiKey, ready } = useApiKey();
  const [mode, setMode] = useState<Mode>("answer");
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [collectionId, setCollectionId] = useState("default");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiKey) return;
    const trimmed = question.trim();
    if (!trimmed) {
      setError("Enter a question first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      if (mode === "answer") {
        const payload = { question: trimmed, top_k: topK, collection_id: collectionId.trim() || "default" };
        setResult({ kind: "answer", data: await answer(apiKey, payload) });
      } else {
        const payload = { query: trimmed, top_k: topK, collection_id: collectionId.trim() || "default" };
        setResult({ kind: "search", data: await search(apiKey, payload) });
      }
    } catch (cause) {
      if (cause instanceof ApiError) {
        if (cause.code === "answer_generation_unavailable") {
          setError(
            "Answer generation is not configured. Set PRISM_LLM_BASE_URL on the backend, or switch to retrieval-only search.",
          );
        } else {
          setError(cause.message);
        }
      } else {
        setError("The request failed.");
      }
    } finally {
      setLoading(false);
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
          title="Ask your documents"
          description="Answers are generated only from retrieved passages and cite their sources."
        />
        <EmptyState
          title="API key required"
          description="Open “Set API key” in the header and enter a key from PRISM_API_KEYS."
        />
      </Card>
    );
  }

  const validMarkers = new Set((result?.kind === "answer" ? result.data.citations : []).map((c) => c.marker));

  return (
    <div className="space-y-6">
      <Card>
        <SectionHeading
          title="Ask your documents"
          description="Answer mode sends retrieved passages to the configured LLM. Retrieval-only mode shows exactly which chunks scored highest without calling an LLM."
        />
        <form onSubmit={handleSubmit} className="space-y-4">
          <fieldset>
            <legend className="text-sm font-medium text-slate-700">Mode</legend>
            <div className="mt-1.5 inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1">
              {(["answer", "search"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  aria-pressed={mode === option}
                  onClick={() => setMode(option)}
                  className={cx(
                    "rounded-md px-3 py-1.5 text-sm font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-prism-600",
                    mode === option ? "bg-white text-ink shadow-sm" : "text-slate-600 hover:text-ink",
                  )}
                >
                  {option === "answer" ? "Grounded answer" : "Retrieval only"}
                </button>
              ))}
            </div>
          </fieldset>

          <div>
            <label htmlFor="question" className="block text-sm font-medium text-slate-700">
              Question
            </label>
            <textarea
              id="question"
              rows={3}
              value={question}
              maxLength={2000}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="How does the Calvin cycle fix carbon dioxide?"
              className="mt-1.5 w-full resize-y rounded-xl border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
            />
          </div>

          <div className="grid gap-3 sm:grid-cols-[10rem_1fr_auto] sm:items-end">
            <div>
              <label htmlFor="top-k" className="block text-sm font-medium text-slate-700">
                Results (top-k)
              </label>
              <select
                id="top-k"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
              >
                {TOP_K_OPTIONS.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="ask-collection" className="block text-sm font-medium text-slate-700">
                Collection
              </label>
              <input
                id="ask-collection"
                value={collectionId}
                onChange={(event) => setCollectionId(event.target.value)}
                maxLength={128}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
              />
            </div>
            <Button type="submit" disabled={loading}>
              {loading ? "Working…" : mode === "answer" ? "Ask" : "Search"}
            </Button>
          </div>
        </form>
        {loading ? (
          <div className="mt-4" aria-live="polite">
            <Spinner label={mode === "answer" ? "Retrieving evidence and generating an answer" : "Retrieving passages"} />
          </div>
        ) : null}
        {error ? <div className="mt-4"><ErrorNotice message={error} /></div> : null}
      </Card>

      <div aria-live="polite" className="space-y-6">
        {result?.kind === "answer" ? (
          <AnswerResult data={result.data} validMarkers={validMarkers} />
        ) : result?.kind === "search" ? (
          <SearchResults data={result.data} />
        ) : !loading && !error ? (
          <Card>
            <EmptyState
              title="Ask a question to begin"
              description="Prism first retrieves candidate passages, then either generates a cited answer or shows the retrieval results directly."
            />
          </Card>
        ) : null}
      </div>
    </div>
  );
}

function AnswerResult({ data, validMarkers }: { data: AnswerResponse; validMarkers: Set<number> }) {
  return (
    <>
      <Card>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold tracking-tight text-ink">Answer</h2>
          {data.insufficient_evidence ? <Badge tone="warning">Insufficient evidence</Badge> : <Badge tone="success">Evidence-backed</Badge>}
          {data.model ? <Badge tone="neutral">{data.model}</Badge> : null}
        </div>
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <AnswerText text={data.answer} validMarkers={validMarkers} />
        </div>
        {data.invalid_citation_markers.length > 0 ? (
          <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800" role="alert">
            The model emitted citation markers that do not map to retrieved passages:{" "}
            {data.invalid_citation_markers.map((marker) => `[${marker}]`).join(", ")}. They are not shown as sources.
          </div>
        ) : null}
        <dl className="mt-4 grid grid-cols-2 gap-3 text-xs text-slate-500 sm:grid-cols-4">
          <div><dt className="font-medium uppercase tracking-wide">Collection</dt><dd className="mt-0.5 text-slate-700">{data.collection_id}</dd></div>
          <div><dt className="font-medium uppercase tracking-wide">Retrieved</dt><dd className="mt-0.5 text-slate-700">{data.evidence.length} passages</dd></div>
          <div><dt className="font-medium uppercase tracking-wide">Input tokens</dt><dd className="mt-0.5 text-slate-700">{data.input_tokens ?? "—"}</dd></div>
          <div><dt className="font-medium uppercase tracking-wide">Output tokens</dt><dd className="mt-0.5 text-slate-700">{data.output_tokens ?? "—"}</dd></div>
        </dl>
      </Card>

      {data.citations.length > 0 ? (
        <Card>
          <SectionHeading title="Citations" description="Only markers that resolve to an actually retrieved chunk appear here." />
          <ol className="space-y-3">
            {data.citations.map((citation) => (
              <li key={citation.marker} id={`citation-${citation.marker}`} className="rounded-xl border border-slate-200 bg-white p-4">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge tone="info">[{citation.marker}]</Badge>
                  <span className="font-medium text-ink">{citation.filename}</span>
                  <span className="text-slate-500">page {citation.page_number}</span>
                  <span className="text-slate-400">score {formatMetric(citation.score)}</span>
                </div>
                <blockquote className="mt-2 border-l-2 border-prism-200 pl-3 text-sm leading-relaxed text-slate-700">
                  {citation.text}
                </blockquote>
                <p className="mt-2 font-mono text-[11px] text-slate-400">{citation.chunk_id}</p>
              </li>
            ))}
          </ol>
        </Card>
      ) : null}

      <Card>
        <SectionHeading
          title="Retrieved passage viewer"
          description="Why these passages? Each was selected by cosine similarity between the question embedding and the chunk embedding, scoped to your owner and collection."
        />
        {data.evidence.length === 0 ? (
          <EmptyState title="No passages retrieved" description="No indexed chunks matched this question in the selected collection." />
        ) : (
          <ol className="space-y-3">
            {data.evidence.map((item, index) => {
              const cited = data.citations.some((citation) => citation.chunk_id === item.chunk.id);
              return (
                <li key={`${item.chunk.id}-${index}`} className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <Badge tone="neutral">rank {index + 1}</Badge>
                    <Badge tone={cited ? "success" : "neutral"}>{cited ? "cited" : "not cited"}</Badge>
                    <span className="font-medium text-slate-700">{item.filename}</span>
                    <span className="text-slate-500">page {item.chunk.page_number}</span>
                    <span className="ml-auto font-mono text-slate-500">cosine {formatMetric(item.score)}</span>
                  </div>
                  <p className="mt-2 text-sm leading-relaxed text-slate-700">{item.chunk.text}</p>
                  <p className="mt-2 font-mono text-[11px] text-slate-400">
                    {item.chunk.id} · chars {item.chunk.start_offset}–{item.chunk.end_offset}
                  </p>
                </li>
              );
            })}
          </ol>
        )}
      </Card>
    </>
  );
}

function SearchResults({ data }: { data: SearchResponse }) {
  return (
    <Card>
      <SectionHeading
        title={`Retrieval results for “${data.query}”`}
        description={`Top ${data.top_k} chunks in collection “${data.collection_id}”. Scores are cosine similarities from the transparent NumPy index.`}
      />
      {data.results.length === 0 ? (
        <EmptyState title="Nothing matched" description="Try another phrasing, a different collection, or upload more documents." />
      ) : (
        <ol className="space-y-3">
          {data.results.map((item, index) => (
            <li key={`${item.chunk.id}-${index}`} className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <Badge tone="info">rank {index + 1}</Badge>
                <span className="font-medium text-slate-700">{item.filename}</span>
                <span className="text-slate-500">page {item.chunk.page_number}</span>
                <span className="ml-auto font-mono text-slate-500">cosine {formatMetric(item.score)}</span>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-700">{item.chunk.text}</p>
              <p className="mt-2 font-mono text-[11px] text-slate-400">
                {item.chunk.id} · chunk {item.chunk.chunk_index} · chars {item.chunk.start_offset}–{item.chunk.end_offset}
              </p>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
