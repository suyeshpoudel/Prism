"use client";

import { useCallback, useEffect, useState } from "react";

import { useApiKey } from "@/components/api-key-provider";
import {
  ApiError,
  getBenchmark,
  getEvaluationRun,
  listEvaluationRuns,
  runAnswerEvaluation,
  runRetrievalEvaluation,
} from "@/lib/api";
import {
  formatCost,
  formatDate,
  formatMetric,
  formatMs,
  formatPercent,
} from "@/lib/format";
import type {
  AnswerCaseReport,
  BenchmarkInfo,
  EvaluationRecord,
  RetrievalConfiguration,
  RetrievalReport,
  RunSummary,
} from "@/lib/types";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorNotice,
  SectionHeading,
  Spinner,
  Stat,
} from "@/components/ui";

interface ConfigRow extends RetrievalConfiguration {
  key: number;
  embedding_model: string;
}

const DEFAULT_CONFIGS: ConfigRow[] = [
  { key: 1, chunk_size: 1000, chunk_overlap: 150, embedding_model: "" },
];

function rateTone(value: number | null | undefined): "success" | "warning" | "danger" | "neutral" {
  if (value === null || value === undefined) return "neutral";
  if (value >= 0.8) return "success";
  if (value >= 0.5) return "warning";
  return "danger";
}

export function EvaluationLab() {
  const { apiKey, ready } = useApiKey();

  const [benchmark, setBenchmark] = useState<BenchmarkInfo | null>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  const [configs, setConfigs] = useState<ConfigRow[]>(DEFAULT_CONFIGS);
  const [kValues, setKValues] = useState<number[]>([1, 3, 5]);
  const [retrievalLoading, setRetrievalLoading] = useState(false);
  const [retrievalError, setRetrievalError] = useState<string | null>(null);
  const [retrievalRecord, setRetrievalRecord] = useState<EvaluationRecord | null>(null);

  const [answerEmbeddingModel, setAnswerEmbeddingModel] = useState("");
  const [answerChunkSize, setAnswerChunkSize] = useState(1000);
  const [answerOverlap, setAnswerOverlap] = useState(150);
  const [answerTopK, setAnswerTopK] = useState(5);
  const [useJudge, setUseJudge] = useState(false);
  const [answerLoading, setAnswerLoading] = useState(false);
  const [answerError, setAnswerError] = useState<string | null>(null);
  const [answerRecord, setAnswerRecord] = useState<EvaluationRecord | null>(null);

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [loadingRunId, setLoadingRunId] = useState<string | null>(null);

  const refreshRuns = useCallback(async () => {
    if (!apiKey) return;
    setRunsLoading(true);
    setRunsError(null);
    try {
      setRuns(await listEvaluationRuns(apiKey));
    } catch (cause) {
      setRunsError(cause instanceof ApiError ? cause.message : "Could not load saved runs.");
    } finally {
      setRunsLoading(false);
    }
  }, [apiKey]);

  useEffect(() => {
    if (!ready || !apiKey) return;
    const controller = new AbortController();
    setBenchmarkLoading(true);
    setBenchmarkError(null);
    getBenchmark(apiKey)
      .then(setBenchmark)
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setBenchmarkError(cause instanceof ApiError ? cause.message : "Could not load the benchmark.");
      })
      .finally(() => setBenchmarkLoading(false));
    refreshRuns();
    return () => controller.abort();
  }, [apiKey, ready, refreshRuns]);

  function updateConfig(key: number, patch: Partial<ConfigRow>) {
    setConfigs((current) => current.map((row) => (row.key === key ? { ...row, ...patch } : row)));
  }

  function addConfig() {
    setConfigs((current) => {
      if (current.length >= 8) return current;
      const nextKey = Math.max(...current.map((row) => row.key)) + 1;
      return [...current, { key: nextKey, chunk_size: 800, chunk_overlap: 100, embedding_model: "" }];
    });
  }

  async function handleRetrievalRun(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiKey) return;
    if (kValues.length === 0) {
      setRetrievalError("Select at least one k value.");
      return;
    }
    for (const row of configs) {
      if (row.chunk_overlap >= row.chunk_size) {
        setRetrievalError("Chunk overlap must be smaller than chunk size for every configuration.");
        return;
      }
    }
    setRetrievalLoading(true);
    setRetrievalError(null);
    try {
      const record = await runRetrievalEvaluation(apiKey, {
        configurations: configs.map(({ chunk_size, chunk_overlap, embedding_model }) => ({
          chunk_size,
          chunk_overlap,
          embedding_model: embedding_model.trim() || undefined,
        })),
        k_values: kValues,
        persist: true,
      });
      setRetrievalRecord(record);
      await refreshRuns();
    } catch (cause) {
      setRetrievalError(cause instanceof ApiError ? cause.message : "The retrieval evaluation failed.");
    } finally {
      setRetrievalLoading(false);
    }
  }

  async function handleAnswerRun(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiKey) return;
    if (answerOverlap >= answerChunkSize) {
      setAnswerError("Chunk overlap must be smaller than chunk size.");
      return;
    }
    setAnswerLoading(true);
    setAnswerError(null);
    try {
      const record = await runAnswerEvaluation(apiKey, {
        chunk_size: answerChunkSize,
        chunk_overlap: answerOverlap,
        top_k: answerTopK,
        include_llm_judge: useJudge,
        embedding_model: answerEmbeddingModel.trim() || undefined,
      });
      setAnswerRecord(record);
      await refreshRuns();
    } catch (cause) {
      if (cause instanceof ApiError && cause.code === "answer_generation_unavailable") {
        setAnswerError("Answer evaluation needs an LLM. Set PRISM_LLM_BASE_URL on the backend first.");
      } else {
        setAnswerError(cause instanceof ApiError ? cause.message : "The answer evaluation failed.");
      }
    } finally {
      setAnswerLoading(false);
    }
  }

  async function loadRun(run: RunSummary) {
    if (!apiKey) return;
    setLoadingRunId(run.id);
    setRunsError(null);
    try {
      const record = await getEvaluationRun(apiKey, run.id);
      if (record.type === "answer") {
        setAnswerRecord(record);
      } else {
        setRetrievalRecord(record);
      }
    } catch (cause) {
      setRunsError(cause instanceof ApiError ? cause.message : "Could not load that run.");
    } finally {
      setLoadingRunId(null);
    }
  }

  if (!ready) {
    return (
      <Card>
        <Spinner label="Loading evaluation lab" />
      </Card>
    );
  }

  if (!apiKey) {
    return (
      <Card>
        <SectionHeading
          title="Retrieval evaluation lab"
          description="Run reproducible retrieval and answer benchmarks against the versioned Prism corpus."
        />
        <EmptyState
          title="API key required"
          description="Open “Set API key” in the header to run and inspect evaluations."
        />
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <SectionHeading
          title="Benchmark corpus"
          description="Version-controlled documents and questions with expected (filename, page) evidence. Metrics are deterministic unless an LLM judge is explicitly enabled."
        />
        {benchmarkLoading ? <Spinner label="Loading benchmark" /> : null}
        {benchmarkError ? <ErrorNotice title="Benchmark unavailable" message={benchmarkError} /> : null}
        {benchmark ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Documents" value={benchmark.documents.length} />
              <Stat label="Pages" value={benchmark.documents.reduce((sum, doc) => sum + doc.page_count, 0)} />
              <Stat label="Questions" value={benchmark.cases.length} />
              <Stat label="Metric type" value="Deterministic" hint="Optional LLM judge is separate" />
            </div>
            <details className="rounded-xl border border-slate-200 bg-slate-50/60 p-3 text-sm">
              <summary className="cursor-pointer font-medium text-slate-700">Inspect corpus and questions</summary>
              <div className="mt-3 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Documents</p>
                  <ul className="mt-1 space-y-1 text-slate-700">
                    {benchmark.documents.map((document) => (
                      <li key={document.filename}>
                        {document.filename} · {document.page_count} pages · {document.char_count.toLocaleString()} chars
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Questions</p>
                  <ul className="mt-1 space-y-2 text-slate-700">
                    {benchmark.cases.map((item) => (
                      <li key={item.id}>
                        <span className="font-medium">{item.question}</span>
                        <span className="block text-xs text-slate-500">
                          expected: {item.relevant.map((key) => `${key.filename} p.${key.page_number}`).join(", ")}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </details>
          </div>
        ) : null}
      </Card>

      <Card>
        <SectionHeading
          title="Retrieval comparison"
          description="Compare chunk sizes, overlaps, and embedding models. Each configuration re-embeds the same corpus and runs the same questions."
        />
        <form onSubmit={handleRetrievalRun} className="space-y-4">
          <div className="space-y-3">
            {configs.map((row, index) => (
              <fieldset key={row.key} className="rounded-xl border border-slate-200 bg-slate-50/60 p-3">
                <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Configuration {index + 1}
                </legend>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[2fr_1fr_1fr_auto] lg:items-end">
                  <div>
                    <label htmlFor={`embedding-model-${row.key}`} className="block text-sm font-medium text-slate-700">
                      Embedding model
                    </label>
                    <input
                      id={`embedding-model-${row.key}`}
                      type="text"
                      value={row.embedding_model}
                      onChange={(event) => updateConfig(row.key, { embedding_model: event.target.value })}
                      placeholder="Server default"
                      className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
                    />
                  </div>
                  <div>
                    <label htmlFor={`chunk-size-${row.key}`} className="block text-sm font-medium text-slate-700">
                      Chunk size
                    </label>
                    <input
                      id={`chunk-size-${row.key}`}
                      type="number"
                      min={100}
                      max={5000}
                      value={row.chunk_size}
                      onChange={(event) => updateConfig(row.key, { chunk_size: Number(event.target.value) })}
                      className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
                    />
                  </div>
                  <div>
                    <label htmlFor={`overlap-${row.key}`} className="block text-sm font-medium text-slate-700">
                      Overlap
                    </label>
                    <input
                      id={`overlap-${row.key}`}
                      type="number"
                      min={0}
                      max={4999}
                      value={row.chunk_overlap}
                      onChange={(event) => updateConfig(row.key, { chunk_overlap: Number(event.target.value) })}
                      className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
                    />
                  </div>
                  <Button
                    variant="danger"
                    onClick={() => setConfigs((current) => current.filter((item) => item.key !== row.key))}
                    disabled={configs.length === 1}
                    aria-label={`Remove configuration ${index + 1}`}
                  >
                    Remove
                  </Button>
                </div>
              </fieldset>
            ))}
          </div>

          <div className="flex flex-wrap items-end gap-4">
            <fieldset>
              <legend className="text-sm font-medium text-slate-700">k values</legend>
              <div className="mt-1.5 flex flex-wrap gap-2">
                {[1, 3, 5, 10].map((value) => (
                  <label key={value} className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-sm text-slate-700">
                    <input
                      type="checkbox"
                      checked={kValues.includes(value)}
                      onChange={(event) =>
                        setKValues((current) =>
                          event.target.checked
                            ? [...current, value].sort((a, b) => a - b)
                            : current.filter((item) => item !== value),
                        )
                      }
                      className="h-4 w-4 rounded border-slate-300 text-prism-600 focus:ring-prism-500"
                    />
                    @{value}
                  </label>
                ))}
              </div>
            </fieldset>
            <Button variant="secondary" onClick={addConfig} disabled={configs.length >= 8}>
              Add configuration
            </Button>
            <Button type="submit" disabled={retrievalLoading}>
              {retrievalLoading ? "Running…" : "Run retrieval evaluation"}
            </Button>
          </div>
        </form>
        {retrievalLoading ? (
          <p className="mt-3 text-sm text-slate-600" aria-live="polite">
            Embedding the corpus and running all questions. The first run may download the embedding model.
          </p>
        ) : null}
        {retrievalError ? <div className="mt-4"><ErrorNotice title="Evaluation failed" message={retrievalError} /></div> : null}
        {retrievalRecord ? <div className="mt-5"><RetrievalComparison record={retrievalRecord} /></div> : null}
      </Card>

      <Card>
        <SectionHeading
          title="Answer and citation evaluation"
          description="Generate an answer for every benchmark question, then score citation validity, citation precision/recall, structural grounding proxies, token usage, and latency."
        />
        <form onSubmit={handleAnswerRun} className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label htmlFor="answer-embedding-model" className="block text-sm font-medium text-slate-700">Embedding model</label>
              <input
                id="answer-embedding-model"
                type="text"
                value={answerEmbeddingModel}
                onChange={(event) => setAnswerEmbeddingModel(event.target.value)}
                placeholder="Server default"
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
              />
            </div>
            <div>
              <label htmlFor="answer-chunk-size" className="block text-sm font-medium text-slate-700">Chunk size</label>
              <input
                id="answer-chunk-size"
                type="number"
                min={100}
                max={5000}
                value={answerChunkSize}
                onChange={(event) => setAnswerChunkSize(Number(event.target.value))}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
              />
            </div>
            <div>
              <label htmlFor="answer-overlap" className="block text-sm font-medium text-slate-700">Overlap</label>
              <input
                id="answer-overlap"
                type="number"
                min={0}
                max={4999}
                value={answerOverlap}
                onChange={(event) => setAnswerOverlap(Number(event.target.value))}
                className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
              />
            </div>
            <div>
              <label htmlFor="answer-top-k" className="block text-sm font-medium text-slate-700">Top-k evidence</label>
              <select
                id="answer-top-k"
                value={answerTopK}
                onChange={(event) => setAnswerTopK(Number(event.target.value))}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
              >
                {[1, 3, 5, 10, 20].map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </div>
          </div>
          <label className="flex items-start gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={useJudge}
              onChange={(event) => setUseJudge(event.target.checked)}
              className="mt-0.5 h-4 w-4 rounded border-slate-300 text-prism-600 focus:ring-prism-500"
            />
            <span>
              Also run the optional LLM faithfulness judge.
              <span className="block text-xs text-slate-500">
                Subjective and not ground truth; the same configured model may judge its own answer. Reported separately.
              </span>
            </span>
          </label>
          <Button type="submit" disabled={answerLoading}>
            {answerLoading ? "Running…" : "Run answer evaluation"}
          </Button>
        </form>
        {answerLoading ? (
          <p className="mt-3 text-sm text-slate-600" aria-live="polite">
            Calling the configured LLM once per question{useJudge ? " (plus judge calls)" : ""}. This may take a while and incur provider cost.
          </p>
        ) : null}
        {answerError ? <div className="mt-4"><ErrorNotice title="Answer evaluation unavailable" message={answerError} /></div> : null}
        {answerRecord ? <div className="mt-5"><AnswerReport record={answerRecord} /></div> : null}
      </Card>

      <Card>
        <SectionHeading
          title="Saved runs"
          description="Reports are stored as JSON on the server and scoped to your owner id."
          action={<Button variant="secondary" onClick={refreshRuns} disabled={runsLoading}>Refresh</Button>}
        />
        {runsError ? <ErrorNotice title="Could not load saved runs" message={runsError} /> : null}
        {runsLoading && runs.length === 0 ? <Spinner label="Loading saved runs" /> : null}
        {!runsLoading && runs.length === 0 && !runsError ? (
          <EmptyState title="No saved runs" description="Run a retrieval or answer evaluation to create a reproducible report." />
        ) : runs.length > 0 ? (
          <ul className="divide-y divide-slate-100">
            {runs.map((run) => (
              <li key={run.id} className="flex flex-wrap items-center gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={run.type === "answer" ? "info" : "neutral"}>{run.type ?? "run"}</Badge>
                    <span className="text-sm font-medium text-slate-800">{formatDate(run.created_at)}</span>
                  </div>
                  <p className="mt-0.5 truncate font-mono text-[11px] text-slate-400">{run.id}</p>
                </div>
                <Button variant="secondary" onClick={() => loadRun(run)} disabled={loadingRunId === run.id}>
                  {loadingRunId === run.id ? "Loading…" : "View report"}
                </Button>
              </li>
            ))}
          </ul>
        ) : null}
      </Card>
    </div>
  );
}

function RetrievalComparison({ record }: { record: EvaluationRecord }) {
  const rows = record.comparison ?? [];
  if (rows.length === 0) return null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-base font-semibold text-ink">Retrieval results</h3>
        <Badge tone="neutral">{record.id}</Badge>
        <Badge tone="neutral">{formatDate(record.created_at)}</Badge>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead>
            <tr className="text-xs uppercase tracking-wide text-slate-500">
              <th scope="col" className="px-2 py-2 font-medium">Model</th>
              <th scope="col" className="px-2 py-2 font-medium">Chunk</th>
              <th scope="col" className="px-2 py-2 font-medium">Overlap</th>
              <th scope="col" className="px-2 py-2 font-medium">Recall@1</th>
              <th scope="col" className="px-2 py-2 font-medium">Recall@3</th>
              <th scope="col" className="px-2 py-2 font-medium">Recall@5</th>
              <th scope="col" className="px-2 py-2 font-medium">MRR</th>
              <th scope="col" className="px-2 py-2 font-medium">Query ms</th>
              <th scope="col" className="px-2 py-2 font-medium">Chunks</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index} className="border-t border-slate-100 text-slate-700">
                <td className="max-w-[15rem] truncate px-2 py-2 font-mono text-xs">{String(row.embedding_model ?? "—")}</td>
                <td className="px-2 py-2">{String(row.chunk_size ?? "—")}</td>
                <td className="px-2 py-2">{String(row.chunk_overlap ?? "—")}</td>
                <td className="px-2 py-2">{formatPercent(row["recall@1"] ?? null)}</td>
                <td className="px-2 py-2">{formatPercent(row["recall@3"] ?? null)}</td>
                <td className="px-2 py-2">{formatPercent(row["recall@5"] ?? null)}</td>
                <td className="px-2 py-2">{formatMetric(row.mrr ?? null)}</td>
                <td className="px-2 py-2">{formatMs(row.query_ms_mean ?? null)}</td>
                <td className="px-2 py-2">{String(row.corpus_chunks ?? "—")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-2">
        {(record.results ?? []).map((report: RetrievalReport, index: number) => (
          <details key={index} className="rounded-xl border border-slate-200 bg-slate-50/60 p-3">
            <summary className="cursor-pointer text-sm font-medium text-slate-700">
              Case details · chunk {report.config.chunk_size}/{report.config.chunk_overlap} · {report.config.embedding_model}
            </summary>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full text-left text-xs text-slate-600">
                <thead>
                  <tr className="uppercase tracking-wide text-slate-500">
                    <th scope="col" className="px-2 py-1 font-medium">Question</th>
                    {report.config.k_values.map((k) => (
                      <th key={k} scope="col" className="px-2 py-1 font-medium">Recall@{k}</th>
                    ))}
                    <th scope="col" className="px-2 py-1 font-medium">MRR</th>
                    <th scope="col" className="px-2 py-1 font-medium">Top result</th>
                  </tr>
                </thead>
                <tbody>
                  {report.cases.map((item) => (
                    <tr key={item.id} className="border-t border-slate-200/70">
                      <td className="max-w-sm px-2 py-2">{item.question}</td>
                      {report.config.k_values.map((k) => (
                        <td key={k} className="px-2 py-2">{formatPercent(item.metrics[`recall@${k}`] ?? null)}</td>
                      ))}
                      <td className="px-2 py-2">{formatMetric(item.metrics.mrr ?? null)}</td>
                      <td className="px-2 py-2">
                        {item.top_results[0]
                          ? `${item.top_results[0].filename} p.${item.top_results[0].page_number} (${formatMetric(item.top_results[0].score)})`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}

function AnswerReport({ record }: { record: EvaluationRecord }) {
  const aggregate = record.aggregate ?? {};
  const cases = record.cases ?? [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-base font-semibold text-ink">Answer results</h3>
        <Badge tone="neutral">{record.id}</Badge>
        <Badge tone="neutral">{formatDate(record.created_at)}</Badge>
        <Badge tone={Number(aggregate.generation_errors ?? 0) > 0 ? "danger" : "success"}>
          {String(aggregate.generation_errors ?? 0)} generation errors
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Citation validity" value={formatPercent(aggregate.citation_validity_mean)} />
        <Stat label="Citation precision" value={formatPercent(aggregate.citation_precision_mean)} />
        <Stat label="Citation recall" value={formatPercent(aggregate.citation_recall_mean)} />
        <Stat label="Lexical grounding" value={formatPercent(aggregate.lexical_grounding_rate_mean)} />
        <Stat label="Uncited sentences" value={formatPercent(aggregate.uncited_sentence_rate_mean)} />
        <Stat label="Insufficient evidence" value={formatPercent(aggregate.insufficient_evidence_rate)} />
        <Stat label="Judge faithful" value={formatPercent(aggregate.llm_judge_faithful_rate)} hint={`${aggregate.llm_judge_cases ?? 0} judged`} />
        <Stat label="Query latency" value={formatMs(
          typeof aggregate.embedding_ms_mean === "number" && typeof aggregate.retrieval_ms_mean === "number" && typeof aggregate.generation_ms_mean === "number"
            ? aggregate.embedding_ms_mean + aggregate.retrieval_ms_mean + aggregate.generation_ms_mean
            : null,
        )} />
        <Stat label="Input tokens" value={record.usage?.input_tokens ?? "—"} />
        <Stat label="Output tokens" value={record.usage?.output_tokens ?? "—"} />
        <Stat label="Estimated cost" value={formatCost(record.usage?.estimated_cost_usd ?? null)} hint="Requires configured rates" />
        <Stat label="Answers with citations" value={formatPercent(aggregate.answers_with_citations_rate)} />
      </div>

      <details className="rounded-xl border border-slate-200 bg-slate-50/60 p-3 text-sm">
        <summary className="cursor-pointer font-medium text-slate-700">How these metrics are computed</summary>
        <dl className="mt-3 space-y-2 text-xs text-slate-600">
          {Object.entries(record.method_notes ?? {}).map(([name, note]) => (
            <div key={name}>
              <dt className="font-semibold text-slate-700">{name.replaceAll("_", " ")}</dt>
              <dd>{note}</dd>
            </div>
          ))}
        </dl>
      </details>

      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead>
            <tr className="text-xs uppercase tracking-wide text-slate-500">
              <th scope="col" className="px-2 py-2 font-medium">Question</th>
              <th scope="col" className="px-2 py-2 font-medium">Citations</th>
              <th scope="col" className="px-2 py-2 font-medium">Validity</th>
              <th scope="col" className="px-2 py-2 font-medium">Precision</th>
              <th scope="col" className="px-2 py-2 font-medium">Recall</th>
              <th scope="col" className="px-2 py-2 font-medium">Grounding</th>
              <th scope="col" className="px-2 py-2 font-medium">Uncited</th>
              <th scope="col" className="px-2 py-2 font-medium">Judge</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((item) => (
              <tr key={item.id} className="border-t border-slate-100 text-slate-700">
                <td className="max-w-sm px-2 py-2">
                  <details>
                    <summary className="cursor-pointer text-slate-800">{item.question}</summary>
                    <CaseDetail item={item} />
                  </details>
                </td>
                <td className="px-2 py-2">{item.citation_details?.valid_markers.length ?? 0}</td>
                <td className="px-2 py-2"><Badge tone={rateTone(item.metrics.citation_validity)}>{formatPercent(item.metrics.citation_validity)}</Badge></td>
                <td className="px-2 py-2">{formatPercent(item.metrics.citation_precision)}</td>
                <td className="px-2 py-2">{formatPercent(item.metrics.citation_recall)}</td>
                <td className="px-2 py-2">{formatPercent(item.metrics.lexical_grounding_rate)}</td>
                <td className="px-2 py-2">{formatPercent(item.metrics.uncited_sentence_rate)}</td>
                <td className="px-2 py-2">
                  {item.llm_judge?.faithful === undefined ? "—" : (
                    <Badge tone={item.llm_judge.faithful ? "success" : "danger"}>
                      {item.llm_judge.faithful ? "faithful" : "unsupported"}
                    </Badge>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CaseDetail({ item }: { item: AnswerCaseReport }) {
  return (
    <div className="mt-2 max-w-2xl space-y-2 pb-2 text-xs text-slate-600">
      <p className="whitespace-pre-wrap rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-800">
        {item.answer ?? "(no answer)"}
      </p>
      <div className="flex flex-wrap gap-2">
        <Badge tone={item.insufficient_evidence ? "warning" : "success"}>
          {item.insufficient_evidence ? "insufficient evidence" : "answered"}
        </Badge>
        {item.model ? <Badge tone="neutral">{item.model}</Badge> : null}
        {item.citation_details?.invalid_markers.length ? (
          <Badge tone="danger">
            invalid markers: {item.citation_details.invalid_markers.map((marker) => `[${marker}]`).join(", ")}
          </Badge>
        ) : null}
        {item.error ? <Badge tone="danger">{item.error}</Badge> : null}
      </div>
      {item.citation_details?.cited_keys.length ? (
        <p>
          Cited: {item.citation_details.cited_keys.map((key) => `${key.filename} p.${key.page_number}`).join(", ")}
        </p>
      ) : null}
      {item.llm_judge ? (
        <p>
          <span className="font-semibold text-slate-700">LLM judge:</span>{" "}
          {item.llm_judge.error
            ? item.llm_judge.error
            : `${item.llm_judge.faithful ? "faithful" : "not faithful"} — ${item.llm_judge.reason ?? ""}`}
        </p>
      ) : null}
      {item.top_results?.length ? (
        <p>
          Top passage: {item.top_results[0].filename} p.{item.top_results[0].page_number} · cosine{" "}
          {formatMetric(item.top_results[0].score)}
        </p>
      ) : null}
      {item.timings_ms ? (
        <p>
          Embedding {formatMs(item.timings_ms.embedding)} · retrieval {formatMs(item.timings_ms.retrieval)} · generation{" "}
          {formatMs(item.timings_ms.generation)}
        </p>
      ) : null}
    </div>
  );
}
