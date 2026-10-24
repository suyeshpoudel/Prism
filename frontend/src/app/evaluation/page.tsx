import { EvaluationLab } from "@/components/evaluation-lab";

export default function EvaluationPage() {
  return (
    <div className="space-y-6">
      <section className="max-w-3xl">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-prism-600">Measurement, not vibes</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink">Retrieval evaluation lab</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Compare chunking and retrieval settings against a versioned benchmark, then measure citation correctness and
          grounding proxies. Deterministic metrics are kept separate from the optional, subjective LLM judge.
        </p>
      </section>
      <EvaluationLab />
    </div>
  );
}
