import { AskPanel } from "@/components/ask-panel";

export default function AskPage() {
  return (
    <div className="space-y-6">
      <section className="max-w-3xl">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-prism-600">Evidence first</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink">Ask your documents</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Prism retrieves passages before answering. Every citation points at a real retrieved chunk, and you can inspect
          the passages the model did and did not cite.
        </p>
      </section>
      <AskPanel />
    </div>
  );
}
