import { DocumentLibrary } from "@/components/document-library";

export default function HomePage() {
  return (
    <div className="space-y-8">
      <section className="max-w-3xl">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-prism-600">Document intelligence</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
          Find the signal in your documents.
        </h1>
        <p className="mt-3 text-base leading-relaxed text-slate-600">
          Upload PDFs, ask questions, and inspect the exact passages behind every answer. Prism keeps retrieval,
          citations, and evaluation visible instead of hiding them behind a chat box.
        </p>
      </section>
      <DocumentLibrary />
    </div>
  );
}
