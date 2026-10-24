"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { API_BASE_URL } from "@/lib/api";
import { Badge, Button, cx } from "@/components/ui";
import { useApiKey } from "@/components/api-key-provider";

const LINKS = [
  { href: "/", label: "Library" },
  { href: "/ask", label: "Ask" },
  { href: "/evaluation", label: "Evaluation lab" },
];

export function Nav() {
  const pathname = usePathname();
  const { apiKey, ready, setApiKey, clearApiKey } = useApiKey();
  const [draft, setDraft] = useState("");
  const detailsRef = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    setDraft(apiKey);
  }, [apiKey]);

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
        <Link
          href="/"
          className="flex items-center gap-2.5 rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-prism-600"
        >
          <span
            aria-hidden="true"
            className="h-8 w-8 rounded-lg bg-linear-to-br from-indigo-500 via-sky-400 to-rose-400 shadow-sm"
          />
          <span>
            <span className="block text-base font-semibold leading-tight tracking-tight text-ink">Prism</span>
            <span className="block text-xs leading-tight text-slate-500">Find the signal</span>
          </span>
        </Link>

        <nav aria-label="Main navigation" className="flex flex-1 items-center gap-1">
          {LINKS.map((link) => {
            const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={cx(
                  "rounded-lg px-3 py-2 text-sm font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-prism-600",
                  active ? "bg-prism-50 text-prism-700" : "text-slate-600 hover:bg-slate-100 hover:text-ink",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <details ref={detailsRef} className="relative">
          <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 hover:border-slate-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-prism-600">
            <span aria-hidden="true" className={cx("h-2 w-2 rounded-full", apiKey ? "bg-emerald-500" : "bg-amber-500")} />
            {ready && apiKey ? "API key set" : "Set API key"}
          </summary>
          <div className="absolute right-0 z-30 mt-2 w-72 rounded-xl border border-slate-200 bg-white p-4 shadow-xl">
            <label htmlFor="api-key-input" className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
              Prism API key
            </label>
            <input
              id="api-key-input"
              type="password"
              autoComplete="off"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="dev-key"
              className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-prism-500 focus:outline-none focus:ring-2 focus:ring-prism-100"
            />
            <p className="mt-2 text-xs text-slate-500">
              Sent as <code className="rounded bg-slate-100 px-1">X-API-Key</code> and stored only in this browser.
              API: {API_BASE_URL}
            </p>
            <div className="mt-3 flex items-center gap-2">
              <Button
                onClick={() => {
                  setApiKey(draft);
                  detailsRef.current?.removeAttribute("open");
                }}
              >
                Save
              </Button>
              <Button
                variant="ghost"
                onClick={() => {
                  clearApiKey();
                  setDraft("");
                }}
              >
                Clear
              </Button>
              <Badge tone={apiKey ? "success" : "warning"}>{apiKey ? "Ready" : "Missing"}</Badge>
            </div>
          </div>
        </details>
      </div>
    </header>
  );
}
