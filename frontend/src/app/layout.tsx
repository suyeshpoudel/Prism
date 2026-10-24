import type { Metadata } from "next";

import "./globals.css";
import { ApiKeyProvider } from "@/components/api-key-provider";
import { Nav } from "@/components/nav";

export const metadata: Metadata = {
  title: "Prism — find the signal in your documents",
  description:
    "Upload PDFs, ask evidence-backed questions, inspect cited passages, and evaluate retrieval quality.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <ApiKeyProvider>
          <a
            href="#main"
            className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2 focus:shadow"
          >
            Skip to content
          </a>
          <Nav />
          <main id="main" className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
            {children}
          </main>
          <footer className="mx-auto w-full max-w-6xl px-4 pb-10 pt-4 text-xs text-slate-500 sm:px-6">
            Prism answers only from retrieved evidence. Treat generated answers and uploaded documents as untrusted.
          </footer>
        </ApiKeyProvider>
      </body>
    </html>
  );
}
