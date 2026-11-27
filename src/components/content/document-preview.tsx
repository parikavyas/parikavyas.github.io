"use client";

import { useState } from "react";
import { Download, FileText } from "lucide-react";

/**
 * PDF / slide-deck preview. Nothing loads until the reader asks for it, and a
 * plain download link is always available as the accessible fallback.
 */
export function DocumentPreview({ href, title }: { href: string; title: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="my-6 overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <span className="inline-flex items-center gap-2 text-sm">
          <FileText size={16} className="text-accent" aria-hidden="true" /> {title}
        </span>
        <span className="flex gap-2">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent/60 hover:text-accent"
          >
            {open ? "Hide preview" : "Preview"}
          </button>
          <a href={href} className="inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent/60 hover:text-accent" download>
            <Download size={13} aria-hidden="true" /> Download
          </a>
        </span>
      </div>
      {open && (
        <object data={href} type="application/pdf" className="h-[70vh] w-full border-t border-border bg-bg" aria-label={title}>
          <p className="p-4 text-sm text-muted">
            Your browser can’t show this preview. <a href={href} className="text-accent underline">Open the file</a> instead.
          </p>
        </object>
      )}
    </div>
  );
}
