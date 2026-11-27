"use client";

import Image from "next/image";
import Link from "next/link";
import { useId, useRef, useState } from "react";
import { ArrowUpRight, RotateCcw, RotateCw } from "lucide-react";
import type { Experience } from "@/lib/schemas";
import { cn } from "@/lib/utils";

export interface ExperienceCardProps {
  experience: Experience;
  period: string | null;
  related: { slug: string; title: string }[];
  /** Development-only placeholder hints (null in production). */
  devHints: boolean;
}

const detailSections: { key: keyof Experience["details"]; label: string }[] = [
  { key: "problems", label: "Problems addressed" },
  { key: "built", label: "What I built or contributed" },
  { key: "owned", label: "What I owned" },
  { key: "decisions", label: "Technical & product decisions" },
  { key: "leadership", label: "Leadership & collaboration" },
  { key: "stakeholders", label: "Client & stakeholder engagement" },
  { key: "impact", label: "Selected impact" },
];

function Hint({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-block rounded border border-dashed border-warm/50 bg-warm/5 px-1.5 py-0.5 font-mono text-[0.7rem] text-warm">
      {children}
    </span>
  );
}

/**
 * Two-sided experience card. Flips on click/tap or with the explicit button
 * (never on hover). Both faces share one grid cell, so the card grows to fit
 * the longer side instead of clipping it. The hidden face is `inert` so
 * keyboard and screen-reader users only meet the visible side.
 */
export function ExperienceCard({ experience: e, period, related, devHints }: ExperienceCardProps) {
  const [flipped, setFlipped] = useState(false);
  const frontBtn = useRef<HTMLButtonElement>(null);
  const backBtn = useRef<HTMLButtonElement>(null);
  const id = useId();
  const hasDetails = detailSections.some((s) => e.details[s.key].length > 0) || related.length > 0;

  const flip = (next: boolean) => {
    setFlipped(next);
    // Move focus to the control on the newly visible face.
    requestAnimationFrame(() => (next ? backBtn : frontBtn).current?.focus({ preventScroll: true }));
  };

  // Clicking empty card space flips too (mouse/touch convenience); links and buttons keep their own behaviour.
  const onSurfaceClick = (ev: React.MouseEvent) => {
    if ((ev.target as HTMLElement).closest("a,button")) return;
    if (window.getSelection()?.toString()) return;
    flip(!flipped);
  };

  return (
    <article className="flip-scene h-full" aria-labelledby={`${id}-org`}>
      <div className="flip-inner h-full" data-flipped={flipped}>
        {/* ---------- Front ---------- */}
        <div
          className="flip-face flip-front flex cursor-pointer flex-col rounded-[var(--radius-card)] border border-border bg-surface p-6 transition-colors hover:border-accent/45"
          inert={flipped}
          onClick={onSurfaceClick}
        >
          <div className="flex items-start gap-4">
            <div className="grid h-12 w-12 shrink-0 place-items-center overflow-hidden rounded-lg border border-border bg-bg font-display text-sm text-blue-soft">
              {e.logo ? (
                <Image src={e.logo} alt="" width={48} height={48} className="h-full w-full object-contain p-1.5" />
              ) : (
                <span aria-hidden="true">{e.organization.replace(/[^A-Za-z ]/g, "").split(" ").filter(Boolean).slice(0, 2).map((w) => w[0]).join("")}</span>
              )}
            </div>
            <div className="min-w-0">
              <h3 id={`${id}-org`} className="font-display text-xl font-semibold leading-tight">
                {e.organization}
              </h3>
              <p className="mt-1 text-[0.95rem] text-text/90">
                {e.role ?? (devHints ? <Hint>role — add in src/data/experience.ts</Hint> : null)}
              </p>
              <p className="mt-1 font-mono text-xs text-muted">
                {period ?? (devHints ? <Hint>dates</Hint> : null)}
                {e.location && <span> · {e.location}</span>}
              </p>
            </div>
          </div>

          <div className="mt-5 flex-1 text-[0.95rem] leading-relaxed text-muted">
            {e.summary ?? (devHints ? <Hint>summary — one or two sentences</Hint> : <span className="italic">Details coming soon.</span>)}
          </div>

          {e.themes.length > 0 && (
            <ul className="mt-5 flex flex-wrap gap-1.5" aria-label="Themes">
              {e.themes.map((t) => (
                <li key={t} className="rounded-md border border-blue/40 bg-blue/10 px-2 py-0.5 font-mono text-[0.7rem] text-blue-soft">
                  {t}
                </li>
              ))}
            </ul>
          )}

          <div className="mt-6 flex items-center justify-between border-t border-border/70 pt-4">
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted">
              {hasDetails ? "More on the back" : "Details to come"}
            </span>
            <button
              ref={frontBtn}
              type="button"
              onClick={() => flip(true)}
              aria-expanded={flipped}
              aria-controls={`${id}-back`}
              className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-text transition-colors hover:border-accent/60 hover:text-accent"
            >
              <RotateCw size={14} aria-hidden="true" />
              Flip card<span className="sr-only">: details for {e.organization}</span>
            </button>
          </div>
        </div>

        {/* ---------- Back ---------- */}
        <div
          id={`${id}-back`}
          className="flip-face flip-back flex cursor-pointer flex-col rounded-[var(--radius-card)] border border-accent/35 bg-surface-elevated/70 p-6"
          inert={!flipped}
          onClick={onSurfaceClick}
        >
          <p className="font-mono text-[0.7rem] uppercase tracking-[0.18em] text-accent">{e.organization}</p>
          <div className="mt-4 flex-1 space-y-4">
            {detailSections.map(({ key, label }) => {
              const items = e.details[key];
              if (!items.length) return null;
              return (
                <div key={key}>
                  <h4 className="font-display text-sm font-semibold text-text">{label}</h4>
                  <ul className="mt-1.5 list-disc space-y-1 pl-4 text-[0.9rem] leading-relaxed text-muted marker:text-blue-soft">
                    {items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              );
            })}
            {!detailSections.some((s) => e.details[s.key].length > 0) &&
              (devHints ? (
                <Hint>details — fill `details` (problems, built, owned, decisions, leadership, stakeholders, impact)</Hint>
              ) : (
                <p className="text-sm italic text-muted">A fuller account of this work is being written.</p>
              ))}
            {related.length > 0 && (
              <div>
                <h4 className="font-display text-sm font-semibold text-text">Related projects</h4>
                <ul className="mt-1.5 space-y-1">
                  {related.map((r) => (
                    <li key={r.slug}>
                      <Link href={`/projects/${r.slug}`} className="inline-flex items-center gap-1 text-sm text-accent hover:underline">
                        {r.title} <ArrowUpRight size={13} aria-hidden="true" />
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <div className="mt-6 flex justify-end border-t border-border/70 pt-4">
            <button
              ref={backBtn}
              type="button"
              onClick={() => flip(false)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-text transition-colors hover:border-accent/60 hover:text-accent",
              )}
            >
              <RotateCcw size={14} aria-hidden="true" />
              Back to summary<span className="sr-only"> for {e.organization}</span>
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}
