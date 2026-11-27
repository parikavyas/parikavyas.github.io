"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ScrollspyItem {
  id: string;
  label: string;
  /** 2 = section, 3 = nested sub-section */
  level?: 2 | 3;
}

/**
 * The "reading line": a section becomes current once its top passes this
 * distance from the viewport top (sticky header + a comfortable band).
 */
function activationOffset() {
  const header = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-height")) || 4) * 16;
  return header + Math.min(window.innerHeight * 0.3, 200);
}

function useActiveSection(items: ScrollspyItem[]) {
  const [active, setActive] = useState<string | null>(items[0]?.id ?? null);
  const frame = useRef<number | null>(null);

  const compute = useCallback(() => {
    frame.current = null;
    const offset = activationOffset();
    let current: string | null = items[0]?.id ?? null;
    for (const item of items) {
      const el = document.getElementById(item.id);
      if (!el) continue;
      if (el.getBoundingClientRect().top - offset <= 1) current = item.id;
      else break;
    }
    // At the very bottom of the page, the last section is current even if short.
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2 && items.length) {
      const last = [...items].reverse().find((i) => document.getElementById(i.id));
      if (last) current = last.id;
    }
    setActive(current);
  }, [items]);

  useEffect(() => {
    const schedule = () => {
      if (frame.current === null) frame.current = requestAnimationFrame(compute);
    };
    // IntersectionObserver tells us when any section crosses the activation band…
    const observer = new IntersectionObserver(schedule, {
      rootMargin: `-${activationOffset()}px 0px -40% 0px`,
      threshold: [0, 1],
    });
    items.forEach((i) => {
      const el = document.getElementById(i.id);
      if (el) observer.observe(el);
    });
    // …and a passive, frame-throttled scroll check covers the page bottom.
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("hashchange", schedule);
    schedule();
    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("hashchange", schedule);
      if (frame.current !== null) cancelAnimationFrame(frame.current);
    };
  }, [items, compute]);

  return [active, setActive] as const;
}

function SpyList({
  items,
  active,
  onNavigate,
}: {
  items: ScrollspyItem[];
  active: string | null;
  onNavigate: (id: string) => void;
}) {
  return (
    <ul className="space-y-0.5 border-l border-border">
      {items.map((item) => {
        const isActive = item.id === active;
        return (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              onClick={() => onNavigate(item.id)}
              aria-current={isActive ? "location" : undefined}
              className={cn(
                "-ml-px block border-l-2 py-1.5 text-[0.85rem] leading-snug transition-colors",
                item.level === 3 ? "pl-7" : "pl-4",
                isActive ? "border-accent text-accent" : "border-transparent text-muted hover:text-text",
              )}
            >
              {item.label}
            </a>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Right-hand "On this page" navigation. On large screens it is a sticky
 * sidebar; on smaller screens it collapses into a sticky disclosure menu.
 * Render it once per page (see DocLayout).
 */
export function Scrollspy({ items, title = "On this page" }: { items: ScrollspyItem[]; title?: string }) {
  const [active, setActive] = useActiveSection(items);
  const [open, setOpen] = useState(false);
  if (!items.length) return null;
  const activeLabel = items.find((i) => i.id === active)?.label ?? items[0].label;

  const navigate = (id: string) => {
    setActive(id);
    setOpen(false);
  };

  return (
    <>
      {/* Desktop */}
      <nav aria-label={title} className="hidden lg:block">
        <div className="sticky top-[calc(var(--header-height)+2rem)] max-h-[calc(100vh-var(--header-height)-4rem)] overflow-y-auto pb-6">
          <p className="mb-3 font-mono text-[0.7rem] uppercase tracking-[0.2em] text-muted">{title}</p>
          <SpyList items={items} active={active} onNavigate={navigate} />
        </div>
      </nav>

      {/* Mobile and tablet */}
      <nav
        aria-label={title}
        className="sticky top-[var(--header-height)] z-30 -mx-4 border-b border-border bg-bg/90 px-4 backdrop-blur-md sm:-mx-6 sm:px-6 lg:hidden"
      >
        <button
          type="button"
          className="flex w-full items-center justify-between gap-3 py-3 text-left text-sm"
          aria-expanded={open}
          aria-controls="mobile-toc"
          onClick={() => setOpen((v) => !v)}
        >
          <span className="min-w-0">
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted">{title}</span>
            <span className="ml-3 truncate text-accent">{activeLabel}</span>
          </span>
          <ChevronDown size={18} aria-hidden="true" className={cn("shrink-0 transition-transform", open && "rotate-180")} />
        </button>
        <div id="mobile-toc" hidden={!open} className="max-h-[60vh] overflow-y-auto pb-4">
          <SpyList items={items} active={active} onNavigate={navigate} />
        </div>
      </nav>
    </>
  );
}
