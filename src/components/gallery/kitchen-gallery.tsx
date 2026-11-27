"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { KitchenItem } from "@/lib/schemas";
import { formatDate } from "@/lib/dates";

/**
 * Masonry-style photo grid with an accessible lightbox built on the native
 * <dialog> element (focus is trapped, Esc closes, focus returns to the photo).
 * Arrow keys move between photos.
 */
export function KitchenGallery({ items }: { items: KitchenItem[] }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const triggers = useRef<(HTMLButtonElement | null)[]>([]);
  const [index, setIndex] = useState<number | null>(null);

  const open = (i: number) => {
    setIndex(i);
    dialog.current?.showModal();
  };
  const close = useCallback(() => dialog.current?.close(), []);
  const move = useCallback((d: number) => setIndex((i) => (i === null ? i : (i + d + items.length) % items.length)), [items.length]);

  useEffect(() => {
    const el = dialog.current;
    if (!el) return;
    const onClose = () => {
      setIndex((i) => {
        if (i !== null) triggers.current[i]?.focus();
        return null;
      });
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") move(1);
      if (e.key === "ArrowLeft") move(-1);
    };
    el.addEventListener("close", onClose);
    el.addEventListener("keydown", onKey);
    return () => {
      el.removeEventListener("close", onClose);
      el.removeEventListener("keydown", onKey);
    };
  }, [move]);

  const item = index !== null ? items[index] : null;

  return (
    <>
      <ul className="columns-1 gap-4 sm:columns-2 lg:columns-3 [&>li]:mb-4">
        {items.map((it, i) => (
          <li key={it.id} className="break-inside-avoid">
            <figure className="group overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface">
              <button
                ref={(el) => {
                  triggers.current[i] = el;
                }}
                type="button"
                onClick={() => open(i)}
                className="block w-full overflow-hidden"
                aria-label={`Enlarge photo: ${it.dish}`}
              >
                <Image
                  src={it.image.src}
                  alt={it.image.alt}
                  width={it.image.width ?? 1200}
                  height={it.image.height ?? 900}
                  sizes="(min-width: 1024px) 30vw, (min-width: 640px) 45vw, 100vw"
                  className="h-auto w-full transition-transform duration-500 group-hover:scale-[1.02]"
                />
              </button>
              <figcaption className="px-4 py-3">
                <p className="font-display font-semibold">{it.dish}</p>
                {it.description && <p className="mt-0.5 text-sm text-muted">{it.description}</p>}
                {it.date && <p className="mt-1 font-mono text-[0.7rem] text-muted">{formatDate(it.date)}</p>}
              </figcaption>
            </figure>
          </li>
        ))}
      </ul>

      <dialog
        ref={dialog}
        aria-label={item ? `Photo: ${item.dish}` : "Photo viewer"}
        className="m-auto max-h-[94vh] w-[min(64rem,94vw)] overflow-y-auto rounded-[var(--radius-card)] border border-border bg-bg p-0 text-text backdrop:bg-black/80"
        onClick={(e) => {
          if (e.target === dialog.current) close();
        }}
      >
        {item && (
          <div>
            <div className="relative bg-black">
              <Image
                src={item.image.src}
                alt={item.image.alt}
                width={item.image.width ?? 1600}
                height={item.image.height ?? 1200}
                sizes="94vw"
                className="mx-auto h-auto max-h-[70vh] w-auto"
              />
            </div>
            <div className="flex flex-col gap-3 p-5 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <p className="font-display text-xl font-semibold">{item.dish}</p>
                {item.description && <p className="mt-1 text-muted">{item.description}</p>}
                {item.story && <p className="mt-3 max-w-prose text-[0.95rem] leading-relaxed">{item.story}</p>}
                {item.recipe && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-sm text-accent">Recipe</summary>
                    <p className="mt-2 whitespace-pre-line text-sm text-muted">{item.recipe}</p>
                  </details>
                )}
              </div>
              <div className="flex shrink-0 gap-2">
                {items.length > 1 && (
                  <>
                    <button type="button" onClick={() => move(-1)} className="grid h-10 w-10 place-items-center rounded-md border border-border hover:border-accent/60" aria-label="Previous photo">
                      <ChevronLeft size={18} aria-hidden="true" />
                    </button>
                    <button type="button" onClick={() => move(1)} className="grid h-10 w-10 place-items-center rounded-md border border-border hover:border-accent/60" aria-label="Next photo">
                      <ChevronRight size={18} aria-hidden="true" />
                    </button>
                  </>
                )}
                <button type="button" onClick={close} className="grid h-10 w-10 place-items-center rounded-md border border-border hover:border-accent/60" aria-label="Close" autoFocus>
                  <X size={18} aria-hidden="true" />
                </button>
              </div>
            </div>
            <p className="sr-only" aria-live="polite">
              Photo {(index ?? 0) + 1} of {items.length}
            </p>
          </div>
        )}
      </dialog>
    </>
  );
}
