"use client";

import Image from "next/image";
import { useState } from "react";
import { Play } from "lucide-react";

/**
 * Privacy-friendly, lazy video embed: shows a lightweight poster until the
 * visitor presses play, then loads the YouTube (no-cookie) or Vimeo player.
 */
export function VideoEmbed({
  provider,
  id,
  title,
  thumbnail,
}: {
  provider: "youtube" | "vimeo" | "file";
  id: string;
  title: string;
  thumbnail?: string;
}) {
  const [active, setActive] = useState(false);
  const poster = thumbnail ?? (provider === "youtube" ? `https://i.ytimg.com/vi/${id}/hqdefault.jpg` : undefined);

  if (provider === "file") {
    return (
      <video controls preload="none" poster={thumbnail} className="aspect-video w-full rounded-lg bg-bg" aria-label={title}>
        <source src={id} />
      </video>
    );
  }

  const src =
    provider === "youtube"
      ? `https://www.youtube-nocookie.com/embed/${encodeURIComponent(id)}?autoplay=1&rel=0`
      : `https://player.vimeo.com/video/${encodeURIComponent(id)}?autoplay=1&dnt=1`;

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-border bg-bg">
      {active ? (
        <iframe
          src={src}
          title={title}
          allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
          allowFullScreen
          className="absolute inset-0 h-full w-full"
        />
      ) : (
        <button type="button" onClick={() => setActive(true)} className="group absolute inset-0 h-full w-full" aria-label={`Play video: ${title}`}>
          {poster && <Image src={poster} alt="" fill sizes="(min-width: 768px) 50vw, 100vw" className="object-cover opacity-80" unoptimized />}
          <span className="absolute left-1/2 top-1/2 grid h-14 w-14 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-accent text-bg transition-transform group-hover:scale-105">
            <Play size={22} aria-hidden="true" />
          </span>
        </button>
      )}
    </div>
  );
}
