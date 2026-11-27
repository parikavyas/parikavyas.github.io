import type { MDXComponents } from "mdx/types";
import { Info } from "lucide-react";
import { LazyVisualization } from "@/components/visualizations/lazy-visualization";
import { DocumentPreview } from "./document-preview";
import { VideoEmbed } from "./video-embed";

/** Components available inside your own .mdx files. See docs/CONTENT_GUIDE.md. */
function Callout({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <aside className="not-prose my-6 rounded-[var(--radius-card)] border border-blue/40 bg-blue/10 px-5 py-4">
      <p className="flex items-center gap-2 font-display text-sm font-semibold text-blue-soft">
        <Info size={15} aria-hidden="true" /> {title ?? "Note"}
      </p>
      <div className="mt-1.5 text-[0.95rem] text-text/90">{children}</div>
    </aside>
  );
}

function Figure({ src, alt, caption }: { src: string; alt: string; caption?: string }) {
  return (
    <figure className="my-8">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} loading="lazy" />
      {caption && <figcaption className="mt-2 text-center font-mono text-xs text-muted">{caption}</figcaption>}
    </figure>
  );
}

export const mdxComponents: MDXComponents = {
  Callout,
  Figure,
  Visualization: ({ id }: { id: string }) => <LazyVisualization id={id} />,
  Document: ({ href, title }: { href: string; title: string }) => <DocumentPreview href={href} title={title} />,
  Video: (props: { provider: "youtube" | "vimeo" | "file"; id: string; title: string; thumbnail?: string }) => (
    <div className="my-6">
      <VideoEmbed {...props} />
    </div>
  ),
};
