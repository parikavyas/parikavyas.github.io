import "server-only";
import type { ReactNode } from "react";
import { renderMarkdown, renderMdx, type Heading, type RepoContext } from "@/lib/markdown";
import { mdxComponents } from "@/components/content/mdx-components";
import { ContentEnhancer } from "@/components/content/content-enhancer";
import { cn } from "@/lib/utils";

export interface RenderedDocument {
  node: ReactNode;
  headings: Heading[];
  readingMinutes: number;
}

/** Render any Markdown/MDX source into a styled, enhanced article body. */
export async function renderDocument(
  source: string,
  opts: { format: "md" | "mdx"; trusted: boolean; repo?: RepoContext; dropLeadingH1?: boolean; className?: string },
): Promise<RenderedDocument> {
  const className = cn("prose", opts.className);
  if (opts.format === "mdx") {
    if (!opts.trusted) throw new Error("Refusing to compile untrusted MDX");
    const { content, headings, readingMinutes } = await renderMdx(source, mdxComponents);
    return {
      node: (
        <ContentEnhancer>
          <div className={className}>{content}</div>
        </ContentEnhancer>
      ),
      headings,
      readingMinutes,
    };
  }
  const { html, headings, readingMinutes } = await renderMarkdown(source, opts);
  return {
    node: (
      <ContentEnhancer>
        <div className={className} dangerouslySetInnerHTML={{ __html: html }} />
      </ContentEnhancer>
    ),
    headings,
    readingMinutes,
  };
}
