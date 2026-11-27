/**
 * Markdown / MDX rendering pipeline (server-only).
 *
 * • Local Markdown and MDX (your own files in content/) are trusted.
 * • Remote GitHub READMEs are untrusted: raw HTML is parsed and then sanitised,
 *   scripts and event handlers are stripped, and nothing is ever executed as MDX.
 *
 * Both paths support GitHub-flavoured Markdown, KaTeX maths, Shiki syntax
 * highlighting, Mermaid diagrams (rendered in the browser), heading anchors and
 * a generated table of contents.
 */
import "server-only";
import type { ReactNode } from "react";
import { unified, type Plugin } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema, type Options as SanitizeSchema } from "rehype-sanitize";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeKatex from "rehype-katex";
import rehypeShiki from "@shikijs/rehype";
import rehypeStringify from "rehype-stringify";
import { visit, SKIP } from "unist-util-visit";
import { toString as hastToString } from "hast-util-to-string";
import type { Element, Root } from "hast";
import readingTime from "reading-time";
import { compileMDX } from "next-mdx-remote/rsc";
import type { MDXComponents } from "mdx/types";

export interface Heading {
  id: string;
  text: string;
  level: 2 | 3;
}

export interface RepoContext {
  owner: string;
  name: string;
  branch: string;
  /** Path of the README inside the repo, used to resolve relative links. */
  readmePath: string;
}

const shikiOptions = {
  theme: "github-dark-dimmed",
  // Unknown languages fall back to plain text instead of failing the build.
  defaultLanguage: "text",
  fallbackLanguage: "text",
  addLanguageClass: true,
};

/* ---------------- Plugins ---------------- */

/** Turn ```mermaid blocks into a placeholder the browser renders. */
const rehypeMermaidPlaceholder: Plugin<[], Root> = () => (tree) => {
  visit(tree, "element", (node: Element, index, parent) => {
    if (node.tagName !== "pre" || !parent || index === undefined) return;
    const code = node.children[0];
    if (!code || code.type !== "element" || code.tagName !== "code") return;
    const cls = (code.properties?.className as string[] | undefined) ?? [];
    if (!cls.includes("language-mermaid")) return;
    const source = hastToString(code);
    parent.children[index] = {
      type: "element",
      tagName: "div",
      properties: { className: ["mermaid-diagram"], dataMermaid: source },
      children: [
        {
          type: "element",
          tagName: "pre",
          properties: { className: ["mermaid-source"] },
          children: [{ type: "text", value: source }],
        },
      ],
    };
    return SKIP;
  });
};

/** Collect h2/h3 headings (after slugs exist) for the table of contents. */
function rehypeCollectHeadings(target: Heading[]): Plugin<[], Root> {
  return () => (tree) => {
    visit(tree, "element", (node: Element) => {
      if (node.tagName !== "h2" && node.tagName !== "h3") return;
      const id = node.properties?.id;
      if (typeof id !== "string") return;
      target.push({ id, text: hastToString(node).trim(), level: node.tagName === "h2" ? 2 : 3 });
    });
  };
}

/** Remove a leading <h1>: the page already shows the project title. */
const rehypeDropLeadingH1: Plugin<[], Root> = () => (tree) => {
  const first = tree.children.find(
    (n) => n.type === "element" || (n.type === "text" && n.value.trim() !== ""),
  );
  if (first && first.type === "element" && first.tagName === "h1") {
    tree.children = tree.children.filter((n) => n !== first);
  }
};

function isRelative(url: string) {
  return !/^([a-z][a-z0-9+.-]*:|\/\/|#)/i.test(url);
}

function resolveRepoPath(repo: RepoContext, url: string) {
  const base = repo.readmePath.includes("/")
    ? repo.readmePath.slice(0, repo.readmePath.lastIndexOf("/") + 1)
    : "";
  const joined = url.startsWith("/") ? url.slice(1) : base + url;
  // Normalise ./ and ../ segments
  const parts: string[] = [];
  for (const seg of joined.split("/")) {
    if (seg === "." || seg === "") continue;
    if (seg === "..") parts.pop();
    else parts.push(seg);
  }
  return parts.join("/");
}

/** Make README-relative images and links point at GitHub. */
function rehypeResolveRepoUrls(repo: RepoContext): Plugin<[], Root> {
  return () => (tree) => {
    const rawBase = `https://raw.githubusercontent.com/${repo.owner}/${repo.name}/${repo.branch}/`;
    const blobBase = `https://github.com/${repo.owner}/${repo.name}/blob/${repo.branch}/`;
    visit(tree, "element", (node: Element) => {
      const props = node.properties ?? {};
      if ((node.tagName === "img" || node.tagName === "source" || node.tagName === "video") && typeof props.src === "string") {
        if (isRelative(props.src)) props.src = rawBase + resolveRepoPath(repo, props.src);
        if (node.tagName === "img") {
          props.loading = "lazy";
          props.decoding = "async";
        }
      }
      if (node.tagName === "a" && typeof props.href === "string") {
        if (isRelative(props.href)) props.href = blobBase + resolveRepoPath(repo, props.href);
        if (/^https?:\/\//.test(props.href)) {
          props.target = "_blank";
          props.rel = ["noopener", "noreferrer"];
        }
      }
      node.properties = props;
    });
  };
}

const sanitizeSchema: SanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    code: [...(defaultSchema.attributes?.code ?? []), ["className", /^language-./, "math-inline", "math-display"]],
    img: [...(defaultSchema.attributes?.img ?? []), "align", "width", "height"],
    p: [...(defaultSchema.attributes?.p ?? []), "align"],
    div: [...(defaultSchema.attributes?.div ?? []), "align"],
    h1: [...(defaultSchema.attributes?.h1 ?? []), "align"],
  },
  tagNames: [...(defaultSchema.tagNames ?? []), "video", "source"],
};

/* ---------------- Public API ---------------- */

export interface RenderedMarkdown {
  html: string;
  headings: Heading[];
  readingMinutes: number;
}

/** Render Markdown to sanitised HTML (used for READMEs and local .md files). */
export async function renderMarkdown(
  source: string,
  options: { trusted: boolean; repo?: RepoContext; dropLeadingH1?: boolean },
): Promise<RenderedMarkdown> {
  const headings: Heading[] = [];
  let processor = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw);

  if (!options.trusted) processor = processor.use(rehypeSanitize, sanitizeSchema);
  if (options.dropLeadingH1) processor = processor.use(rehypeDropLeadingH1);
  if (options.repo) processor = processor.use(rehypeResolveRepoUrls(options.repo));

  const file = await processor
    .use(rehypeSlug)
    .use(rehypeAutolinkHeadings, {
      behavior: "wrap",
      properties: { className: ["heading-anchor"] },
    })
    .use(rehypeCollectHeadings(headings))
    .use(rehypeKatex)
    .use(rehypeMermaidPlaceholder)
    .use(rehypeShiki, shikiOptions)
    .use(rehypeStringify)
    .process(source);

  return {
    html: String(file),
    headings,
    readingMinutes: Math.max(1, Math.round(readingTime(source).minutes)),
  };
}

export interface RenderedMdx {
  content: ReactNode;
  headings: Heading[];
  readingMinutes: number;
}

/** Compile trusted local MDX (your own files only — never remote content). */
export async function renderMdx(source: string, components: MDXComponents): Promise<RenderedMdx> {
  const headings: Heading[] = [];
  const { content } = await compileMDX({
    source,
    components,
    options: {
      parseFrontmatter: true,
      mdxOptions: {
        remarkPlugins: [remarkGfm, remarkMath],
        rehypePlugins: [
          rehypeSlug,
          [rehypeAutolinkHeadings, { behavior: "wrap", properties: { className: ["heading-anchor"] } }],
          rehypeCollectHeadings(headings),
          rehypeKatex,
          rehypeMermaidPlaceholder,
          [rehypeShiki, shikiOptions],
        ],
      },
    },
  });
  return {
    content,
    headings,
    readingMinutes: Math.max(1, Math.round(readingTime(source).minutes)),
  };
}
