"use client";

import { useEffect, useRef } from "react";

/**
 * Progressive enhancement for rendered Markdown:
 *  • adds an accessible "Copy" button to every code block
 *  • renders ```mermaid blocks as diagrams (Mermaid is loaded only if needed,
 *    with securityLevel "strict" so diagram text cannot run scripts)
 * Without JavaScript, code stays readable and Mermaid source stays visible.
 */
export function ContentEnhancer({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;

    // Copy buttons
    const added: HTMLButtonElement[] = [];
    root.querySelectorAll("pre").forEach((pre) => {
      if (pre.classList.contains("mermaid-source") || pre.querySelector(".copy-code-button")) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-code-button";
      btn.textContent = "Copy";
      btn.setAttribute("aria-label", "Copy code to clipboard");
      btn.addEventListener("click", async () => {
        const code = pre.querySelector("code")?.innerText ?? pre.innerText;
        try {
          await navigator.clipboard.writeText(code.replace(/\nCopy$/, ""));
          btn.textContent = "Copied";
        } catch {
          btn.textContent = "Press Ctrl+C";
        }
        setTimeout(() => (btn.textContent = "Copy"), 1800);
      });
      pre.appendChild(btn);
      added.push(btn);
    });

    // Mermaid
    const diagrams = Array.from(root.querySelectorAll<HTMLElement>(".mermaid-diagram[data-mermaid]"));
    let cancelled = false;
    if (diagrams.length) {
      import("mermaid").then(async ({ default: mermaid }) => {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "base",
          themeVariables: {
            darkMode: true,
            background: "#111E33",
            primaryColor: "#192C48",
            primaryTextColor: "#F5F5F0",
            primaryBorderColor: "#4169E1",
            lineColor: "#7B9CF5",
            secondaryColor: "#001C40",
            tertiaryColor: "#080F20",
            fontFamily: "var(--font-jakarta), sans-serif",
          },
        });
        for (const [i, el] of diagrams.entries()) {
          if (cancelled || el.dataset.rendered) continue;
          try {
            const { svg } = await mermaid.render(`mermaid-${Date.now()}-${i}`, el.dataset.mermaid ?? "");
            el.innerHTML = svg;
            el.dataset.rendered = "true";
            el.setAttribute("role", "img");
            el.setAttribute("aria-label", "Diagram");
          } catch {
            // Leave the source visible if the diagram is invalid.
          }
        }
      });
    }
    return () => {
      cancelled = true;
      added.forEach((b) => b.remove());
    };
  }, []);

  return <div ref={ref}>{children}</div>;
}
