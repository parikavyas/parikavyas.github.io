import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import { siteConfig } from "@/config/site";
import { Container } from "@/components/layout/container";
import { SocialLinks } from "@/components/layout/social-links";
import { HelloBackdrop } from "@/components/illustrations/page-motifs";
import { Portrait } from "@/components/content/portrait";
import { Reveal } from "@/components/ui/reveal";

export const metadata: Metadata = { ...canonical("/") };

export default function HelloPage() {
  const { hero } = siteConfig;
  return (
    <div className="relative isolate overflow-hidden">
      <HelloBackdrop />
      <Container className="relative grid min-h-[calc(100vh-var(--header-height))] items-center gap-14 py-16 md:grid-cols-[1.25fr_1fr] md:py-24 lg:gap-20">
        <div className="max-w-2xl">
          <Reveal>
            <p className="font-mono text-xs uppercase tracking-[0.25em] text-muted">
              <span className="text-accent">00</span> · origin point
            </p>
            <h1 className="mt-6 font-display text-[clamp(3rem,9vw,6.75rem)] font-bold leading-[0.95] tracking-[-0.035em]">
              {hero.greetingBefore}
              <span className="text-accent">{hero.greetingHighlight}</span>
              {hero.greetingAfter}
            </h1>
          </Reveal>
          <Reveal delay={0.15}>
            <div className="mt-8 space-y-5 text-[1.075rem] leading-[1.8] text-muted sm:text-lg">
              {hero.paragraphs.map((p) => (
                <p key={p.slice(0, 24)} className="first:text-text">
                  {p}
                </p>
              ))}
            </div>
            <SocialLinks className="mt-9" showLabels />
          </Reveal>
        </div>
        <Reveal delay={0.25} className="w-full md:justify-self-end">
          <Portrait />
        </Reveal>
      </Container>
    </div>
  );
}
