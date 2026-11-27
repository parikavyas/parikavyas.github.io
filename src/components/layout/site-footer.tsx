import Link from "next/link";
import { siteConfig } from "@/config/site";
import { primaryNav } from "@/config/navigation";
import { formatDate } from "@/lib/dates";
import { getSiteLastUpdated } from "@/lib/last-updated";
import { Constellation } from "@/components/illustrations/primitives";
import { SocialLinks } from "./social-links";
import { Container } from "./container";

export function SiteFooter() {
  const lastUpdated = formatDate(getSiteLastUpdated());
  const year = (getSiteLastUpdated() ?? siteConfig.lastUpdated).slice(0, 4);
  return (
    <footer className="relative mt-24 overflow-hidden border-t border-border bg-bg-secondary/40">
      <Constellation className="pointer-events-none absolute -right-10 -top-6 w-72 opacity-40 sm:w-96" seed={64} count={8} />
      <Container className="relative grid gap-10 py-14 md:grid-cols-[1.5fr_1fr]">
        <div>
          <p className="font-display text-2xl font-semibold tracking-tight">{siteConfig.footer.tagline}</p>
          <p className="mt-3 max-w-xl text-muted">{siteConfig.footer.invitation}</p>
          <SocialLinks className="mt-6" showLabels />
        </div>
        <nav aria-label="Footer" className="md:justify-self-end">
          <ul className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
            {primaryNav.map((item) => (
              <li key={item.href}>
                <Link href={item.href} className="text-muted transition-colors hover:text-accent">
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </Container>
      <Container className="relative flex flex-col gap-2 border-t border-border/70 py-6 font-mono text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
        <p>
          © {year} {siteConfig.name}
        </p>
        {lastUpdated && (
          <p>
            Last updated <time dateTime={getSiteLastUpdated() ?? undefined}>{lastUpdated}</time>
          </p>
        )}
      </Container>
    </footer>
  );
}
