import { Mail } from "lucide-react";
import { Github, Linkedin } from "@/components/ui/brand-icons";
import { siteConfig } from "@/config/site";
import { cn, isDev } from "@/lib/utils";

/**
 * GitHub / LinkedIn / Email links from src/config/site.ts.
 * A link that is not configured is not rendered (a dashed hint appears in
 * development only, so nothing broken ever ships).
 */
export function SocialLinks({ className, showLabels = false }: { className?: string; showLabels?: boolean }) {
  const { github, linkedin, email } = siteConfig.social;
  const links = [
    { key: "github", label: "GitHub", href: github, Icon: Github },
    { key: "linkedin", label: "LinkedIn", href: linkedin, Icon: Linkedin },
    { key: "email", label: "Email", href: email ? `mailto:${email}` : null, Icon: Mail },
  ];
  return (
    <ul className={cn("flex flex-wrap items-center gap-2", className)}>
      {links.map(({ key, label, href, Icon }) =>
        href ? (
          <li key={key}>
            <a
              href={href}
              {...(key !== "email" ? { target: "_blank", rel: "noopener noreferrer" } : {})}
              className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-muted transition-colors hover:border-accent/60 hover:text-accent"
              aria-label={showLabels ? undefined : label}
            >
              <Icon size={16} aria-hidden="true" />
              {showLabels && <span>{label}</span>}
            </a>
          </li>
        ) : isDev ? (
          <li key={key}>
            <span
              className="inline-flex items-center gap-2 rounded-md border border-dashed border-warm/60 px-3 py-2 font-mono text-xs text-warm"
              title={`Set siteConfig.social.${key} in src/config/site.ts`}
            >
              <Icon size={14} aria-hidden="true" /> {label}: not set (dev only)
            </span>
          </li>
        ) : null,
      )}
    </ul>
  );
}
