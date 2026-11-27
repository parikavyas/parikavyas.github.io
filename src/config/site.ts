/**
 * Central site configuration.
 *
 * Everything personal (name, introduction, links, footer wording) lives here,
 * so no component hardcodes it. See docs/CONTENT_GUIDE.md → "Homepage".
 *
 * A link set to `null` is not rendered anywhere on the site. Fill it in when
 * the real value is ready.
 */
export const siteConfig = {
  name: "Parika Vyas",
  firstName: "Parika",
  initials: "PV",
  /**
   * Production URL, used for canonical links, the sitemap and social previews.
   * Set NEXT_PUBLIC_SITE_URL in Vercel (e.g. https://parika.vercel.app).
   * On Vercel it otherwise falls back to the project's production domain.
   */
  url:
    process.env.NEXT_PUBLIC_SITE_URL ??
    (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : null),
  title: "Parika Vyas",
  description:
    "A personal universe: what Parika Vyas does, builds, learns and wonders about — AI, products and technology for social good.",
  locale: "en_US",

  hero: {
    /** The word inside the greeting that is highlighted in yellow. */
    greetingBefore: "Hello, I’m ",
    greetingHighlight: "Parika",
    greetingAfter: "!",
    paragraphs: [
      "I’m a builder at heart, endlessly curious by nature, and passionate about technology for social good. My journey has taken me from understanding real-world problems to writing code, deploying AI solutions, and shaping business decisions. I’m fascinated by the intersection of AI, business, and social impact — and what becomes possible when we build with intention.",
      "This is my little corner of the internet to share what I do, what I build, what I learn, and everything in between.",
    ],
  },

  /**
   * Portrait shown on the homepage. Put the file in public/images/profile/
   * and set `src` to e.g. "/images/profile/parika.jpg".
   */
  portrait: {
    src: null as string | null,
    alt: "Portrait of Parika Vyas",
  },

  social: {
    /** Taken from the GitHub account this site is built under. */
    github: "https://github.com/parikavyas" as string | null,
    /** Taken from the previous version of this site. */
    linkedin: "https://www.linkedin.com/in/parika-vyas/" as string | null,
    /** e.g. "hello@example.com" (no "mailto:") */
    email: null as string | null,
  },

  footer: {
    tagline: "Made with curiosity and a little bit of wonder.",
    invitation:
      "Found something interesting? Have a different perspective? Want to build something together? My inbox is always open.",
  },

  /**
   * Manually bump when you change content that has no date of its own
   * (e.g. the homepage text). The footer shows the newest date across this
   * value and every dated content entry. Format: YYYY-MM-DD.
   */
  lastUpdated: "2026-11-27",
} as const;

export type SiteConfig = typeof siteConfig;
