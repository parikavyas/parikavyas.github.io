import type { Metadata } from "next";
import { siteConfig } from "@/config/site";

/** Canonical link for a path — only emitted once a production URL is known. */
export function canonical(path: string): Pick<Metadata, "alternates"> {
  return siteConfig.url ? { alternates: { canonical: path } } : {};
}

/** True on Vercel preview deployments, which should never be indexed. */
export const isPreviewDeployment = process.env.VERCEL_ENV !== undefined && process.env.VERCEL_ENV !== "production";
