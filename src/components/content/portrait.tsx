import Image from "next/image";
import { siteConfig } from "@/config/site";
import { DimensionalFrame, OrbitalPaths } from "@/components/illustrations/primitives";
import { isDev } from "@/lib/utils";

/**
 * Homepage portrait inside a dimensional frame. Until a photo is configured
 * (siteConfig.portrait.src) an abstract origin-point illustration stands in.
 */
export function Portrait() {
  const { src, alt } = siteConfig.portrait;
  return (
    <div className="relative mx-auto aspect-[4/5] w-full max-w-[25rem]">
      <DimensionalFrame className="absolute -inset-6 h-[calc(100%+3rem)] w-[calc(100%+3rem)] opacity-70" depth={6} />
      <div className="absolute inset-0 overflow-hidden rounded-[18px] border border-border bg-gradient-to-b from-bg-secondary to-surface">
        {src ? (
          <Image src={src} alt={alt} fill priority sizes="(min-width: 768px) 400px, 80vw" className="object-cover" />
        ) : (
          <div className="relative h-full w-full">
            <OrbitalPaths className="absolute inset-0 m-auto h-[85%] w-[85%] opacity-90" rings={4} seed={12} />
            <span className="absolute inset-x-0 bottom-6 text-center font-mono text-[0.7rem] uppercase tracking-[0.25em] text-muted">
              {isDev ? "Portrait placeholder · set siteConfig.portrait.src" : siteConfig.initials}
            </span>
          </div>
        )}
      </div>
      <span className="absolute -bottom-3 -left-3 h-6 w-6 border-b-2 border-l-2 border-accent" aria-hidden="true" />
      <span className="absolute -right-3 -top-3 h-6 w-6 border-r-2 border-t-2 border-accent" aria-hidden="true" />
    </div>
  );
}
