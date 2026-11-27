import { hashString, seededRandom } from "@/lib/utils";

/**
 * A small, unique, procedurally generated strip for a project card,
 * seeded by the project's slug so it never changes between builds.
 */
export function ProjectGlyph({ slug, accentIndex }: { slug: string; accentIndex?: number }) {
  const rand = seededRandom(hashString(slug));
  const n = 9 + Math.floor(rand() * 5);
  const pts = Array.from({ length: n }, (_, i) => ({
    x: 12 + (i / (n - 1)) * 296 + (rand() - 0.5) * 18,
    y: 14 + rand() * 46,
    r: 1.6 + rand() * 2.2,
  }));
  const hot = accentIndex ?? Math.floor(rand() * n);
  return (
    <svg viewBox="0 0 320 74" className="h-16 w-full" aria-hidden="true" focusable="false" fill="none">
      <polyline points={pts.map((p) => `${p.x},${p.y}`).join(" ")} className="stroke-blue" strokeOpacity={0.5} strokeWidth={0.9} />
      {pts.map((p, i) =>
        i + 2 < pts.length && rand() > 0.6 ? (
          <line key={`b${i}`} x1={p.x} y1={p.y} x2={pts[i + 2].x} y2={pts[i + 2].y} className="stroke-blue-soft" strokeOpacity={0.25} strokeDasharray="2 3" />
        ) : null,
      )}
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={i === hot ? p.r + 1.5 : p.r} className={i === hot ? "fill-accent" : "fill-blue-soft"} opacity={i === hot ? 1 : 0.75} />
      ))}
    </svg>
  );
}
