/**
 * Reusable illustration primitives.
 *
 * All artwork is original, procedural SVG: thin blue and yellow lines on navy,
 * with seeded randomness so the server and browser draw identical shapes.
 * Every illustration is decorative (aria-hidden). Motion classes only animate
 * when the visitor has not asked for reduced motion (see globals.css).
 */
import { useId } from "react";
import { cn, seededRandom } from "@/lib/utils";

interface SvgProps {
  className?: string;
}

function Svg({
  className,
  viewBox,
  children,
  preserveAspectRatio,
}: SvgProps & { viewBox: string; children: React.ReactNode; preserveAspectRatio?: string }) {
  return (
    <svg
      viewBox={viewBox}
      className={cn("pointer-events-none select-none", className)}
      aria-hidden="true"
      focusable="false"
      preserveAspectRatio={preserveAspectRatio}
      fill="none"
    >
      {children}
    </svg>
  );
}

/* ------------------------------------------------------------------ */

/** A floor grid receding to a vanishing point. */
export function PerspectiveGrid({
  className,
  rows = 12,
  cols = 16,
  horizon = 0.38,
}: SvgProps & { rows?: number; cols?: number; horizon?: number }) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const W = 1000;
  const H = 600;
  const hy = H * horizon;
  const vx = W / 2;
  const lines: React.ReactNode[] = [];
  for (let i = 0; i <= cols; i++) {
    const x = (i / cols) * W * 2.4 - W * 0.7;
    lines.push(<line key={`v${i}`} x1={vx} y1={hy} x2={x} y2={H} />);
  }
  for (let j = 1; j <= rows; j++) {
    const t = Math.pow(j / rows, 2.2);
    const y = hy + (H - hy) * t;
    lines.push(<line key={`h${j}`} x1={0} y1={y} x2={W} y2={y} />);
  }
  return (
    <Svg viewBox={`0 0 ${W} ${H}`} className={className} preserveAspectRatio="xMidYMax slice">
      <defs>
        <linearGradient id={`pg-fade-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="white" stopOpacity="0" />
          <stop offset="0.45" stopColor="white" stopOpacity="0.5" />
          <stop offset="1" stopColor="white" stopOpacity="1" />
        </linearGradient>
        <mask id={`pg-mask-${uid}`}>
          <rect width={W} height={H} fill={`url(#pg-fade-${uid})`} />
        </mask>
      </defs>
      <g mask={`url(#pg-mask-${uid})`} className="stroke-blue" strokeWidth={0.8} strokeOpacity={0.55}>
        {lines}
      </g>
      <line x1={0} y1={hy} x2={W} y2={hy} className="stroke-blue-soft" strokeOpacity={0.25} strokeWidth={0.8} />
    </Svg>
  );
}

/* ------------------------------------------------------------------ */

/** Tilted concentric orbits with small bodies. */
export function OrbitalPaths({
  className,
  rings = 4,
  seed = 7,
  tilt = -18,
}: SvgProps & { rings?: number; seed?: number; tilt?: number }) {
  const rand = seededRandom(seed);
  return (
    <Svg viewBox="0 0 400 400" className={className}>
      <g transform={`rotate(${tilt} 200 200)`}>
        {Array.from({ length: rings }, (_, i) => {
          const rx = 60 + i * 42;
          const ry = rx * 0.42;
          const a = rand() * Math.PI * 2;
          const px = 200 + rx * Math.cos(a);
          const py = 200 + ry * Math.sin(a);
          return (
            <g key={i} className={i % 2 ? "anim-spin-slower" : "anim-spin-slow"}>
              <ellipse
                cx={200}
                cy={200}
                rx={rx}
                ry={ry}
                className="stroke-blue"
                strokeOpacity={0.55 - i * 0.08}
                strokeWidth={0.9}
                strokeDasharray={i === rings - 1 ? "2 6" : undefined}
              />
              <circle cx={px} cy={py} r={i === 1 ? 4 : 2.4} className={i === 1 ? "fill-accent" : "fill-blue-soft"} />
            </g>
          );
        })}
      </g>
      <circle cx={200} cy={200} r={9} className="fill-bg stroke-accent" strokeWidth={1.2} />
      <circle cx={200} cy={200} r={2.5} className="fill-accent" />
    </Svg>
  );
}

/* ------------------------------------------------------------------ */

export interface StarPoint {
  x: number;
  y: number;
  r: number;
  bright: boolean;
}

export function makeConstellation(seed: number, count: number, w = 400, h = 300): StarPoint[] {
  const rand = seededRandom(seed);
  return Array.from({ length: count }, (_, i) => ({
    x: 30 + rand() * (w - 60),
    y: 30 + rand() * (h - 60),
    r: 1.2 + rand() * 2.2,
    bright: i % 4 === 0,
  }));
}

/** Stars joined by thin lines (a minimum-ish spanning path). */
export function Constellation({
  className,
  seed = 3,
  count = 9,
  field = 40,
  accent = true,
}: SvgProps & { seed?: number; count?: number; field?: number; accent?: boolean }) {
  const pts = makeConstellation(seed, count);
  // Greedy nearest-neighbour path so lines read as a constellation, not a web.
  const order = [0];
  const used = new Set(order);
  while (order.length < pts.length) {
    const last = pts[order[order.length - 1]];
    let best = -1;
    let bestD = Infinity;
    pts.forEach((p, i) => {
      if (used.has(i)) return;
      const d = (p.x - last.x) ** 2 + (p.y - last.y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    order.push(best);
    used.add(best);
  }
  const rand = seededRandom(seed * 13 + 1);
  const dust = Array.from({ length: field }, () => ({ x: rand() * 400, y: rand() * 300, r: rand() * 0.9 + 0.3 }));
  return (
    <Svg viewBox="0 0 400 300" className={className}>
      {dust.map((d, i) => (
        <circle key={`d${i}`} cx={d.x} cy={d.y} r={d.r} className="fill-muted" opacity={0.35} />
      ))}
      <polyline
        points={order.map((i) => `${pts[i].x},${pts[i].y}`).join(" ")}
        className="stroke-blue-soft"
        strokeOpacity={0.55}
        strokeWidth={0.9}
      />
      {/* one closing branch for shape */}
      <line
        x1={pts[order[1]].x}
        y1={pts[order[1]].y}
        x2={pts[order[Math.min(4, order.length - 1)]].x}
        y2={pts[order[Math.min(4, order.length - 1)]].y}
        className="stroke-blue"
        strokeOpacity={0.35}
        strokeWidth={0.8}
        strokeDasharray="3 4"
      />
      {pts.map((p, i) => (
        <g key={i}>
          {p.bright && accent && <circle cx={p.x} cy={p.y} r={p.r + 4} className="fill-accent" opacity={0.12} />}
          <circle
            cx={p.x}
            cy={p.y}
            r={p.r}
            className={cn(p.bright && accent ? "fill-accent anim-twinkle" : "fill-text")}
            style={{ animationDelay: `${(i * 0.7) % 4}s` }}
          />
        </g>
      ))}
    </Svg>
  );
}

/* ------------------------------------------------------------------ */

/** Isometric floating rooms, stairs and walkways — impossible architecture. */
export function FloatingStructure({ className, seed = 11, warm = false }: SvgProps & { seed?: number; warm?: boolean }) {
  const rand = seededRandom(seed);
  const iso = (x: number, y: number, z: number) => {
    const px = 200 + (x - y) * 0.866 * 22;
    const py = 210 + (x + y) * 0.5 * 22 - z * 22;
    return [px, py] as const;
  };
  const box = (x: number, y: number, z: number, w: number, d: number, h: number, key: string) => {
    const p = (a: number, b: number, c: number) => iso(a, b, c).join(",");
    const top = [p(x, y, z + h), p(x + w, y, z + h), p(x + w, y + d, z + h), p(x, y + d, z + h)].join(" ");
    const left = [p(x, y + d, z), p(x, y + d, z + h), p(x + w, y + d, z + h), p(x + w, y + d, z)].join(" ");
    const right = [p(x + w, y, z), p(x + w, y, z + h), p(x + w, y + d, z + h), p(x + w, y + d, z)].join(" ");
    return (
      <g key={key}>
        <polygon points={left} className="fill-surface stroke-blue" strokeWidth={0.8} strokeOpacity={0.7} />
        <polygon points={right} className="fill-surface-elevated stroke-blue" strokeWidth={0.8} strokeOpacity={0.7} />
        <polygon points={top} className="fill-bg-secondary stroke-blue-soft" strokeWidth={0.8} strokeOpacity={0.9} />
      </g>
    );
  };
  const rooms = Array.from({ length: 5 }, (_, i) => ({
    x: Math.round(rand() * 6 - 3),
    y: Math.round(rand() * 6 - 3),
    z: Math.round(rand() * 5 - 1) + i,
    w: 1 + Math.round(rand() * 2),
    d: 1 + Math.round(rand() * 2),
    h: 0.4 + rand() * 1.4,
  })).sort((a, b) => a.x + a.y - (b.x + b.y) || a.z - b.z);
  const stairs: React.ReactNode[] = [];
  for (let s = 0; s < 6; s++) {
    stairs.push(box(-4 + s * 0.5, 2, -1 + s * 0.35, 0.5, 1, 0.35, `s${s}`));
  }
  const accentStroke = warm ? "stroke-warm" : "stroke-accent";
  const [ax, ay] = iso(rooms[2].x, rooms[2].y, rooms[2].z + rooms[2].h);
  return (
    <Svg viewBox="0 0 400 400" className={className}>
      {stairs}
      {rooms.map((r, i) => box(r.x, r.y, r.z, r.w, r.d, r.h, `r${i}`))}
      {/* a thread of light linking two rooms */}
      <path
        d={`M ${iso(rooms[0].x, rooms[0].y, rooms[0].z + rooms[0].h).join(" ")} Q 200 60 ${ax} ${ay}`}
        className={accentStroke}
        strokeWidth={1}
        strokeOpacity={0.8}
        strokeDasharray="2 5"
      />
      <circle cx={ax} cy={ay} r={3} className={warm ? "fill-warm" : "fill-accent"} />
    </Svg>
  );
}

/* ------------------------------------------------------------------ */

/** A graph of nodes and edges. */
export function NodeNetwork({
  className,
  seed = 5,
  nodes = 14,
  highlight = 2,
}: SvgProps & { seed?: number; nodes?: number; highlight?: number }) {
  const rand = seededRandom(seed);
  const pts = Array.from({ length: nodes }, () => ({ x: 30 + rand() * 340, y: 30 + rand() * 240 }));
  const edges: [number, number][] = [];
  pts.forEach((p, i) => {
    const near = pts
      .map((q, j) => ({ j, d: (p.x - q.x) ** 2 + (p.y - q.y) ** 2 }))
      .filter((o) => o.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, 2);
    near.forEach(({ j }) => {
      if (!edges.some(([a, b]) => (a === j && b === i) || (a === i && b === j))) edges.push([i, j]);
    });
  });
  return (
    <Svg viewBox="0 0 400 300" className={className}>
      {edges.map(([a, b], k) => {
        const hot = a < highlight || b < highlight;
        return (
          <line
            key={k}
            x1={pts[a].x}
            y1={pts[a].y}
            x2={pts[b].x}
            y2={pts[b].y}
            className={hot ? "stroke-accent" : "stroke-blue"}
            strokeOpacity={hot ? 0.7 : 0.45}
            strokeWidth={hot ? 1.1 : 0.8}
          />
        );
      })}
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r={i < highlight ? 5 : 3.2}
          className={i < highlight ? "fill-accent" : "fill-bg stroke-blue-soft"}
          strokeWidth={1}
        />
      ))}
    </Svg>
  );
}

/* ------------------------------------------------------------------ */

/** Receding nested frames — a doorway into another dimension. */
export function DimensionalFrame({ className, depth = 7 }: SvgProps & { depth?: number }) {
  return (
    <Svg viewBox="0 0 400 400" className={className}>
      {Array.from({ length: depth }, (_, i) => {
        const t = i / depth;
        const size = 340 * (1 - t * 0.8);
        const x = 200 - size / 2 + t * 40;
        const y = 200 - size / 2 - t * 30;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={size}
            height={size}
            rx={6}
            className={i === depth - 1 ? "stroke-accent" : "stroke-blue"}
            strokeOpacity={i === depth - 1 ? 0.9 : 0.6 - t * 0.4}
            strokeWidth={0.9}
          />
        );
      })}
      {/* corner connectors give the frames depth */}
      {[
        [30, 30],
        [370, 30],
        [370, 370],
        [30, 370],
      ].map(([cx, cy], k) => {
        const t = (depth - 1) / depth;
        const size = 340 * (1 - t * 0.8);
        const ix = 200 - size / 2 + t * 40 + (cx > 200 ? size : 0);
        const iy = 200 - size / 2 - t * 30 + (cy > 200 ? size : 0);
        return <line key={k} x1={cx} y1={cy} x2={ix} y2={iy} className="stroke-blue-soft" strokeOpacity={0.3} strokeWidth={0.7} />;
      })}
    </Svg>
  );
}

/* ------------------------------------------------------------------ */

/** Concentric rotating geometry around a bright origin point. */
export function Portal({ className }: SvgProps) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  return (
    <Svg viewBox="0 0 400 400" className={className}>
      <defs>
        <radialGradient id={`portal-core-${uid}`} cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#e8ff47" stopOpacity="0.35" />
          <stop offset="0.35" stopColor="#4169e1" stopOpacity="0.18" />
          <stop offset="1" stopColor="#080f20" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx={200} cy={200} r={190} fill={`url(#portal-core-${uid})`} />
      <g className="anim-spin-slow">
        {Array.from({ length: 6 }, (_, i) => (
          <rect
            key={i}
            x={200 - (150 - i * 20)}
            y={200 - (150 - i * 20)}
            width={(150 - i * 20) * 2}
            height={(150 - i * 20) * 2}
            transform={`rotate(${i * 15} 200 200)`}
            className="stroke-blue"
            strokeOpacity={0.25 + i * 0.08}
            strokeWidth={0.8}
          />
        ))}
      </g>
      <g className="anim-spin-slower">
        <circle cx={200} cy={200} r={168} className="stroke-blue-soft" strokeOpacity={0.4} strokeWidth={0.7} strokeDasharray="1 7" />
        <circle cx={200} cy={32} r={3} className="fill-accent" />
      </g>
      <circle cx={200} cy={200} r={36} className="stroke-accent" strokeOpacity={0.8} strokeWidth={1} />
      <circle cx={200} cy={200} r={4} className="fill-accent" />
    </Svg>
  );
}

/* ------------------------------------------------------------------ */

/** Warmer, organic forms for Beyond the Desk. */
export function OrganicForms({ className, seed = 21 }: SvgProps & { seed?: number }) {
  const rand = seededRandom(seed);
  const blobs = Array.from({ length: 4 }, (_, i) => {
    const cx = 80 + rand() * 240;
    const cy = 70 + rand() * 160;
    const r = 30 + rand() * 50;
    const pts = Array.from({ length: 8 }, (_, k) => {
      const a = (k / 8) * Math.PI * 2;
      const rr = r * (0.82 + rand() * 0.3);
      return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)];
    });
    // smooth closed path through points
    const d =
      pts
        .map((p, k) => {
          const next = pts[(k + 1) % pts.length];
          const mx = (p[0] + next[0]) / 2;
          const my = (p[1] + next[1]) / 2;
          return k === 0 ? `M ${mx} ${my}` : `Q ${p[0]} ${p[1]} ${mx} ${my}`;
        })
        .join(" ") + ` Q ${pts[0][0]} ${pts[0][1]} ${(pts[0][0] + pts[1][0]) / 2} ${(pts[0][1] + pts[1][1]) / 2} Z`;
    return { d, i };
  });
  return (
    <Svg viewBox="0 0 400 300" className={className}>
      {blobs.map(({ d, i }) => (
        <path
          key={i}
          d={d}
          className={i === 0 ? "stroke-warm anim-drift" : i === 1 ? "stroke-accent" : "stroke-blue-soft"}
          strokeOpacity={i === 0 ? 0.9 : 0.5}
          strokeWidth={1}
          style={{ animationDelay: `${i * 2}s` }}
        />
      ))}
      {Array.from({ length: 3 }, (_, k) => (
        <path
          key={`w${k}`}
          d={`M 20 ${230 + k * 14} C 120 ${200 + k * 14}, 220 ${260 + k * 14}, 380 ${222 + k * 14}`}
          className="stroke-warm"
          strokeOpacity={0.35 - k * 0.08}
          strokeWidth={0.9}
        />
      ))}
    </Svg>
  );
}
