/**
 * Section-specific technical illustrations for the eleven project categories.
 * Each is an abstract, original diagram in the shared visual language.
 * Add a new kind by adding a case here and to `illustrationKinds` in schemas.ts.
 */
import type { IllustrationKind } from "@/lib/schemas";
import { cn, seededRandom } from "@/lib/utils";

const stroke = { strokeWidth: 1, fill: "none" } as const;

function Frame({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <svg viewBox="0 0 320 180" className={cn("h-auto w-full", className)} aria-hidden="true" focusable="false" fill="none">
      {/* faint technical grid */}
      <g className="stroke-border" strokeWidth={0.6}>
        {Array.from({ length: 9 }, (_, i) => (
          <line key={`gx${i}`} x1={i * 40} y1={0} x2={i * 40} y2={180} />
        ))}
        {Array.from({ length: 5 }, (_, i) => (
          <line key={`gy${i}`} x1={0} y1={i * 45} x2={320} y2={i * 45} />
        ))}
      </g>
      {children}
    </svg>
  );
}

function Regression() {
  const r = seededRandom(1);
  const pts = Array.from({ length: 16 }, (_, i) => {
    const x = 24 + i * 17.5;
    return { x, y: 150 - (x - 20) * 0.38 + (r() - 0.5) * 40 };
  });
  return (
    <Frame>
      <line x1={14} y1={156} x2={306} y2={44} className="stroke-accent" {...stroke} strokeWidth={1.4} />
      {pts.map((p, i) => (
        <g key={i}>
          <line x1={p.x} y1={p.y} x2={p.x} y2={156 - (p.x - 14) * (112 / 292)} className="stroke-blue-soft" strokeOpacity={0.45} {...stroke} strokeDasharray="2 3" />
          <circle cx={p.x} cy={p.y} r={3} className="fill-blue-soft" />
        </g>
      ))}
    </Frame>
  );
}

function Clusters() {
  const centers = [
    [80, 60],
    [220, 70],
    [150, 135],
  ];
  const r = seededRandom(2);
  return (
    <Frame>
      {centers.map(([cx, cy], k) => (
        <g key={k}>
          <circle cx={cx} cy={cy} r={34} className={k === 0 ? "stroke-accent" : "stroke-blue"} strokeOpacity={0.6} {...stroke} strokeDasharray="3 4" />
          {Array.from({ length: 10 }, (_, i) => {
            const a = r() * Math.PI * 2;
            const d = r() * 26;
            return <circle key={i} cx={cx + Math.cos(a) * d} cy={cy + Math.sin(a) * d} r={2.4} className={k === 0 ? "fill-accent" : "fill-blue-soft"} opacity={0.85} />;
          })}
          <path d={`M ${cx - 5} ${cy} h 10 M ${cx} ${cy - 5} v 10`} className="stroke-text" {...stroke} />
        </g>
      ))}
    </Frame>
  );
}

function Tree({ x, y, s, hot }: { x: number; y: number; s: number; hot?: boolean }) {
  const c = hot ? "stroke-accent" : "stroke-blue-soft";
  return (
    <g>
      <path d={`M ${x} ${y} L ${x - s} ${y + s} M ${x} ${y} L ${x + s} ${y + s} M ${x - s} ${y + s} L ${x - s * 1.5} ${y + s * 2} M ${x - s} ${y + s} L ${x - s * 0.5} ${y + s * 2} M ${x + s} ${y + s} L ${x + s * 1.5} ${y + s * 2}`} className={c} {...stroke} />
      {[
        [x, y],
        [x - s, y + s],
        [x + s, y + s],
        [x - s * 1.5, y + s * 2],
        [x - s * 0.5, y + s * 2],
        [x + s * 1.5, y + s * 2],
      ].map(([a, b], i) => (
        <circle key={i} cx={a} cy={b} r={3} className={hot && i === 0 ? "fill-accent" : "fill-bg stroke-blue-soft"} strokeWidth={1} />
      ))}
    </g>
  );
}

function Ensemble() {
  return (
    <Frame>
      <Tree x={60} y={36} s={22} />
      <Tree x={160} y={36} s={22} />
      <Tree x={260} y={36} s={22} hot />
      <path d="M 60 120 Q 160 170 260 120" className="stroke-blue" {...stroke} strokeDasharray="3 4" />
      <path d="M 160 120 L 160 150" className="stroke-blue" {...stroke} />
      <rect x={140} y={150} width={40} height={18} rx={4} className="stroke-accent fill-bg" strokeWidth={1} />
    </Frame>
  );
}

function Neural() {
  const layers = [3, 5, 5, 2];
  const pos = layers.map((n, li) => Array.from({ length: n }, (_, i) => [50 + li * 73, 90 + (i - (n - 1) / 2) * 30] as const));
  return (
    <Frame>
      {pos.slice(0, -1).map((layer, li) =>
        layer.flatMap(([x1, y1], a) =>
          pos[li + 1].map(([x2, y2], b) => (
            <line key={`${li}-${a}-${b}`} x1={x1} y1={y1} x2={x2} y2={y2} className={a === 0 && b === 1 ? "stroke-accent" : "stroke-blue"} strokeOpacity={a === 0 && b === 1 ? 0.9 : 0.35} strokeWidth={0.8} />
          )),
        ),
      )}
      {pos.flat().map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={6} className="fill-bg stroke-blue-soft" strokeWidth={1.1} />
      ))}
    </Frame>
  );
}

function Vision() {
  return (
    <Frame>
      <path d="M 40 90 Q 160 10 280 90 Q 160 170 40 90 Z" className="stroke-blue-soft" {...stroke} />
      <circle cx={160} cy={90} r={30} className="stroke-blue" {...stroke} />
      <circle cx={160} cy={90} r={11} className="fill-accent" opacity={0.85} />
      <rect x={196} y={30} width={70} height={48} className="stroke-accent" {...stroke} strokeDasharray="4 3" />
      <rect x={48} y={110} width={54} height={40} className="stroke-blue" {...stroke} strokeDasharray="4 3" />
      <text x={198} y={26} className="fill-accent font-mono" fontSize={9}>0.94</text>
    </Frame>
  );
}

function Graph() {
  const n = [
    [160, 90],
    [90, 50],
    [92, 132],
    [230, 46],
    [238, 136],
    [40, 92],
    [286, 90],
  ];
  const e = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 5],
    [2, 5],
    [3, 6],
    [4, 6],
  ];
  return (
    <Frame>
      {e.map(([a, b], i) => (
        <line key={i} x1={n[a][0]} y1={n[a][1]} x2={n[b][0]} y2={n[b][1]} className={a === 0 ? "stroke-accent" : "stroke-blue"} strokeOpacity={a === 0 ? 0.8 : 0.5} {...stroke} />
      ))}
      {/* message arrows into the centre node */}
      {[1, 2, 3, 4].map((k) => {
        const [x, y] = n[k];
        const mx = x + (160 - x) * 0.55;
        const my = y + (90 - y) * 0.55;
        return <circle key={k} cx={mx} cy={my} r={2.5} className="fill-accent" />;
      })}
      {n.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i === 0 ? 10 : 7} className={i === 0 ? "fill-bg stroke-accent" : "fill-bg stroke-blue-soft"} strokeWidth={1.2} />
      ))}
    </Frame>
  );
}

function Federated() {
  const clients = [
    [50, 40],
    [50, 140],
    [270, 40],
    [270, 140],
    [160, 160],
  ];
  return (
    <Frame>
      {clients.map(([x, y], i) => (
        <g key={i}>
          <line x1={x} y1={y} x2={160} y2={75} className="stroke-blue" strokeOpacity={0.55} {...stroke} strokeDasharray="3 4" />
          <rect x={x - 14} y={y - 10} width={28} height={20} rx={3} className="fill-bg stroke-blue-soft" strokeWidth={1} />
        </g>
      ))}
      <circle cx={160} cy={75} r={20} className="fill-bg stroke-accent" strokeWidth={1.2} />
      <text x={160} y={79} textAnchor="middle" className="fill-accent font-mono" fontSize={10}>Σ</text>
    </Frame>
  );
}

function Privacy() {
  const r = seededRandom(9);
  return (
    <Frame>
      {Array.from({ length: 60 }, (_, i) => (
        <circle key={i} cx={20 + r() * 280} cy={20 + r() * 140} r={1.2} className="fill-blue-soft" opacity={0.35} />
      ))}
      <rect x={128} y={80} width={64} height={52} rx={6} className="fill-bg stroke-accent" strokeWidth={1.2} />
      <path d="M 142 80 v -14 a 18 18 0 0 1 36 0 v 14" className="stroke-accent" {...stroke} strokeWidth={1.2} />
      <circle cx={160} cy={104} r={5} className="fill-accent" />
      <path d="M 20 150 Q 90 110 128 110 M 192 110 Q 240 110 300 150" className="stroke-blue" {...stroke} strokeDasharray="2 4" />
    </Frame>
  );
}

function Retrieval() {
  return (
    <Frame>
      {[0, 1, 2].map((i) => (
        <rect key={i} x={30 + i * 8} y={40 + i * 8} width={50} height={64} rx={4} className="fill-bg stroke-blue-soft" strokeWidth={1} />
      ))}
      {[0, 1, 2, 3].map((i) => (
        <line key={i} x1={54} y1={68 + i * 10} x2={90} y2={68 + i * 10} className="stroke-blue" strokeOpacity={0.6} strokeWidth={1} />
      ))}
      <path d="M 104 90 H 140" className="stroke-blue" {...stroke} />
      {Array.from({ length: 12 }, (_, i) => (
        <circle key={i} cx={150 + (i % 4) * 14} cy={70 + Math.floor(i / 4) * 14} r={3.2} className={i === 5 || i === 10 ? "fill-accent" : "fill-blue-soft"} opacity={0.8} />
      ))}
      <path d="M 212 90 H 236" className="stroke-accent" {...stroke} />
      <rect x={240} y={62} width={56} height={56} rx={8} className="fill-bg stroke-accent" strokeWidth={1.2} />
      <path d="M 252 82 h 32 M 252 92 h 24 M 252 102 h 28" className="stroke-text" strokeOpacity={0.7} {...stroke} />
    </Frame>
  );
}

function Generative() {
  const pts = Array.from({ length: 120 }, (_, i) => {
    const t = i / 12;
    const r = 4 + i * 0.62;
    return `${160 + Math.cos(t) * r},${90 + Math.sin(t) * r * 0.82}`;
  }).join(" ");
  return (
    <Frame>
      <polyline points={pts} className="stroke-blue-soft" strokeOpacity={0.7} {...stroke} />
      <circle cx={160} cy={90} r={4} className="fill-accent" />
      <circle cx={232} cy={124} r={3} className="fill-accent anim-twinkle" />
    </Frame>
  );
}

function Agentic() {
  const steps = [
    [40, 90, "plan"],
    [120, 40, "tool"],
    [200, 90, "observe"],
    [120, 140, "act"],
  ] as const;
  return (
    <Frame>
      <path d="M 40 90 Q 70 40 120 40 Q 180 40 200 90 Q 180 140 120 140 Q 70 140 40 90" className="stroke-blue" {...stroke} strokeDasharray="3 4" />
      <path d="M 200 90 H 284" className="stroke-accent" {...stroke} />
      <path d="M 278 84 L 286 90 L 278 96" className="stroke-accent" {...stroke} />
      {steps.map(([x, y, label]) => (
        <g key={label}>
          <rect x={x - 26} y={y - 11} width={52} height={22} rx={5} className="fill-bg stroke-blue-soft" strokeWidth={1} />
          <text x={x} y={y + 3.5} textAnchor="middle" className="fill-text font-mono" fontSize={9}>
            {label}
          </text>
        </g>
      ))}
    </Frame>
  );
}

const registry: Record<IllustrationKind, () => React.ReactElement> = {
  regression: Regression,
  clusters: Clusters,
  ensemble: Ensemble,
  neural: Neural,
  vision: Vision,
  graph: Graph,
  federated: Federated,
  privacy: Privacy,
  retrieval: Retrieval,
  generative: Generative,
  agentic: Agentic,
};

export function CategoryIllustration({ kind, className }: { kind: IllustrationKind; className?: string }) {
  const Component = registry[kind];
  return (
    <div className={cn("text-blue", className)}>
      <Component />
    </div>
  );
}
