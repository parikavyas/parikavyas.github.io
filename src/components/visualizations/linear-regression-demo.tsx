"use client";

import { useMemo, useRef, useState } from "react";
import { Plus, RotateCcw, Trash2, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { seededRandom } from "@/lib/utils";
import { Stat, VisualizationFrame } from "./visualization-frame";

interface Point {
  x: number;
  y: number;
}

const W = 560;
const H = 360;
const PAD = 36;
const MAX = 10;

const sx = (x: number) => PAD + (x / MAX) * (W - 2 * PAD);
const sy = (y: number) => H - PAD - (y / MAX) * (H - 2 * PAD);

function initialPoints(seed = 3): Point[] {
  const r = seededRandom(seed);
  return Array.from({ length: 12 }, () => {
    const x = 0.6 + r() * 8.8;
    const y = Math.min(9.6, Math.max(0.3, 0.7 * x + 1.4 + (r() - 0.5) * 2.6));
    return { x: +x.toFixed(2), y: +y.toFixed(2) };
  });
}

/** Ordinary least squares: closed-form slope and intercept. */
export function fitOls(points: Point[]) {
  const n = points.length;
  if (n < 2) return null;
  const mx = points.reduce((s, p) => s + p.x, 0) / n;
  const my = points.reduce((s, p) => s + p.y, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (const p of points) {
    sxy += (p.x - mx) * (p.y - my);
    sxx += (p.x - mx) ** 2;
    syy += (p.y - my) ** 2;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const sse = points.reduce((s, p) => s + (p.y - (slope * p.x + intercept)) ** 2, 0);
  return { slope, intercept, r2: syy === 0 ? 1 : 1 - sse / syy, mse: sse / n };
}

export default function LinearRegressionDemo() {
  const [points, setPoints] = useState<Point[]>(() => initialPoints());
  const [showResiduals, setShowResiduals] = useState(true);
  const seed = useRef(10);
  const svgRef = useRef<SVGSVGElement>(null);
  const fit = useMemo(() => fitOls(points), [points]);

  const addAtClient = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const vx = ((clientX - rect.left) / rect.width) * W;
    const vy = ((clientY - rect.top) / rect.height) * H;
    const x = ((vx - PAD) / (W - 2 * PAD)) * MAX;
    const y = ((H - PAD - vy) / (H - 2 * PAD)) * MAX;
    if (x < 0 || x > MAX || y < 0 || y > MAX) return;
    setPoints((p) => [...p, { x: +x.toFixed(2), y: +y.toFixed(2) }]);
  };

  const addRandom = () => {
    const r = seededRandom(seed.current++);
    const x = 0.3 + r() * 9.4;
    setPoints((p) => [...p, { x: +x.toFixed(2), y: +(0.3 + r() * 9.4).toFixed(2) }]);
  };

  const line = fit
    ? { x1: sx(0), y1: sy(fit.intercept), x2: sx(MAX), y2: sy(fit.slope * MAX + fit.intercept) }
    : null;

  const equation = fit
    ? `ŷ = ${fit.slope.toFixed(3)}x ${fit.intercept >= 0 ? "+" : "−"} ${Math.abs(fit.intercept).toFixed(3)}`
    : "Add at least two points with different x values";

  return (
    <VisualizationFrame
      title="Interactive linear regression"
      description="Click or tap inside the plot to add points. The least-squares line refits instantly; dashed lines show each residual."
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_14rem]">
        <div>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            className="h-auto w-full cursor-crosshair touch-manipulation rounded-lg border border-border bg-bg"
            role="img"
            aria-label={`Scatter plot of ${points.length} points with fitted line ${equation}`}
            onClick={(e) => addAtClient(e.clientX, e.clientY)}
          >
            <defs>
              <clipPath id="lr-clip">
                <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} />
              </clipPath>
            </defs>
            {Array.from({ length: MAX + 1 }, (_, i) => (
              <g key={i}>
                <line x1={sx(i)} y1={PAD} x2={sx(i)} y2={H - PAD} className="stroke-border" strokeWidth={0.7} />
                <line x1={PAD} y1={sy(i)} x2={W - PAD} y2={sy(i)} className="stroke-border" strokeWidth={0.7} />
                {i % 2 === 0 && (
                  <>
                    <text x={sx(i)} y={H - PAD + 16} textAnchor="middle" className="fill-muted font-mono" fontSize={10}>
                      {i}
                    </text>
                    <text x={PAD - 10} y={sy(i) + 3} textAnchor="end" className="fill-muted font-mono" fontSize={10}>
                      {i}
                    </text>
                  </>
                )}
              </g>
            ))}
            <g clipPath="url(#lr-clip)">
              {fit &&
                showResiduals &&
                points.map((p, i) => (
                  <line
                    key={`r${i}`}
                    x1={sx(p.x)}
                    y1={sy(p.y)}
                    x2={sx(p.x)}
                    y2={sy(fit.slope * p.x + fit.intercept)}
                    className="stroke-blue-soft"
                    strokeOpacity={0.6}
                    strokeDasharray="3 3"
                  />
                ))}
              {line && <line {...line} className="stroke-accent" strokeWidth={2.2} />}
            </g>
            {points.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={5} className="fill-blue stroke-text" strokeWidth={1} />
            ))}
          </svg>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button onClick={addRandom}>
              <Plus size={15} aria-hidden="true" /> Random point
            </Button>
            <Button onClick={() => setPoints((p) => p.slice(0, -1))} disabled={!points.length}>
              <Undo2 size={15} aria-hidden="true" /> Undo
            </Button>
            <Button onClick={() => setPoints([])} disabled={!points.length}>
              <Trash2 size={15} aria-hidden="true" /> Clear
            </Button>
            <Button onClick={() => setPoints(initialPoints())}>
              <RotateCcw size={15} aria-hidden="true" /> Reset
            </Button>
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-sm text-muted">
              <input
                type="checkbox"
                checked={showResiduals}
                onChange={(e) => setShowResiduals(e.target.checked)}
                className="h-4 w-4 accent-[var(--accent)]"
              />
              Show residuals
            </label>
          </div>
        </div>
        <div>
          <p className="rounded-lg border border-accent/30 bg-accent/5 px-3 py-2.5 font-mono text-sm text-accent" aria-live="polite">
            {equation}
          </p>
          <dl className="mt-3 grid grid-cols-2 gap-2">
            <Stat label="Points" value={String(points.length)} />
            <Stat label="R²" value={fit ? fit.r2.toFixed(3) : "—"} />
            <Stat label="MSE" value={fit ? fit.mse.toFixed(3) : "—"} />
            <Stat label="Slope" value={fit ? fit.slope.toFixed(3) : "—"} />
          </dl>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Try adding one point far from the others and watch how strongly a single outlier pulls the line — least squares
            penalises the <em>square</em> of each residual.
          </p>
        </div>
      </div>
    </VisualizationFrame>
  );
}
