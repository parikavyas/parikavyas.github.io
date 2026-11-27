"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RefreshCw, Shuffle, StepForward } from "lucide-react";
import { useReducedMotion } from "motion/react";
import { Button } from "@/components/ui/button";
import { seededRandom } from "@/lib/utils";
import { Stat, VisualizationFrame } from "./visualization-frame";

interface P {
  x: number;
  y: number;
}

const W = 560;
const H = 380;
const COLORS = ["#E8FF47", "#4169E1", "#F2B880", "#7FD4C1", "#C79BF2", "#FF8F8F", "#7B9CF5", "#F5F5F0"];

/** Synthetic data: a few Gaussian blobs (Box–Muller). */
function makeData(seed: number): P[] {
  const r = seededRandom(seed);
  const blobs = 3 + Math.floor(r() * 3);
  const centres = Array.from({ length: blobs }, () => ({ x: 70 + r() * (W - 140), y: 60 + r() * (H - 120) }));
  const pts: P[] = [];
  centres.forEach((c) => {
    const spread = 22 + r() * 26;
    for (let i = 0; i < 48; i++) {
      const u = Math.max(r(), 1e-9);
      const v = r();
      const mag = Math.sqrt(-2 * Math.log(u)) * spread;
      pts.push({
        x: Math.min(W - 8, Math.max(8, c.x + mag * Math.cos(2 * Math.PI * v))),
        y: Math.min(H - 8, Math.max(8, c.y + mag * Math.sin(2 * Math.PI * v))),
      });
    }
  });
  return pts;
}

/** k-means++ seeding: spread initial centroids apart. */
function initCentroids(points: P[], k: number, seed: number): P[] {
  const r = seededRandom(seed);
  const cs: P[] = [{ ...points[Math.floor(r() * points.length)] }];
  while (cs.length < k) {
    const d = points.map((p) => Math.min(...cs.map((c) => (p.x - c.x) ** 2 + (p.y - c.y) ** 2)));
    const total = d.reduce((a, b) => a + b, 0);
    let t = r() * total;
    let idx = 0;
    for (; idx < d.length - 1; idx++) {
      t -= d[idx];
      if (t <= 0) break;
    }
    cs.push({ ...points[idx] });
  }
  return cs;
}

function nearest(p: P, cs: P[]) {
  let best = 0;
  let bd = Infinity;
  cs.forEach((c, i) => {
    const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
    if (d < bd) {
      bd = d;
      best = i;
    }
  });
  return best;
}

function assign(points: P[], cs: P[]) {
  return points.map((p) => nearest(p, cs));
}

function update(points: P[], labels: number[], cs: P[]) {
  return cs.map((c, k) => {
    let sx = 0;
    let sy = 0;
    let n = 0;
    points.forEach((p, i) => {
      if (labels[i] === k) {
        sx += p.x;
        sy += p.y;
        n++;
      }
    });
    return n ? { x: sx / n, y: sy / n } : c; // empty cluster keeps its position
  });
}

function inertia(points: P[], labels: number[], cs: P[]) {
  return points.reduce((s, p, i) => s + (p.x - cs[labels[i]].x) ** 2 + (p.y - cs[labels[i]].y) ** 2, 0);
}

interface State {
  centroids: P[];
  labels: number[] | null;
  iteration: number;
  converged: boolean;
  trail: P[][];
}

function fresh(points: P[], k: number, seed: number): State {
  const centroids = initCentroids(points, k, seed);
  return { centroids, labels: null, iteration: 0, converged: false, trail: [centroids] };
}

function iterate(points: P[], s: State): State {
  const labels = assign(points, s.centroids);
  const next = update(points, labels, s.centroids);
  const moved = next.reduce((m, c, i) => Math.max(m, Math.hypot(c.x - s.centroids[i].x, c.y - s.centroids[i].y)), 0);
  return {
    centroids: next,
    labels: assign(points, next),
    iteration: s.iteration + 1,
    converged: moved < 0.01,
    trail: [...s.trail, next],
  };
}

export default function KMeansDemo() {
  const reduceMotion = useReducedMotion();
  const [dataSeed, setDataSeed] = useState(7);
  const [k, setK] = useState(3);
  const initSeed = useRef(1);
  const points = useMemo(() => makeData(dataSeed), [dataSeed]);
  const [state, setState] = useState<State>(() => fresh(points, k, initSeed.current));
  const [running, setRunning] = useState(false);

  const reset = useCallback(
    (nextK = k, pts = points) => {
      setRunning(false);
      initSeed.current += 1;
      setState(fresh(pts, nextK, initSeed.current));
    },
    [k, points],
  );

  const step = useCallback(() => setState((s) => (s.converged ? s : iterate(points, s))), [points]);

  // Animated run: one iteration every 550ms until convergence.
  useEffect(() => {
    if (!running) return;
    if (state.converged || state.iteration >= 50) {
      setRunning(false);
      return;
    }
    if (reduceMotion) {
      let s = state;
      while (!s.converged && s.iteration < 50) s = iterate(points, s);
      setState(s);
      setRunning(false);
      return;
    }
    const t = setTimeout(step, 550);
    return () => clearTimeout(t);
  }, [running, state, step, points, reduceMotion]);

  // Faint decision regions: colour a coarse grid by nearest centroid.
  const cells = useMemo(() => {
    const size = 20;
    const out: { x: number; y: number; c: number }[] = [];
    for (let x = 0; x < W; x += size) for (let y = 0; y < H; y += size) out.push({ x, y, c: nearest({ x: x + size / 2, y: y + size / 2 }, state.centroids) });
    return { out, size };
  }, [state.centroids]);

  const currentInertia = state.labels ? inertia(points, state.labels, state.centroids) : null;
  const status = state.converged
    ? `Converged after ${state.iteration} iteration${state.iteration === 1 ? "" : "s"}`
    : state.iteration === 0
      ? "Centroids initialised with k-means++. Press Step or Run."
      : `Iteration ${state.iteration}`;

  return (
    <VisualizationFrame
      title="Interactive k-means clustering"
      description="Each step assigns every point to its nearest centroid, then moves each centroid to the mean of its points. Change K or re-seed to see different outcomes."
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_14rem]">
        <div>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="h-auto w-full rounded-lg border border-border bg-bg"
            role="img"
            aria-label={`${points.length} synthetic points clustered into ${k} groups. ${status}.`}
          >
            <g opacity={state.labels ? 0.09 : 0}>
              {cells.out.map((c, i) => (
                <rect key={i} x={c.x} y={c.y} width={cells.size} height={cells.size} fill={COLORS[c.c]} />
              ))}
            </g>
            {points.map((p, i) => (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={3.4}
                fill={state.labels ? COLORS[state.labels[i]] : "#A5B2C8"}
                opacity={0.85}
              />
            ))}
            {/* centroid trails */}
            {state.centroids.map((_, ci) => (
              <polyline
                key={`t${ci}`}
                points={state.trail.map((cs) => `${cs[ci]?.x ?? 0},${cs[ci]?.y ?? 0}`).join(" ")}
                fill="none"
                stroke={COLORS[ci]}
                strokeOpacity={0.6}
                strokeDasharray="3 3"
              />
            ))}
            {state.centroids.map((c, ci) => (
              <g key={`c${ci}`} style={{ transition: reduceMotion ? undefined : "transform 0.45s ease" }} transform={`translate(${c.x} ${c.y})`}>
                <circle r={11} fill="#080F20" stroke={COLORS[ci]} strokeWidth={2} />
                <path d="M -5 0 H 5 M 0 -5 V 5" stroke={COLORS[ci]} strokeWidth={2} />
              </g>
            ))}
          </svg>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button variant="primary" onClick={() => setRunning((r) => !r)} disabled={state.converged}>
              {running ? <Pause size={15} aria-hidden="true" /> : <Play size={15} aria-hidden="true" />}
              {running ? "Pause" : "Run"}
            </Button>
            <Button onClick={step} disabled={running || state.converged}>
              <StepForward size={15} aria-hidden="true" /> Step
            </Button>
            <Button onClick={() => reset()}>
              <RefreshCw size={15} aria-hidden="true" /> Re-seed centroids
            </Button>
            <Button
              onClick={() => {
                const next = dataSeed + 1;
                setDataSeed(next);
                reset(k, makeData(next));
              }}
            >
              <Shuffle size={15} aria-hidden="true" /> New data
            </Button>
          </div>
        </div>
        <div>
          <label htmlFor="kmeans-k" className="flex items-baseline justify-between text-sm">
            <span>Number of clusters</span>
            <span className="font-mono text-accent">K = {k}</span>
          </label>
          <input
            id="kmeans-k"
            type="range"
            min={1}
            max={8}
            value={k}
            onChange={(e) => {
              const next = Number(e.target.value);
              setK(next);
              reset(next);
            }}
            className="mt-2 w-full accent-[var(--accent)]"
          />
          <p className="mt-3 rounded-lg border border-accent/30 bg-accent/5 px-3 py-2.5 font-mono text-xs leading-relaxed text-accent" aria-live="polite">
            {status}
          </p>
          <dl className="mt-3 grid grid-cols-2 gap-2">
            <Stat label="Points" value={String(points.length)} />
            <Stat label="Iteration" value={String(state.iteration)} />
            <Stat label="Inertia" value={currentInertia === null ? "—" : Math.round(currentInertia).toLocaleString("en-US")} />
            <Stat label="K" value={String(k)} />
          </dl>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Inertia is the total squared distance from each point to its centroid. It never increases between iterations —
            but different starting centroids can settle in different local optima.
          </p>
        </div>
      </div>
    </VisualizationFrame>
  );
}
