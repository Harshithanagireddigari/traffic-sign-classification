/**
 * charts.tsx — tiny dependency-free SVG charts used by the dashboard.
 * All values rendered here come from the real in-browser training run.
 */
import { CLASS_NAMES } from "./gtsrb";

export interface Series {
  label: string;
  color: string;
  values: number[];
}

export function LineChart({
  series,
  yDomain,
  xLabel = "epoch",
  height = 260,
}: {
  series: Series[];
  yDomain?: [number, number];
  xLabel?: string;
  height?: number;
}) {
  const W = 680;
  const H = height;
  const padL = 46;
  const padR = 14;
  const padT = 30;
  const padB = 28;
  const n = Math.max(...series.map((s) => s.values.length), 1);
  const all = series.flatMap((s) => s.values);
  let min = yDomain ? yDomain[0] : Math.min(...all);
  let max = yDomain ? yDomain[1] : Math.max(...all);
  if (max - min < 1e-6) {
    max = min + 1;
  }
  const x = (i: number) => padL + (i / Math.max(1, n - 1)) * (W - padL - padR);
  const y = (v: number) => padT + (1 - (v - min) / (max - min)) * (H - padT - padB);

  const gridVals: number[] = [];
  for (let i = 0; i <= 4; i++) gridVals.push(min + ((max - min) * i) / 4);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full">
      {gridVals.map((g, i) => (
        <g key={i}>
          <line x1={padL} x2={W - padR} y1={y(g)} y2={y(g)} stroke="#1e293b" strokeWidth={1} />
          <text x={padL - 8} y={y(g) + 4} textAnchor="end" fontSize={10} fill="#64748b">
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke="#334155" strokeWidth={1} />
      {n > 1 &&
        [0, Math.floor(n / 2), n - 1].map((i) => (
          <text key={i} x={x(i)} y={H - padB + 16} textAnchor="middle" fontSize={10} fill="#64748b">
            {i + 1}
          </text>
        ))}
      <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="#475569">
        {xLabel}
      </text>
      {series.map((s, si) => (
        <path
          key={si}
          d={s.values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}
          fill="none"
          stroke={s.color}
          strokeWidth={2.2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
      {series.map((s, si) => (
        <g key={si} transform={`translate(${W - padR - 150}, ${padT - 18 + si * 16})`}>
          <line x1={0} x2={18} y1={5} y2={5} stroke={s.color} strokeWidth={3} />
          <text x={24} y={9} fontSize={11} fill="#94a3b8">
            {s.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Confusion-matrix heatmap. Cell (i, j) = true class i predicted as j. */
export function ConfusionHeatmap({ matrix, size = 620 }: { matrix: number[][]; size?: number }) {
  const n = matrix.length;
  const cell = size / n;
  let max = 1;
  for (const row of matrix) for (const v of row) if (v > max) max = v;
  const c0 = [15, 23, 42]; // slate-900
  const c1 = [52, 211, 153]; // emerald-400
  const color = (v: number) => {
    const t = v / max;
    const c = c0.map((a, i) => Math.round(a + (c1[i] - a) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="h-auto w-full rounded-lg border border-slate-800">
      {matrix.map((row, i) =>
        row.map((v, j) => (
          <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell + 0.5} height={cell + 0.5} fill={color(v)}>
            <title>{`true ${i} (${CLASS_NAMES[i]}) → predicted ${j} (${CLASS_NAMES[j]}): ${v} sample(s)`}</title>
          </rect>
        )),
      )}
    </svg>
  );
}
