import type { ReactNode } from "react";
import { cn } from "@/utils/cn";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("rounded-2xl border border-slate-800 bg-slate-900/60 shadow-xl shadow-black/20", className)}>
      {children}
    </div>
  );
}

export function Badge({ children, tone = "teal" }: { children: ReactNode; tone?: "teal" | "amber" | "sky" | "rose" }) {
  const tones: Record<string, string> = {
    teal: "border-teal-400/30 bg-teal-400/10 text-teal-300",
    amber: "border-amber-400/30 bg-amber-400/10 text-amber-300",
    sky: "border-sky-400/30 bg-sky-400/10 text-sky-300",
    rose: "border-rose-400/30 bg-rose-400/10 text-rose-300",
  };
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold", tones[tone])}>
      {children}
    </span>
  );
}

export function SectionHeading({
  id,
  eyebrow,
  title,
  desc,
}: {
  id: string;
  eyebrow: string;
  title: string;
  desc?: string;
}) {
  return (
    <div id={id} className="mx-auto max-w-6xl scroll-mt-24 px-6 pt-20">
      <Badge tone="sky">{eyebrow}</Badge>
      <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">{title}</h2>
      {desc && <p className="mt-3 max-w-3xl text-base leading-relaxed text-slate-400">{desc}</p>}
    </div>
  );
}

export function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 text-center">
      <div className="bg-gradient-to-r from-teal-300 to-emerald-400 bg-clip-text text-3xl font-black text-transparent">
        {value}
      </div>
      <div className="mt-1 text-xs font-medium uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  );
}
