import { useModel } from "@/lib/model";
import { cn } from "@/utils/cn";

const LINKS = [
  { href: "#overview", label: "Overview" },
  { href: "#demo", label: "Live Demo" },
  { href: "#dashboard", label: "Dashboard" },
  { href: "#code", label: "Source Code" },
  { href: "#concepts", label: "ML Concepts" },
  { href: "#run", label: "Run Locally" },
];

function statusPill(phase: string, epoch: number, total: number, error: string | null) {
  switch (phase) {
    case "preparing":
      return (
        <span className="inline-flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-400/10 px-3 py-1 text-xs font-semibold text-amber-300">
          <span className="pulse-dot inline-block h-2 w-2 rounded-full bg-amber-400" />
          Training (Preparing…)
        </span>
      );
    case "training":
      return (
        <span className="inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1 text-xs font-semibold text-sky-300">
          <span className="pulse-dot inline-block h-2 w-2 rounded-full bg-sky-400" />
          Training · epoch {epoch}/{total}
        </span>
      );
    case "ready":
      return (
        <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-300">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
          Model Ready
        </span>
      );
    case "error":
      return (
        <span className="inline-flex max-w-xs items-center gap-2 truncate rounded-full border border-rose-400/30 bg-rose-400/10 px-3 py-1 text-xs font-semibold text-rose-300" title={error ?? "Training failed"}>
          <span className="inline-block h-2 w-2 shrink-0 rounded-full bg-rose-400" />
          <span className="truncate">Training Failed{error ? `: ${error}` : ""}</span>
        </span>
      );
    default:
      return null;
  }
}

export default function Navbar() {
  const { phase, epoch, totalEpochs, error } = useModel();
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <a href="#overview" className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-teal-400 to-emerald-600 shadow-lg shadow-teal-500/20">
            <svg viewBox="0 0 24 24" className="h-5 w-5 text-slate-950" fill="currentColor">
              <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 3a2 2 0 1 1 0 4 2 2 0 0 1 0-4Zm0 5a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z" opacity=".9" />
              <path d="M13.8 14.5 12 22l-1.8-7.5h3.6Z" />
            </svg>
          </span>
          <span className="text-sm font-extrabold tracking-tight text-white sm:text-base">
            Traffic Sign <span className="text-teal-400">CNN</span>
          </span>
        </a>
        <nav className="hidden items-center gap-1 lg:flex">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className={cn(
                "rounded-lg px-3 py-2 text-sm font-medium text-slate-400 transition hover:bg-slate-800/70 hover:text-white",
              )}
            >
              {l.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-3">{statusPill(phase, epoch, totalEpochs, error)}</div>
      </div>
    </header>
  );
}
