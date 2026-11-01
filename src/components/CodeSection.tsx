import { useState } from "react";
import type { ReactNode } from "react";
import { CODE_FILES } from "@/lib/pythonCode";
import { Card, SectionHeading } from "./ui";
import { cn } from "@/utils/cn";

/* ------------------------- tiny syntax highlighters ------------------------- */

const PY_RE =
  /(#.*$)|("""[\s\S]*?"""|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|\b(import|from|def|return|if|elif|else|for|while|in|not|and|or|is|class|try|except|finally|with|as|raise|lambda|pass|None|True|False|self|print|len|range|open|int|float|list|dict|set|str|enumerate|sorted|zip|os|sys|json)\b|(@\w+)|\b(\d+(?:\.\d+)?)\b/gm;

function highlightPython(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  PY_RE.lastIndex = 0;
  let key = 0;
  while ((m = PY_RE.exec(line)) !== null) {
    if (m.index > last) out.push(<span key={key++}>{line.slice(last, m.index)}</span>);
    const cls =
      m[1] !== undefined
        ? "text-slate-500 italic"
        : m[2] !== undefined
          ? "text-emerald-300"
          : m[3] !== undefined
            ? "text-fuchsia-400"
            : m[4] !== undefined
              ? "text-sky-300"
              : "text-amber-300";
    out.push(
      <span key={key++} className={cls}>
        {m[0]}
      </span>,
    );
    last = m.index + m[0].length;
    if (m[0].length === 0) PY_RE.lastIndex++;
  }
  if (last < line.length) out.push(<span key={key++}>{line.slice(last)}</span>);
  return out;
}

function highlightMarkdown(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(```|\*\*[^*]+\*\*|`[^`]+`|^#{1,6}\s.*$|^-\s.*|^\|.*\|)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) out.push(<span key={key++}>{line.slice(last, m.index)}</span>);
    let cls = "text-slate-200";
    if (m[0].startsWith("#")) cls = "text-sky-300 font-bold";
    else if (m[0].startsWith("-")) cls = "text-slate-300";
    else if (m[0].startsWith("|")) cls = "text-slate-400";
    else if (m[0].startsWith("**")) cls = "text-amber-300 font-semibold";
    else if (m[0].startsWith("`")) cls = "text-emerald-300 font-mono";
    else if (m[0] === "```") cls = "text-slate-500";
    out.push(
      <span key={key++} className={cls}>
        {m[0]}
      </span>,
    );
    last = m.index + m[0].length;
  }
  if (last < line.length) out.push(<span key={key++}>{line.slice(last)}</span>);
  return out;
}

function highlight(file: (typeof CODE_FILES)[number], line: string): ReactNode[] {
  if (file.lang === "python") return highlightPython(line);
  if (file.lang === "markdown") return highlightMarkdown(line);
  return [line];
}

/* -------------------------------- component -------------------------------- */

export default function CodeSection() {
  const [active, setActive] = useState(0);
  const [copied, setCopied] = useState(false);
  const file = CODE_FILES[active];
  const lines = file.code.split("\n");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(file.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <section>
      <SectionHeading
        id="code"
        eyebrow="Source code"
        title="Complete Python project — train.py, predict.py, app.py"
        desc="The full, executable mini-project lives in Traffic-Sign-Classification/ at the repository root. Browse, copy or download each file below."
      />

      <div className="mx-auto mt-10 max-w-6xl px-6">
        <Card className="overflow-hidden">
          {/* file tabs */}
          <div className="flex flex-wrap items-center gap-1 border-b border-slate-800 bg-slate-950/70 px-3 py-2">
            {CODE_FILES.map((f, i) => (
              <button
                key={f.name}
                onClick={() => setActive(i)}
                className={cn(
                  "rounded-lg px-3.5 py-2 font-mono text-xs font-semibold transition",
                  i === active ? "bg-slate-800 text-teal-300" : "text-slate-500 hover:bg-slate-800/60 hover:text-slate-300",
                )}
              >
                {f.name}
              </button>
            ))}
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={copy}
                className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-bold text-slate-300 transition hover:border-teal-400/50 hover:text-teal-300"
              >
                {copied ? "✓ Copied!" : "⧉ Copy"}
              </button>
              <a
                href={file.url}
                download={file.name}
                className="rounded-lg bg-teal-400/10 border border-teal-400/30 px-3 py-1.5 text-xs font-bold text-teal-300 transition hover:bg-teal-400/20"
              >
                ⬇ Download
              </a>
            </div>
          </div>

          {/* code view */}
          <div className="max-h-[560px] overflow-auto bg-[#0b1120]">
            <div className="flex items-center justify-between border-b border-slate-800/70 px-4 py-2">
              <span className="font-mono text-xs text-slate-500">{file.path}</span>
              <span className="rounded border border-slate-700 px-2 py-0.5 font-mono text-[10px] uppercase text-slate-400">
                {file.lang}
              </span>
            </div>
            <pre className="p-4 font-mono text-[12.5px] leading-relaxed text-slate-300">
              <code>
                {lines.map((line, i) => (
                  <div key={i} className="flex hover:bg-slate-800/30">
                    <span className="w-10 shrink-0 select-none pr-4 text-right text-slate-600">{i + 1}</span>
                    <span className="whitespace-pre-wrap break-all">{highlight(file, line)}</span>
                  </div>
                ))}
              </code>
            </pre>
          </div>
        </Card>
      </div>
    </section>
  );
}
