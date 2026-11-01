import { Stat } from "./ui";
import { CLASS_NAMES } from "@/lib/gtsrb";

export default function Hero() {
  return (
    <section id="overview" className="relative overflow-hidden pt-16">
      {/* background glow */}
      <div className="pointer-events-none absolute -top-40 left-1/2 h-[480px] w-[820px] -translate-x-1/2 rounded-full bg-teal-500/15 blur-3xl" />
      <div className="pointer-events-none absolute top-40 -left-32 h-72 w-72 rounded-full bg-emerald-500/10 blur-3xl" />
      <div className="pointer-events-none absolute top-64 -right-32 h-72 w-72 rounded-full bg-sky-500/10 blur-3xl" />

      <div className="relative mx-auto max-w-6xl px-6 pb-16 pt-16 sm:pt-24">
        <div className="grid items-center gap-12 lg:grid-cols-[1.4fr_1fr]">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-teal-400/30 bg-teal-400/10 px-4 py-1.5 text-xs font-bold uppercase tracking-widest text-teal-300">
              B.Tech CSE · AI &amp; ML Mini Project
            </span>
            <h1 className="mt-6 text-4xl font-black leading-[1.08] tracking-tight text-white sm:text-5xl lg:text-6xl">
              Traffic Sign Classification{" "}
              <span className="bg-gradient-to-r from-teal-300 via-emerald-400 to-teal-300 bg-clip-text text-transparent">
                with a CNN
              </span>
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-400">
              Upload any traffic-sign photo and a <strong className="text-slate-200">Convolutional Neural Network</strong>{" "}
              identifies it among <strong className="text-slate-200">{CLASS_NAMES.length} GTSRB classes</strong> — built with{" "}
              <strong className="text-slate-200">Python, TensorFlow/Keras, OpenCV and Streamlit</strong>. This page runs a
              real CNN in your browser: it trains live on GTSRB-style data, then classifies your upload.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <a
                href="#demo"
                className="rounded-xl bg-gradient-to-r from-teal-400 to-emerald-500 px-6 py-3 text-sm font-bold text-slate-950 shadow-lg shadow-teal-500/25 transition hover:brightness-110"
              >
                ▶ Try the live demo
              </a>
              <a
                href="#code"
                className="rounded-xl border border-slate-700 bg-slate-900/70 px-6 py-3 text-sm font-bold text-slate-200 transition hover:border-teal-400/50 hover:text-teal-300"
              >
                View Python source
              </a>
            </div>
            <div className="mt-10 grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat value={String(CLASS_NAMES.length)} label="sign classes" />
              <Stat value="32×32" label="input pixels" />
              <Stat value="3×Conv" label="CNN blocks" />
              <Stat value="~50k" label="GTSRB images" />
            </div>
          </div>

          {/* animated traffic light */}
          <div className="animate-float mx-auto hidden lg:block">
            <div className="rounded-3xl border border-slate-700/80 bg-slate-900 p-5 shadow-2xl shadow-black/40">
              <div className="mx-auto flex h-72 w-24 flex-col items-center justify-evenly rounded-2xl border-4 border-slate-800 bg-slate-950 py-4">
                <span className="tl-red h-12 w-12 rounded-full bg-red-500 shadow-lg shadow-red-500/40" />
                <span className="tl-amber h-12 w-12 rounded-full bg-amber-400 shadow-lg shadow-amber-400/40" />
                <span className="tl-green h-12 w-12 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/40" />
              </div>
              <div className="mx-auto h-16 w-6 bg-slate-800" />
              <div className="mx-auto h-3 w-28 rounded-full bg-slate-800" />
            </div>
            <p className="mt-4 text-center text-xs font-medium text-slate-500">
              pipeline: image → 32×32 → normalize → CNN → softmax
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
