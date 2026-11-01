import { CLASS_NAMES } from "@/lib/gtsrb";

export default function Footer() {
  return (
    <footer className="mt-24 border-t border-slate-800/80 bg-slate-950">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-col items-start justify-between gap-6 sm:flex-row sm:items-center">
          <div>
            <p className="text-sm font-extrabold text-white">
              Traffic Sign Classification <span className="text-teal-400">CNN</span>
            </p>
            <p className="mt-1 max-w-md text-xs leading-relaxed text-slate-500">
              B.Tech CSE (AIML) mini project · Python + TensorFlow/Keras + OpenCV + Streamlit · GTSRB dataset ·{" "}
              {CLASS_NAMES.length} classes. The in-browser model is a real CNN trained live with TensorFlow.js.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-[11px] font-semibold">
            {["Python", "TensorFlow/Keras", "CNN", "OpenCV", "NumPy/Pandas", "Streamlit", "GTSRB", "TensorFlow.js"].map((t) => (
              <span key={t} className="rounded-full border border-slate-800 bg-slate-900 px-3 py-1 text-slate-400">
                {t}
              </span>
            ))}
          </div>
        </div>
        <p className="mt-6 text-center text-[11px] text-slate-600">
          Built as an interactive companion to the Traffic-Sign-Classification Python project · train.py · predict.py · app.py
        </p>
      </div>
    </footer>
  );
}
