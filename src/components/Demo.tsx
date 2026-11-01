import { useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { useModel } from "@/lib/model";
import type { Prediction } from "@/lib/model";
import { signThumbnailDataURL } from "@/lib/gtsrb";
import { Badge, Card, SectionHeading } from "./ui";
import { cn } from "@/utils/cn";

const ACCEPTED = [".jpg", ".jpeg", ".png"];

export default function Demo() {
  const { ready, phase, epoch, totalEpochs, error: modelError, predict } = useModel();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File | undefined | null) => {
    if (!f) return;
    setError(null);
    setResult(null);
    const ext = "." + (f.name.split(".").pop() ?? "").toLowerCase();
    if (!ACCEPTED.includes(ext) && !/^image\/(jpeg|png)$/.test(f.type)) {
      setError(`"${f.name}" is not supported. Please upload a JPG, JPEG or PNG image.`);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  const onPick = (e: ChangeEvent<HTMLInputElement>) => handleFile(e.target.files?.[0]);

  const onPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await predict(file);
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section>
      <SectionHeading
        id="demo"
        eyebrow="Live demo"
        title="Upload a traffic sign — the CNN classifies it"
        desc="This is not a mock: a CNN (Conv2D + ReLU → MaxPool ×3 → Flatten → Dense → Softmax) was trained live in your browser on procedurally rendered GTSRB-style signs. Your image goes through the exact same preprocessing as the Python app: resize to 32×32 and normalize pixels to [0, 1]."
      />

      <div className="mx-auto mt-10 grid max-w-6xl gap-6 px-6 lg:grid-cols-2">
        {/* ------------------------------ upload card ------------------------------ */}
        <Card className="p-6">
          <h3 className="text-lg font-bold text-white">1 · Upload image</h3>
          <p className="mt-1 text-sm text-slate-400">JPG, JPEG or PNG — drag &amp; drop or click to browse.</p>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={cn(
              "mt-4 flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition",
              dragging ? "border-teal-400 bg-teal-400/10" : "border-slate-700 bg-slate-950/50 hover:border-teal-400/50",
            )}
          >
            <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" className="hidden" onChange={onPick} />
            {preview ? (
              <img src={preview} alt="uploaded traffic sign" className="max-h-64 rounded-lg object-contain shadow-lg" />
            ) : (
              <>
                <svg viewBox="0 0 24 24" className="h-10 w-10 text-slate-600" fill="none" stroke="currentColor" strokeWidth={1.6}>
                  <path d="M12 16V4m0 0 4 4m-4-4L8 8" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" strokeLinecap="round" />
                </svg>
                <p className="mt-3 text-sm font-semibold text-slate-300">Drop a traffic-sign image here</p>
                <p className="mt-1 text-xs text-slate-500">or click to choose a file (JPG / JPEG / PNG)</p>
              </>
            )}
          </div>
          {file && <p className="mt-3 truncate font-mono text-xs text-slate-500">📄 {file.name}</p>}

          {error && (
            <div className="mt-4 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
              ⚠️ {error}
            </div>
          )}

          <button
            onClick={onPredict}
            disabled={!file || !ready || loading}
            className={cn(
              "mt-5 w-full rounded-xl px-6 py-3 text-sm font-bold transition",
              !file || !ready || loading
                ? "cursor-not-allowed bg-slate-800 text-slate-500"
                : "bg-gradient-to-r from-teal-400 to-emerald-500 text-slate-950 shadow-lg shadow-teal-500/25 hover:brightness-110",
            )}
          >
            {loading
              ? "Predicting…"
              : !ready
                ? phase === "error"
                  ? "Training Failed"
                  : `Training CNN… epoch ${epoch}/${totalEpochs}`
                : "🚀 2 · Predict with CNN"}
          </button>
          {!ready && phase !== "error" && (
            <p className="mt-3 text-center text-xs text-slate-500">
              The model is training in your browser right now (epoch {epoch}/{totalEpochs}) — predictions unlock when done.
            </p>
          )}
          {phase === "error" && modelError && (
            <div className="mt-4 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
              ⚠️ Training Failed: {modelError}
            </div>
          )}
        </Card>

        {/* ---------------------------- preprocessing ------------------------------ */}
        <Card className="p-6">
          <h3 className="text-lg font-bold text-white">Preprocessing pipeline</h3>
          <p className="mt-1 text-sm text-slate-400">Same steps as <code className="font-mono text-teal-300">predict.py</code> / OpenCV.</p>
          <div className="mt-4 grid grid-cols-3 gap-3">
            {[
              { label: "① Original", img: preview, note: "uploaded file" },
              { label: "② Resized 32×32", img: result?.stages.resized, note: "cv2.resize / drawImage" },
              { label: "③ Normalized", img: result?.stages.normalized, note: "pixels ÷ 255 → [0, 1]" },
            ].map((s) => (
              <div key={s.label} className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-center">
                {s.img ? (
                  <img
                    src={s.img}
                    alt={s.label}
                    className={cn("mx-auto h-24 w-24 rounded-md object-contain", s.label.startsWith("②") || s.label.startsWith("③") ? "[image-rendering:pixelated]" : "")}
                  />
                ) : (
                  <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-md bg-slate-900 text-xs text-slate-600">
                    {s.label.startsWith("①") ? "no image" : "after predict"}
                  </div>
                )}
                <p className="mt-2 text-xs font-bold text-slate-300">{s.label}</p>
                <p className="font-mono text-[10px] text-slate-500">{s.note}</p>
              </div>
            ))}
          </div>
          {result && (
            <p className="mt-3 font-mono text-[11px] text-slate-500">
              tensor shape (1, 32, 32, 3) · dtype float32 · mean pixel = {result.stages.mean.toFixed(3)} · range [0.0, 1.0]
            </p>
          )}

          {/* ------------------------------ result ------------------------------ */}
          {result && (
            <div className="mt-5 rounded-xl border border-teal-500/30 bg-teal-500/5 p-5">
              <div className="flex items-center gap-4">
                <img
                  src={signThumbnailDataURL(result.classId, 96)}
                  alt={result.name}
                  className="h-20 w-20 rounded-lg border border-slate-700 bg-slate-100 object-contain"
                />
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="teal">predicted class [{String(result.classId).padStart(2, "0")}]</Badge>
                    {result.modelSource && (
                      <span className="inline-flex items-center rounded-md border border-slate-700 bg-slate-900/90 px-2 py-0.5 text-[10px] font-semibold text-slate-300">
                        ⚡ {result.modelSource}
                      </span>
                    )}
                  </div>
                  <h4 className="mt-1.5 text-xl font-extrabold text-white">{result.name}</h4>
                  <p className="text-sm font-bold text-teal-300">confidence { (result.confidence * 100).toFixed(2) }%</p>
                </div>
              </div>
              <div className="mt-4 space-y-2.5">
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Top-3 predictions</p>
                {result.top3.map((t, i) => (
                  <div key={t.classId}>
                    <div className="flex justify-between text-xs">
                      <span className={cn("font-semibold", i === 0 ? "text-teal-300" : "text-slate-300")}>
                        {i + 1}. [{String(t.classId).padStart(2, "0")}] {t.name}
                      </span>
                      <span className="font-mono text-slate-400">{(t.confidence * 100).toFixed(2)}%</span>
                    </div>
                    <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className={cn("h-full rounded-full", i === 0 ? "bg-gradient-to-r from-teal-400 to-emerald-400" : "bg-slate-600")}
                        style={{ width: `${Math.max(2, t.confidence * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              {/* ----------------- Short explanation of predicted class ----------------- */}
              {result.metadata && (
                <div className="mt-4 rounded-lg border border-slate-800 bg-slate-900/60 p-3 text-xs text-slate-300">
                  <div className="flex items-center gap-2 font-bold text-teal-300">
                    <span>💡 Sign Category: {result.metadata.category}</span>
                  </div>
                  <p className="mt-1 leading-relaxed text-slate-400">{result.metadata.description}</p>
                  <p className="mt-1 font-mono text-[10px] text-slate-500">Key visual markers: {result.metadata.visualFeatures}</p>
                </div>
              )}

              {/* ----------------- Grad-CAM explainability ----------------- */}
              {result.gradCam && (
                <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/70 p-3.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-200">🔍 Grad-CAM Attention Heatmap</span>
                    <a href="#inside-cnn" className="text-[11px] font-semibold text-teal-400 hover:underline">
                      Explore all layers ↓
                    </a>
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
                    Highlights which spatial regions most strongly influenced the CNN's prediction of this sign.
                  </p>
                  <div className="mt-3 grid grid-cols-3 gap-2 text-center text-[10px]">
                    <div>
                      <img src={result.stages.resized} alt="32x32 input" className="mx-auto h-20 w-20 rounded-md border border-slate-800 [image-rendering:pixelated]" />
                      <span className="mt-1 block text-slate-400">Input (32×32)</span>
                    </div>
                    <div>
                      <img src={result.gradCam.heatmapUrl} alt="Grad-CAM Heatmap" className="mx-auto h-20 w-20 rounded-md border border-slate-800" />
                      <span className="mt-1 block text-teal-400">Grad-CAM Heatmap</span>
                    </div>
                    <div>
                      <img src={result.gradCam.overlayUrl} alt="Grad-CAM Overlay" className="mx-auto h-20 w-20 rounded-md border border-slate-800" />
                      <span className="mt-1 block text-emerald-400">Attention Overlay</span>
                    </div>
                  </div>
                  <p className="mt-2 text-[10px] italic text-slate-500">
                    *Grad-CAM highlights influential feature map activations at layer conv3; it visualizes network sensitivity, not human reasoning.
                  </p>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </section>
  );
}
