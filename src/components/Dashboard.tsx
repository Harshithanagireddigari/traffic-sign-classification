import { useModel } from "@/lib/model";
import { ConfusionHeatmap, LineChart } from "@/lib/charts";
import { Card, SectionHeading } from "./ui";
import { cn } from "@/utils/cn";

export default function Dashboard() {
  const { phase, epoch, totalEpochs, history, evalData, error, retrain } = useModel();

  const training = phase === "preparing" || phase === "training";
  const progress = training ? (phase === "preparing" ? 0.03 : epoch / totalEpochs) : 1;

  return (
    <section>
      <SectionHeading
        id="dashboard"
        eyebrow="Model performance"
        title="Real metrics from the live training run"
        desc="Every number below was computed by the CNN that just trained in your browser — accuracy/loss per epoch, a confusion matrix on the held-out validation split, and a full classification report. The Python train.py produces the same artefacts on the full GTSRB test set."
      />

      <div className="mx-auto mt-10 max-w-6xl space-y-6 px-6">
        {/* status bar */}
        <Card className="p-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Training Status:</span>
                <span
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold",
                    phase === "ready"
                      ? "border border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                      : training
                        ? "border border-sky-400/30 bg-sky-400/10 text-sky-300"
                        : phase === "error"
                          ? "border border-rose-400/30 bg-rose-400/10 text-rose-300"
                          : "border border-slate-700 bg-slate-800 text-slate-400",
                  )}
                >
                  <span
                    className={cn(
                      "inline-block h-2 w-2 rounded-full",
                      phase === "ready"
                        ? "bg-emerald-400"
                        : training
                          ? "pulse-dot bg-sky-400"
                          : phase === "error"
                            ? "bg-rose-400"
                            : "bg-slate-400",
                    )}
                  />
                  {phase === "ready"
                    ? "Model Ready"
                    : training
                      ? `Training (epoch ${epoch}/${totalEpochs})`
                      : phase === "error"
                        ? `Training Failed: ${error ?? "Unknown error"}`
                        : "Starting…"}
                </span>
              </div>
              <h3 className="mt-2 text-lg font-bold text-white">
                {phase === "ready"
                  ? "✅ Model Ready"
                  : training
                    ? "🧠 Training CNN in browser…"
                    : phase === "error"
                      ? `❌ Training Failed: ${error ?? "Unknown error"}`
                      : "⏳ Starting…"}
              </h3>
              <p className="mt-1 text-sm text-slate-400">
                {phase === "ready"
                  ? `Finished ${totalEpochs} epochs · validation accuracy ${((evalData?.valAcc ?? 0) * 100).toFixed(2)}%`
                  : training
                    ? `epoch ${epoch}/${totalEpochs} — Conv2D(32)→MaxPool → Conv2D(64)→MaxPool → Conv2D(128)→MaxPool → Flatten → Dense(256)+ReLU+Dropout → Dense(43)+Softmax`
                    : phase === "error"
                      ? `Error: ${error ?? "Training failed"}`
                      : "…"}
              </p>
            </div>
            <button
              onClick={retrain}
              disabled={training}
              className="rounded-xl border border-slate-700 bg-slate-900 px-5 py-2.5 text-sm font-bold text-slate-200 transition hover:border-teal-400/50 hover:text-teal-300 disabled:opacity-40"
            >
              ↻ Retrain model
            </button>
          </div>
          <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-teal-400 to-emerald-400 transition-all duration-500"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </Card>

        {/* charts */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="p-5">
            <h3 className="text-base font-bold text-white">Accuracy per epoch</h3>
            {history.length > 0 ? (
              <LineChart
                yDomain={[0, 1]}
                series={[
                  { label: "training accuracy", color: "#2dd4bf", values: history.map((h) => h.acc) },
                  { label: "validation accuracy", color: "#818cf8", values: history.map((h) => h.valAcc) },
                ]}
              />
            ) : (
              <p className="py-16 text-center text-sm text-slate-500">Waiting for first epoch…</p>
            )}
          </Card>
          <Card className="p-5">
            <h3 className="text-base font-bold text-white">Loss per epoch</h3>
            {history.length > 0 ? (
              <LineChart
                series={[
                  { label: "training loss", color: "#fbbf24", values: history.map((h) => h.loss) },
                  { label: "validation loss", color: "#fb7185", values: history.map((h) => h.valLoss) },
                ]}
              />
            ) : (
              <p className="py-16 text-center text-sm text-slate-500">Waiting for first epoch…</p>
            )}
          </Card>
        </div>

        {/* confusion matrix + report */}
        {evalData && (
          <>
            <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
              <Card className="p-5">
                <h3 className="text-base font-bold text-white">Confusion matrix (validation split)</h3>
                <p className="mt-1 text-xs text-slate-500">
                  Rows = true class, columns = predicted class. Hover a cell for details. Val accuracy:{" "}
                  <span className="font-bold text-teal-300">{(evalData.valAcc * 100).toFixed(2)}%</span> · val loss:{" "}
                  <span className="font-bold text-amber-300">{evalData.valLoss.toFixed(4)}</span>
                </p>
                <div className="mt-4">
                  <ConfusionHeatmap matrix={evalData.confusion} />
                </div>
                <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
                  <span>0</span>
                  <div className="h-2.5 flex-1 rounded-full" style={{ background: "linear-gradient(to right, rgb(15,23,42), rgb(52,211,153))" }} />
                  <span>max</span>
                </div>
              </Card>

              <Card className="p-5">
                <h3 className="text-base font-bold text-white">Classification report</h3>
                <p className="mt-1 text-xs text-slate-500">
                  Macro avg — precision <span className="font-mono text-slate-300">{(evalData.macro.precision * 100).toFixed(2)}%</span> ·
                  recall <span className="font-mono text-slate-300">{(evalData.macro.recall * 100).toFixed(2)}%</span> · f1{" "}
                  <span className="font-mono text-slate-300">{(evalData.macro.f1 * 100).toFixed(2)}%</span>
                </p>
                <div className="mt-4 max-h-[420px] overflow-auto rounded-lg border border-slate-800">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-slate-900 text-slate-400">
                      <tr>
                        <th className="px-3 py-2.5 font-semibold">#</th>
                        <th className="px-3 py-2.5 font-semibold">class</th>
                        <th className="px-3 py-2.5 text-right font-semibold">precision</th>
                        <th className="px-3 py-2.5 text-right font-semibold">recall</th>
                        <th className="px-3 py-2.5 text-right font-semibold">f1</th>
                        <th className="px-3 py-2.5 text-right font-semibold">support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalData.report.map((r, i) => (
                        <tr key={i} className="border-t border-slate-800/70 hover:bg-slate-800/40">
                          <td className="px-3 py-2 font-mono text-slate-500">{String(i).padStart(2, "0")}</td>
                          <td className="px-3 py-2 text-slate-300">{r.name}</td>
                          <td className="px-3 py-2 text-right font-mono text-slate-400">{(r.precision * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right font-mono text-slate-400">{(r.recall * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right font-mono text-slate-400">{(r.f1 * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right font-mono text-slate-500">{r.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
