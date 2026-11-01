import { useState, useEffect } from "react";
import { useModel } from "@/lib/model";
import { Card, SectionHeading, Badge } from "./ui";
import { LineChart } from "@/lib/charts";
import { CLASS_NAMES, signThumbnailDataURL } from "@/lib/gtsrb";
import { SIGN_METADATA, type SignClassInfo } from "@/lib/signMetadata";
import { generateSampleImagesForClass } from "@/lib/cnnExplainability";
import { cn } from "@/utils/cn";

export default function InsideTheCNN() {
  const {
    phase,
    history,
    evalData,
    totalEpochs,
    trainPerClass,
    valPerClass,
    lastPrediction,
    explainSampleSign,
  } = useModel();

  // Dataset Explorer state
  const [selectedClassId, setSelectedClassId] = useState<number>(14); // Default: Stop sign (class 14)
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [categoryFilter, setCategoryFilter] = useState<string>("All");
  const [explaining, setExplaining] = useState<boolean>(false);

  // Active prediction to visualize (either from uploaded file or sampled sign)
  const [activePrediction, setActivePrediction] = useState(lastPrediction);

  // Synchronize when a new prediction happens in Demo.tsx
  useEffect(() => {
    if (lastPrediction) {
      setActivePrediction(lastPrediction);
    }
  }, [lastPrediction]);

  // Update sample images whenever selected class changes
  useEffect(() => {
    try {
      const imgs = generateSampleImagesForClass(selectedClassId, 8);
      setSampleImages(imgs);
    } catch (e) {
      console.error("Error generating sample images:", e);
    }
  }, [selectedClassId]);

  // If user clicks "Inspect this sign through CNN"
  const handleInspectClass = async (cId: number) => {
    if (phase !== "ready") return;
    setExplaining(true);
    try {
      const res = await explainSampleSign(cId);
      setActivePrediction(res);
    } catch (err) {
      console.error("Failed to inspect class:", err);
    } finally {
      setExplaining(false);
    }
  };

  const selectedMeta: SignClassInfo = SIGN_METADATA[selectedClassId] || {
    id: selectedClassId,
    name: CLASS_NAMES[selectedClassId],
    category: "Regulatory / Prohibitory",
    description: "Standard road traffic sign.",
    visualFeatures: "Distinctive geometric shape and color contrast.",
  };

  const categories = [
    "All",
    "Speed Limit",
    "Warning / Hazard",
    "Regulatory / Prohibitory",
    "Mandatory",
    "Priority",
    "Derestriction",
  ];

  const filteredSigns = SIGN_METADATA.filter(
    (s) => categoryFilter === "All" || s.category === categoryFilter,
  );

  const totalPerClass = trainPerClass + valPerClass;
  const totalDatasetImages = CLASS_NAMES.length * totalPerClass;
  const totalTrainImages = CLASS_NAMES.length * trainPerClass;
  const totalValImages = CLASS_NAMES.length * valPerClass;

  const currentAcc = history.length > 0 ? history[history.length - 1].acc : 0;
  const currentLoss = history.length > 0 ? history[history.length - 1].loss : 0;
  const currentValAcc = evalData?.valAcc ?? (history.length > 0 ? history[history.length - 1].valAcc : 0);
  const currentValLoss = evalData?.valLoss ?? (history.length > 0 ? history[history.length - 1].valLoss : 0);

  return (
    <section id="inside-cnn" className="relative scroll-mt-20">
      <SectionHeading
        id="inside-cnn-heading"
        eyebrow="Explainable AI & Interpretability"
        title="Inside the CNN — What Did It Learn?"
        desc="Peel back the layers of the Convolutional Neural Network. Inspect the actual dataset samples, intermediate feature activations from early to deep layers, and Grad-CAM gradient attention heatmaps computed directly from the trained model."
      />

      <div className="mx-auto mt-10 max-w-6xl space-y-8 px-6">
        {/* ========================================================================= */}
        {/* FEATURE 6: Model Architecture & Specifications                           */}
        {/* ========================================================================= */}
        <Card className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-teal-400">Feature 6 · Specifications</span>
              <h3 className="mt-1 text-xl font-extrabold text-white">CNN Architecture &amp; Hyperparameters</h3>
            </div>
            <Badge tone="teal">Sequential CNN (3 Blocks)</Badge>
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Input Dimensions</span>
              <p className="mt-1 font-mono text-base font-bold text-white">32 × 32 × 3 RGB</p>
              <p className="text-[11px] text-slate-400">Pixels normalized to [0, 1]</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Convolutional Layers</span>
              <p className="mt-1 font-mono text-base font-bold text-white">3 Conv2D Blocks</p>
              <p className="text-[11px] text-slate-400">32 → 64 → 128 filters (3×3)</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Activations &amp; Pooling</span>
              <p className="mt-1 font-mono text-base font-bold text-white">ReLU + MaxPool</p>
              <p className="text-[11px] text-slate-400">3× MaxPool(2×2) + Softmax</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Loss &amp; Optimizer</span>
              <p className="mt-1 font-mono text-base font-bold text-white">Adam (lr=0.001)</p>
              <p className="text-[11px] text-slate-400">sparseCategoricalCrossentropy</p>
            </div>
          </div>

          {/* Interactive Layer Flow Diagram */}
          <div className="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/40 p-4 text-xs font-mono">
            <span className="rounded-md bg-slate-800 px-2.5 py-1 text-slate-300">Input [32,32,3]</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-teal-950/70 text-teal-300 border border-teal-500/30 px-2.5 py-1">Conv1 (32) + ReLU</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-slate-800 px-2 py-1 text-slate-400">Pool1 (2×2)</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-teal-950/70 text-teal-300 border border-teal-500/30 px-2.5 py-1">Conv2 (64) + ReLU</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-slate-800 px-2 py-1 text-slate-400">Pool2 (2×2)</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-teal-950/70 text-teal-300 border border-teal-500/30 px-2.5 py-1">Conv3 (128) + ReLU</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-slate-800 px-2 py-1 text-slate-400">Pool3 (2×2)</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-indigo-950/70 text-indigo-300 border border-indigo-500/30 px-2.5 py-1">Dense (256) + Drop(0.5)</span>
            <span className="text-slate-600">→</span>
            <span className="rounded-md bg-emerald-950/70 text-emerald-300 border border-emerald-500/30 px-2.5 py-1 font-bold">Dense (43 Softmax)</span>
          </div>
        </Card>

        {/* ========================================================================= */}
        {/* FEATURE 1: Training Dataset Explorer                                      */}
        {/* ========================================================================= */}
        <Card className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-teal-400">Feature 1 · Dataset Inspection</span>
              <h3 className="mt-1 text-xl font-extrabold text-white">Training Dataset Explorer</h3>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Category:</span>
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-200 outline-none focus:border-teal-400"
              >
                {categories.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Academic distinction banner */}
          <div className="mt-4 rounded-xl border border-sky-500/30 bg-sky-500/10 p-3.5 text-xs text-slate-300">
            <span className="font-bold text-sky-300">📌 Dataset Origin &amp; Integrity:</span> The images shown below represent the actual procedurally generated training &amp; validation samples rendered in-browser for real-time training (featuring randomized affine rotation, scale, brightness/contrast jitter, and Gaussian noise). The standalone Python model trains on the official GTSRB dataset (~39,000 real photographic PPM images).
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-[1.2fr_1.8fr]">
            {/* Left: Class picker & details */}
            <div className="space-y-4">
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-400">
                Select Class to Explore ({filteredSigns.length} available):
              </label>
              <select
                value={selectedClassId}
                onChange={(e) => setSelectedClassId(Number(e.target.value))}
                className="w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm font-bold text-white outline-none transition focus:border-teal-400"
              >
                {filteredSigns.map((s) => (
                  <option key={s.id} value={s.id}>
                    [{String(s.id).padStart(2, "0")}] {s.name} ({s.category})
                  </option>
                ))}
              </select>

              {/* Class summary card */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                <div className="flex items-center gap-3">
                  <img
                    src={signThumbnailDataURL(selectedClassId, 72)}
                    alt={selectedMeta.name}
                    className="h-16 w-16 rounded-lg border border-slate-700 bg-slate-100 object-contain p-1"
                  />
                  <div>
                    <Badge tone="teal">{selectedMeta.category}</Badge>
                    <h4 className="mt-1 font-bold text-white">
                      [{String(selectedClassId).padStart(2, "0")}] {selectedMeta.name}
                    </h4>
                  </div>
                </div>

                <p className="mt-3 text-xs leading-relaxed text-slate-400">{selectedMeta.description}</p>
                <div className="mt-2 text-[11px] text-slate-500">
                  <strong className="text-slate-400">CNN Key Features:</strong> {selectedMeta.visualFeatures}
                </div>

                <div className="mt-4 grid grid-cols-2 gap-2 border-t border-slate-800 pt-3 text-center">
                  <div className="rounded-lg bg-slate-900 p-2">
                    <span className="text-[10px] uppercase text-slate-500">Training Samples</span>
                    <p className="font-mono text-sm font-bold text-teal-300">{trainPerClass} images</p>
                  </div>
                  <div className="rounded-lg bg-slate-900 p-2">
                    <span className="text-[10px] uppercase text-slate-500">Validation Samples</span>
                    <p className="font-mono text-sm font-bold text-emerald-300">{valPerClass} images</p>
                  </div>
                </div>

                <button
                  onClick={() => handleInspectClass(selectedClassId)}
                  disabled={phase !== "ready" || explaining}
                  className="mt-4 w-full rounded-xl bg-gradient-to-r from-teal-400 to-emerald-500 px-4 py-2.5 text-xs font-bold text-slate-950 shadow-lg shadow-teal-500/20 transition hover:brightness-110 disabled:opacity-40"
                >
                  {explaining ? "Computing activations…" : `⚡ Feed this sign into CNN to visualize features & Grad-CAM`}
                </button>
              </div>
            </div>

            {/* Right: Actual training samples */}
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Actual Synthesized Training Samples (Class {selectedClassId})
                </span>
                <span className="text-xs text-slate-500">{sampleImages.length} sample variations</span>
              </div>
              <p className="mt-1 text-xs text-slate-400">
                Notice the rotation, lighting shifts, and Gaussian blur — the exact augmentations that force the CNN to learn invariant sign motifs.
              </p>

              <div className="mt-4 grid grid-cols-4 gap-3">
                {sampleImages.map((url, idx) => (
                  <div
                    key={idx}
                    className="group relative rounded-xl border border-slate-800 bg-slate-950 p-2 text-center transition hover:border-teal-400/50"
                  >
                    <img
                      src={url}
                      alt={`Sample ${idx}`}
                      className="mx-auto h-20 w-20 rounded-md object-contain [image-rendering:pixelated]"
                    />
                    <span className="mt-1.5 block font-mono text-[10px] text-slate-500">Sample #{idx + 1}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>

        {/* ========================================================================= */}
        {/* FEATURE 2 & 3: Intermediate Feature Maps & Grad-CAM                       */}
        {/* ========================================================================= */}
        <Card className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-teal-400">
                Feature 2 &amp; 3 · Visualizing Layers &amp; Attention
              </span>
              <h3 className="mt-1 text-xl font-extrabold text-white">Intermediate Feature Maps &amp; Grad-CAM</h3>
            </div>
            {activePrediction && (
              <Badge tone="teal">
                Active sign: [{String(activePrediction.classId).padStart(2, "0")}] {activePrediction.name}
              </Badge>
            )}
          </div>

          {!activePrediction ? (
            <div className="my-8 rounded-xl border border-dashed border-slate-800 p-8 text-center">
              <p className="text-sm text-slate-400">
                No active sign loaded yet. Upload an image in the Live Demo above or select a sign in the Dataset Explorer and click <strong>&quot;Feed this sign into CNN&quot;</strong>.
              </p>
              <button
                onClick={() => handleInspectClass(14)}
                disabled={phase !== "ready"}
                className="mt-4 rounded-xl bg-teal-400 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-teal-300 disabled:opacity-40"
              >
                Inspect Stop Sign (Default)
              </button>
            </div>
          ) : (
            <div className="mt-6 space-y-8">
              {/* --- FEATURE 3: GRAD-CAM EXPLAINABILITY --- */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-5">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                  <div>
                    <h4 className="font-bold text-white flex items-center gap-2">
                      <span>🎯 Grad-CAM Attention Heatmap</span>
                      <span className="text-xs font-normal text-slate-400">(Gradient-weighted Class Activation Mapping)</span>
                    </h4>
                    <p className="mt-1 text-xs text-slate-400">
                      Calculates the gradients of the score for <span className="text-teal-300 font-bold">{activePrediction.name}</span> with respect to the feature map activations of layer <code className="font-mono text-teal-300">conv3</code>.
                    </p>
                  </div>
                  <Badge tone="sky">Output of Conv3</Badge>
                </div>

                <div className="mt-5 grid grid-cols-1 gap-6 sm:grid-cols-3 text-center">
                  {/* 1. Input Image */}
                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <span className="text-xs font-bold text-slate-300">1. Preprocessed Input</span>
                    <p className="text-[10px] text-slate-500 font-mono">32×32 pixels · Normalized</p>
                    <img
                      src={activePrediction.stages.resized}
                      alt="Input sign"
                      className="mx-auto mt-3 h-32 w-32 rounded-lg border border-slate-700 object-contain shadow-lg [image-rendering:pixelated]"
                    />
                  </div>

                  {/* 2. Pure Grad-CAM Heatmap */}
                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <span className="text-xs font-bold text-teal-300">2. Grad-CAM Heatmap</span>
                    <p className="text-[10px] text-slate-500 font-mono">Normalized Gradients · Jet Scale</p>
                    {activePrediction.gradCam ? (
                      <img
                        src={activePrediction.gradCam.heatmapUrl}
                        alt="Grad-CAM Heatmap"
                        className="mx-auto mt-3 h-32 w-32 rounded-lg border border-slate-700 object-contain shadow-lg"
                      />
                    ) : (
                      <div className="mx-auto mt-3 flex h-32 w-32 items-center justify-center rounded-lg bg-slate-950 text-xs text-slate-600">
                        Generating…
                      </div>
                    )}
                  </div>

                  {/* 3. Attention Overlay */}
                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <span className="text-xs font-bold text-emerald-300">3. Attention Overlay</span>
                    <p className="text-[10px] text-slate-500 font-mono">Input + 60% Activation Blend</p>
                    {activePrediction.gradCam ? (
                      <img
                        src={activePrediction.gradCam.overlayUrl}
                        alt="Grad-CAM Overlay"
                        className="mx-auto mt-3 h-32 w-32 rounded-lg border border-slate-700 object-contain shadow-lg"
                      />
                    ) : (
                      <div className="mx-auto mt-3 flex h-32 w-32 items-center justify-center rounded-lg bg-slate-950 text-xs text-slate-600">
                        Generating…
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-4 rounded-lg bg-slate-900/80 p-3 text-xs leading-relaxed text-slate-400">
                  <strong className="text-slate-300">Explainability Limitation:</strong> Grad-CAM highlights which spatial receptive fields exerted the strongest positive gradient pull toward class [{activePrediction.classId}]. Red/yellow regions indicate high sensitivity, while blue regions indicate neutral or negligible impact. This proves <em>which parts of the image the model focused on</em>, but does not represent symbolic or human reasoning.
                </div>
              </div>

              {/* --- FEATURE 2: INTERMEDIATE FEATURE MAPS --- */}
              <div className="space-y-6">
                <div>
                  <h4 className="text-base font-bold text-white flex items-center gap-2">
                    <span>🔬 Intermediate Feature Map Activations</span>
                  </h4>
                  <p className="mt-1 text-xs text-slate-400">
                    Real tensor outputs as the sign passes through each of the 3 convolutional layers.
                  </p>
                </div>

                {activePrediction.featureMaps?.map((layer, lIdx) => (
                  <div key={layer.layerName} className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2">
                      <div>
                        <span className="text-xs font-bold text-teal-300">{layer.stageName}</span>
                        <p className="font-mono text-[11px] text-slate-500">
                          Layer: {layer.layerName} · Output shape: [{layer.shape.join(", ")}]
                        </p>
                      </div>
                      <span className="text-[11px] text-slate-400">
                        Showing first {layer.channelDataUrls.length} filter channels
                      </span>
                    </div>

                    <p className="mt-2 text-xs leading-relaxed text-slate-300">
                      <strong className="text-slate-200">What this layer learned:</strong> {layer.patternDescription}
                    </p>

                    <div className="mt-3 grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-12">
                      {layer.channelDataUrls.map((url, chIdx) => (
                        <div
                          key={chIdx}
                          className="rounded-lg border border-slate-800/80 bg-slate-900 p-1 text-center transition hover:border-teal-400/50"
                        >
                          <img
                            src={url}
                            alt={`Filter ${chIdx}`}
                            className="mx-auto h-12 w-12 rounded object-contain [image-rendering:pixelated]"
                          />
                          <span className="mt-1 block font-mono text-[9px] text-slate-400">ch #{chIdx}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* ========================================================================= */}
        {/* FEATURE 4: Training History & Performance Graphs                         */}
        {/* ========================================================================= */}
        <Card className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-teal-400">Feature 4 · Real Metrics</span>
              <h3 className="mt-1 text-xl font-extrabold text-white">Training History &amp; Convergence</h3>
            </div>
            <Badge tone="teal">{history.length} Epochs Recorded</Badge>
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5 text-center">
              <span className="text-[11px] uppercase text-slate-500">Train Accuracy</span>
              <p className="mt-1 text-2xl font-black text-teal-300">{(currentAcc * 100).toFixed(2)}%</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5 text-center">
              <span className="text-[11px] uppercase text-slate-500">Validation Accuracy</span>
              <p className="mt-1 text-2xl font-black text-emerald-400">{(currentValAcc * 100).toFixed(2)}%</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5 text-center">
              <span className="text-[11px] uppercase text-slate-500">Train Loss</span>
              <p className="mt-1 font-mono text-2xl font-black text-slate-200">{currentLoss.toFixed(4)}</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5 text-center">
              <span className="text-[11px] uppercase text-slate-500">Validation Loss</span>
              <p className="mt-1 font-mono text-2xl font-black text-slate-200">{currentValLoss.toFixed(4)}</p>
            </div>
          </div>

          {/* Graphs */}
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-white">Epoch vs Accuracy</span>
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="inline-flex items-center gap-1 text-teal-400">
                    <span className="h-2 w-2 rounded-full bg-teal-400" /> Train Acc
                  </span>
                  <span className="inline-flex items-center gap-1 text-emerald-400">
                    <span className="h-2 w-2 rounded-full bg-emerald-400" /> Val Acc
                  </span>
                </div>
              </div>
              {history.length > 0 ? (
                <LineChart
                  yDomain={[0, 1]}
                  series={[
                    { label: "Train Accuracy", color: "#2dd4bf", values: history.map((h) => h.acc) },
                    { label: "Val Accuracy", color: "#34d399", values: history.map((h) => h.valAcc) },
                  ]}
                />
              ) : (
                <div className="flex h-48 items-center justify-center text-xs text-slate-500">
                  Training in progress…
                </div>
              )}
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-white">Epoch vs Loss</span>
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="inline-flex items-center gap-1 text-amber-400">
                    <span className="h-2 w-2 rounded-full bg-amber-400" /> Train Loss
                  </span>
                  <span className="inline-flex items-center gap-1 text-rose-400">
                    <span className="h-2 w-2 rounded-full bg-rose-400" /> Val Loss
                  </span>
                </div>
              </div>
              {history.length > 0 ? (
                <LineChart
                  series={[
                    { label: "Train Loss", color: "#fbbf24", values: history.map((h) => h.loss) },
                    { label: "Val Loss", color: "#fb7185", values: history.map((h) => h.valLoss) },
                  ]}
                />
              ) : (
                <div className="flex h-48 items-center justify-center text-xs text-slate-500">
                  Training in progress…
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4 text-xs text-slate-400">
            <span className="font-bold text-slate-300">Dataset Configuration:</span> {totalDatasetImages} total images ({totalTrainImages} training + {totalValImages} validation across {CLASS_NAMES.length} classes). Trained for {totalEpochs} epochs with batch size 64 using Adam optimizer and sparse categorical crossentropy. Every metric on this page is genuinely measured from the TensorFlow.js training loop.
          </div>
        </Card>
      </div>
    </section>
  );
}
