/**
 * model.tsx
 * ---------
 * React context that owns a REAL CNN built with TensorFlow.js.
 *
 * On first load it:
 *   1. synthesises a GTSRB-style dataset (procedurally rendered signs,
 *      augmented with rotation / scale / brightness / noise — same spirit as
 *      the Python pipeline: resize to 32x32 and normalize to [0, 1]),
 *   2. trains a CNN (Conv2D+ReLU -> MaxPool x3 -> Flatten -> Dense -> Softmax),
 *   3. evaluates it on a held-out validation split and computes a genuine
 *      confusion matrix + classification report.
 *
 * `predict(file)` runs the trained model on an uploaded image and returns the
 * top-3 classes with real softmax confidences — nothing is hardcoded.
 */
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import * as tf from "@tensorflow/tfjs";
import { CLASS_NAMES, generateDataset, IMG, renderSample } from "./gtsrb";
import {
  extractFeatureMaps,
  generateGradCam,
  type GradCamResult,
  type LayerFeatureMaps,
} from "./cnnExplainability";
import { SIGN_METADATA, type SignClassInfo } from "./signMetadata";

export const TOTAL_EPOCHS = 16;
export const TRAIN_PER_CLASS = 34;
export const VAL_PER_CLASS = 8;

export interface HistoryPoint {
  epoch: number;
  acc: number;
  loss: number;
  valAcc: number;
  valLoss: number;
}

export interface ReportRow {
  name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EvalData {
  valAcc: number;
  valLoss: number;
  confusion: number[][];
  report: ReportRow[];
  macro: { precision: number; recall: number; f1: number };
}

export interface StageImages {
  original: string;
  resized: string;
  normalized: string;
  mean: number;
}

export interface Prediction {
  classId: number;
  name: string;
  confidence: number;
  top3: Array<{ classId: number; name: string; confidence: number }>;
  stages: StageImages;
  modelSource?: string;
  gradCam?: GradCamResult | null;
  featureMaps?: LayerFeatureMaps[];
  metadata?: SignClassInfo;
}

export type Phase = "idle" | "preparing" | "training" | "ready" | "error";

interface ModelContextValue {
  phase: Phase;
  epoch: number;
  totalEpochs: number;
  trainPerClass: number;
  valPerClass: number;
  history: HistoryPoint[];
  evalData: EvalData | null;
  error: string | null;
  ready: boolean;
  lastPrediction: Prediction | null;
  predict: (file: File) => Promise<Prediction>;
  explainSampleSign: (classId: number) => Promise<Prediction>;
  retrain: () => void;
}

const ModelContext = createContext<ModelContextValue | null>(null);

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error("useModel must be used inside <ModelProvider>");
  return ctx;
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("decode failed"));
    img.src = url;
  });
}

function buildCnn(): tf.LayersModel {
  // Same architecture as train.py:
  // Conv2D(32)+ReLU -> MaxPool -> Conv2D(64)+ReLU -> MaxPool ->
  // Conv2D(128)+ReLU -> MaxPool -> Flatten -> Dense(256)+ReLU -> Dropout -> Dense(43)+Softmax
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      name: "conv1",
      inputShape: [IMG, IMG, 3],
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
    }),
  );
  model.add(tf.layers.maxPooling2d({ name: "pool1", poolSize: 2 }));
  model.add(tf.layers.conv2d({ name: "conv2", filters: 64, kernelSize: 3, activation: "relu", padding: "same" }));
  model.add(tf.layers.maxPooling2d({ name: "pool2", poolSize: 2 }));
  model.add(tf.layers.conv2d({ name: "conv3", filters: 128, kernelSize: 3, activation: "relu", padding: "same" }));
  model.add(tf.layers.maxPooling2d({ name: "pool3", poolSize: 2 }));
  model.add(tf.layers.flatten({ name: "flatten" }));
  model.add(tf.layers.dense({ name: "dense1", units: 256, activation: "relu" }));
  model.add(tf.layers.dropout({ name: "dropout", rate: 0.5 }));
  model.add(tf.layers.dense({ name: "output", units: CLASS_NAMES.length, activation: "softmax" }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export function ModelProvider({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [epoch, setEpoch] = useState(0);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [evalData, setEvalData] = useState<EvalData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null);

  const modelRef = useRef<tf.LayersModel | null>(null);
  const runningRef = useRef(false);

  const train = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    try {
      setError(null);
      setHistory([]);
      setEvalData(null);
      setEpoch(0);
      setPhase("preparing");

      // prefer WebGL (GPU) for fast training, fall back to CPU
      try {
        await tf.setBackend("webgl");
      } catch {
        await tf.setBackend("cpu");
      }
      await tf.ready();

      // 1. synthetic GTSRB-style dataset: 32x32, normalized to [0, 1]
      const data = generateDataset(TRAIN_PER_CLASS, VAL_PER_CLASS);
      const xs = tf.tensor4d(data.xs, [data.total, IMG, IMG, 3], "float32");
      // Convert label tensor to float32 preserving integer class IDs as whole numbers (e.g. 0.0, 1.0, ..., 42.0)
      // TensorFlow.js sparseCategoricalCrossentropy requires float32 target for internal floor() operation
      const ys = tf.tensor1d(new Float32Array(data.ys), "float32");
      const xTrain = xs.slice([0, 0, 0, 0], [data.trainCount, IMG, IMG, 3]);
      const yTrain = ys.slice([0], [data.trainCount]);
      const xVal = xs.slice([data.trainCount, 0, 0, 0], [-1, IMG, IMG, 3]);
      const yVal = ys.slice([data.trainCount], [-1]);
      xs.dispose();
      ys.dispose();

      // 2. build + train the CNN, streaming real metrics into state
      setPhase("training");
      const model = buildCnn();
      await model.fit(xTrain, yTrain, {
        epochs: TOTAL_EPOCHS,
        batchSize: 64,
        shuffle: true,
        validationData: [xVal, yVal],
        callbacks: {
          onEpochEnd: (e, logs) => {
            const l = logs as Record<string, number>;
            setEpoch(e + 1);
            setHistory((h) => [
              ...h,
              {
                epoch: e + 1,
                acc: l["acc"] ?? l["accuracy"] ?? 0,
                loss: l["loss"] ?? 0,
                valAcc: l["val_acc"] ?? l["val_accuracy"] ?? 0,
                valLoss: l["val_loss"] ?? 0,
              },
            ]);
          },
        },
      });
      xTrain.dispose();
      yTrain.dispose();

      // 3. evaluate on the held-out validation split (real numbers only)
      const evalTensors = model.evaluate(xVal, yVal) as tf.Scalar[];
      const valLoss = (await evalTensors[0].data())[0];
      const valAcc = (await evalTensors[1].data())[0];
      evalTensors.forEach((t) => t.dispose());

      const predT = model.predict(xVal) as tf.Tensor2D;
      const predLabels = (await predT.argMax(1).data()) as Int32Array;
      const trueLabels = Int32Array.from(await yVal.data());
      predT.dispose();
      xVal.dispose();
      yVal.dispose();

      const n = CLASS_NAMES.length;
      const confusion: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
      for (let i = 0; i < trueLabels.length; i++) confusion[trueLabels[i]][predLabels[i]]++;

      const report: ReportRow[] = [];
      let pSum = 0;
      let rSum = 0;
      let fSum = 0;
      for (let c = 0; c < n; c++) {
        const tp = confusion[c][c];
        let fp = 0;
        let fn = 0;
        for (let k = 0; k < n; k++) {
          if (k !== c) {
            fp += confusion[k][c];
            fn += confusion[c][k];
          }
        }
        const support = tp + fn;
        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = support > 0 ? tp / support : 0;
        const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
        report.push({ name: CLASS_NAMES[c], precision, recall, f1, support });
        pSum += precision;
        rSum += recall;
        fSum += f1;
      }
      setEvalData({
        valAcc,
        valLoss,
        confusion,
        report,
        macro: { precision: pSum / n, recall: rSum / n, f1: fSum / n },
      });

      modelRef.current = model;
      setPhase("ready");
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Training failed");
      setPhase("error");
    } finally {
      runningRef.current = false;
    }
  }, []);

  useEffect(() => {
    void train();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const retrain = useCallback(() => {
    if (runningRef.current) return;
    modelRef.current?.dispose();
    modelRef.current = null;
    setLastPrediction(null);
    void train();
  }, [train]);

  const explainSampleSign = useCallback(async (classId: number): Promise<Prediction> => {
    const model = modelRef.current;
    if (!model || phase !== "ready") throw new Error("The CNN is still training — please wait until it finishes.");

    const raw = renderSample(classId); // Float32Array [32*32*3]
    const c32 = document.createElement("canvas");
    c32.width = c32.height = IMG;
    const cctx = c32.getContext("2d")!;
    const idata = cctx.createImageData(IMG, IMG);
    let sum = 0;
    for (let p = 0; p < IMG * IMG; p++) {
      for (let ch = 0; ch < 3; ch++) {
        const v = raw[p * 3 + ch];
        idata.data[p * 4 + ch] = Math.round(v * 255);
        sum += v;
      }
      idata.data[p * 4 + 3] = 255;
    }
    cctx.putImageData(idata, 0, 0);

    const up = (src: HTMLCanvasElement) => {
      const c = document.createElement("canvas");
      c.width = c.height = 160;
      const x = c.getContext("2d")!;
      x.imageSmoothingEnabled = false;
      x.drawImage(src, 0, 0, 160, 160);
      return c.toDataURL("image/png");
    };

    const input = tf.tensor4d(raw, [1, IMG, IMG, 3], "float32");
    const probsT = model.predict(input) as tf.Tensor2D;
    const probs = Array.from(await probsT.data());
    probsT.dispose();

    const order = probs.map((v, i) => [v, i] as [number, number]).sort((a, b) => b[0] - a[0]);
    const top3 = order.slice(0, 3).map(([v, i]) => ({ classId: i, name: CLASS_NAMES[i], confidence: v }));
    const best = top3[0];

    const gradCam = await generateGradCam(model, input, best.classId, c32);
    const featureMaps = await extractFeatureMaps(model, input, 12);
    input.dispose();

    const sampleUrl = up(c32);
    const result: Prediction = {
      classId: best.classId,
      name: best.name,
      confidence: best.confidence,
      top3,
      stages: {
        original: sampleUrl,
        resized: sampleUrl,
        normalized: sampleUrl,
        mean: sum / raw.length,
      },
      modelSource: "Browser CNN (interactive demo)",
      gradCam,
      featureMaps,
      metadata: SIGN_METADATA[best.classId],
    };
    setLastPrediction(result);
    return result;
  }, [phase]);

  const predict = useCallback(async (file: File): Promise<Prediction> => {
    const model = modelRef.current;
    if (!model || phase !== "ready") throw new Error("The CNN is still training — please wait until it finishes.");

    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    if (!["jpg", "jpeg", "png"].includes(ext) && !/^image\/(jpeg|png)$/.test(file.type)) {
      throw new Error("Unsupported file type. Please upload a JPG, JPEG or PNG image.");
    }

    const url = URL.createObjectURL(file);
    let img: HTMLImageElement;
    try {
      img = await loadImage(url);
    } catch {
      URL.revokeObjectURL(url);
      throw new Error("Could not decode this file as an image. Please upload a valid JPG/JPEG/PNG image.");
    }

    // ---- preprocessing, identical to the Python pipeline ----
    const c32 = document.createElement("canvas");
    c32.width = c32.height = IMG;
    const cctx = c32.getContext("2d", { willReadFrequently: true })!;
    cctx.drawImage(img, 0, 0, IMG, IMG); // resize to 32x32
    const data = cctx.getImageData(0, 0, IMG, IMG).data;
    const arr = new Float32Array(IMG * IMG * 3);
    let sum = 0;
    for (let p = 0; p < IMG * IMG; p++) {
      for (let ch = 0; ch < 3; ch++) {
        const v = data[p * 4 + ch] / 255; // normalize to [0, 1]
        arr[p * 3 + ch] = v;
        sum += v;
      }
    }
    const mean = sum / arr.length;

    // preview stages: original, 32x32 (upscaled, pixelated), normalized round-trip
    const up = (src: HTMLCanvasElement) => {
      const c = document.createElement("canvas");
      c.width = c.height = 160;
      const x = c.getContext("2d")!;
      x.imageSmoothingEnabled = false;
      x.drawImage(src, 0, 0, 160, 160);
      return c.toDataURL("image/png");
    };
    const normCanvas = document.createElement("canvas");
    normCanvas.width = normCanvas.height = IMG;
    const nctx = normCanvas.getContext("2d")!;
    const idata = nctx.createImageData(IMG, IMG);
    for (let p = 0; p < IMG * IMG; p++) {
      for (let ch = 0; ch < 3; ch++) idata.data[p * 4 + ch] = Math.round(arr[p * 3 + ch] * 255);
      idata.data[p * 4 + 3] = 255;
    }
    nctx.putImageData(idata, 0, 0);

    // ---- Optional: Check for connected Python FastAPI backend (Render / Local) ----
    const apiUrl = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "");
    if (apiUrl) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`${apiUrl}/api/predict`, {
          method: "POST",
          body: formData,
        });
        if (res.ok) {
          const apiData = (await res.json()) as {
            classId: number;
            name: string;
            confidence: number;
            top3: Array<{ classId: number; name: string; confidence: number }>;
            modelSource?: string;
          };
          const resObj: Prediction = {
            classId: apiData.classId,
            name: apiData.name,
            confidence: apiData.confidence,
            top3: apiData.top3,
            stages: {
              original: url,
              resized: up(c32),
              normalized: up(normCanvas),
              mean,
            },
            modelSource: apiData.modelSource ?? "Python TensorFlow (GTSRB trained)",
            metadata: SIGN_METADATA[apiData.classId],
          };
          setLastPrediction(resObj);
          return resObj;
        }
      } catch (err) {
        console.warn("Backend API request failed, using in-browser CNN:", err);
      }
    }

    // ---- run the CNN ----
    const input = tf.tensor4d(arr, [1, IMG, IMG, 3], "float32");
    const probsT = model.predict(input) as tf.Tensor2D;
    const probs = Array.from(await probsT.data());
    probsT.dispose();
    // NOTE: `url` is intentionally kept alive — it is returned as stages.original.

    const order = probs.map((v, i) => [v, i] as [number, number]).sort((a, b) => b[0] - a[0]);
    const top3 = order.slice(0, 3).map(([v, i]) => ({ classId: i, name: CLASS_NAMES[i], confidence: v }));
    const best = top3[0];

    // Compute genuine Grad-CAM and intermediate feature maps
    const gradCam = await generateGradCam(model, input, best.classId, c32);
    const featureMaps = await extractFeatureMaps(model, input, 12);
    input.dispose();

    const result: Prediction = {
      classId: best.classId,
      name: best.name,
      confidence: best.confidence,
      top3,
      stages: {
        original: url,
        resized: up(c32),
        normalized: up(normCanvas),
        mean,
      },
      modelSource: "Browser CNN (interactive demo)",
      gradCam,
      featureMaps,
      metadata: SIGN_METADATA[best.classId],
    };
    setLastPrediction(result);
    return result;
  }, [phase]);

  return (
    <ModelContext.Provider
      value={{
        phase,
        epoch,
        totalEpochs: TOTAL_EPOCHS,
        trainPerClass: TRAIN_PER_CLASS,
        valPerClass: VAL_PER_CLASS,
        history,
        evalData,
        error,
        ready: phase === "ready",
        lastPrediction,
        predict,
        explainSampleSign,
        retrain,
      }}
    >
      {children}
    </ModelContext.Provider>
  );
}
