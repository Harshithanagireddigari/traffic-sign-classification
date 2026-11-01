/**
 * cnnExplainability.ts
 * --------------------
 * Genuine explainability utilities for the TensorFlow.js CNN:
 * 1. Intermediate Feature Maps extraction for conv1, conv2, and conv3.
 * 2. True mathematical Grad-CAM (Gradient-weighted Class Activation Mapping).
 * 3. Actual sample training image rendering from the procedural dataset generator.
 */

import * as tf from "@tensorflow/tfjs";
import { renderSample, IMG } from "./gtsrb";

export interface LayerFeatureMaps {
  layerName: string;
  stageName: string;
  shape: [number, number, number]; // [H, W, Channels]
  patternDescription: string;
  channelDataUrls: string[];
}

export interface GradCamResult {
  heatmapUrl: string;
  overlayUrl: string;
  camValues: number[][]; // 8x8 normalized CAM grid
}

/**
 * Maps a single scalar in [0, 1] to an RGB color using the Jet/Turbo colormap.
 */
function turboColormap(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v));
  // Jet-style palette: blue -> cyan -> green -> yellow -> red
  let r = 0, g = 0, b = 0;
  if (x < 0.125) {
    b = 0.5 + 4 * x;
  } else if (x < 0.375) {
    b = 1;
    g = 4 * (x - 0.125);
  } else if (x < 0.625) {
    b = 1 - 4 * (x - 0.375);
    g = 1;
    r = 4 * (x - 0.375);
  } else if (x < 0.875) {
    g = 1 - 4 * (x - 0.625);
    r = 1;
  } else {
    r = 1 - 2 * (x - 0.875);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Extracts intermediate feature map activations from early (conv1), middle (conv2),
 * and deeper (conv3) convolutional layers using real tensor evaluations.
 */
export async function extractFeatureMaps(
  model: tf.LayersModel,
  inputTensor: tf.Tensor4D,
  channelsToSample = 12,
): Promise<LayerFeatureMaps[]> {
  try {
    const conv1 = model.getLayer("conv1");
    const conv2 = model.getLayer("conv2");
    const conv3 = model.getLayer("conv3");

    // Create multi-output submodel
    const fmapModel = tf.model({
      inputs: model.inputs,
      outputs: [conv1.output, conv2.output, conv3.output],
    });

    const [out1, out2, out3] = fmapModel.predict(inputTensor) as tf.Tensor4D[];

    const specs = [
      {
        tensor: out1,
        name: "conv1",
        stage: "Early Convolutional Layer (Conv2D · 32 filters)",
        pattern: "Detects low-level visual primitives: sharp edge orientations, contrast gradients, outer sign border arcs, and color transitions.",
      },
      {
        tensor: out2,
        name: "conv2",
        stage: "Middle Convolutional Layer (Conv2D · 64 filters)",
        pattern: "Assembles low-level edges into geometric sign components: circular perimeters, equilateral warning triangles, corner junctures, and contour rings.",
      },
      {
        tensor: out3,
        name: "conv3",
        stage: "Deep Convolutional Layer (Conv2D · 128 filters)",
        pattern: "Extracts high-level semantic motifs: numerical glyphs (e.g. '50', '80'), directional arrows, pedestrians, exclamation points, and restriction bars.",
      },
    ];

    const results: LayerFeatureMaps[] = [];

    for (const spec of specs) {
      const shape = spec.tensor.shape; // [1, H, W, C]
      const h = shape[1];
      const w = shape[2];
      const c = shape[3];
      const sampleCount = Math.min(channelsToSample, c);

      // Extract raw data
      const tensorData = await spec.tensor.data();
      const channelUrls: string[] = [];

      for (let ch = 0; ch < sampleCount; ch++) {
        // Find min and max for this channel to normalize [0, 1]
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const val = tensorData[y * w * c + x * c + ch];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
          }
        }
        const range = maxVal - minVal > 1e-6 ? maxVal - minVal : 1;

        // Render to canvas
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d")!;
        const imgData = ctx.createImageData(w, h);

        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const val = tensorData[y * w * c + x * c + ch];
            const norm = (val - minVal) / range;
            const [cr, cg, cb] = turboColormap(norm);
            const idx = (y * w + x) * 4;
            imgData.data[idx] = cr;
            imgData.data[idx + 1] = cg;
            imgData.data[idx + 2] = cb;
            imgData.data[idx + 3] = 255;
          }
        }
        ctx.putImageData(imgData, 0, 0);

        // Upscale nicely to 80x80 for crisp display
        const upCanvas = document.createElement("canvas");
        upCanvas.width = 80;
        upCanvas.height = 80;
        const upCtx = upCanvas.getContext("2d")!;
        upCtx.imageSmoothingEnabled = false;
        upCtx.drawImage(canvas, 0, 0, 80, 80);
        channelUrls.push(upCanvas.toDataURL("image/png"));
      }

      spec.tensor.dispose();

      results.push({
        layerName: spec.name,
        stageName: spec.stage,
        shape: [h, w, c],
        patternDescription: spec.pattern,
        channelDataUrls: channelUrls,
      });
    }

    return results;
  } catch (err) {
    console.error("Failed to extract feature maps:", err);
    return [];
  }
}

/**
 * Computes Grad-CAM (Gradient-weighted Class Activation Mapping) for the predicted class.
 * Calculates gradients of the target class score with respect to the last convolutional layer (conv3),
 * weights feature maps by global average pooled gradients, and produces a spatial heatmap.
 */
export async function generateGradCam(
  model: tf.LayersModel,
  inputTensor: tf.Tensor4D,
  targetClass: number,
  originalCanvas: HTMLCanvasElement,
): Promise<GradCamResult | null> {
  try {
    const conv3Layer = model.getLayer("conv3");
    const convModel = tf.model({
      inputs: model.inputs,
      outputs: conv3Layer.output,
    });

    const conv3Output = convModel.predict(inputTensor) as tf.Tensor4D; // [1, 8, 8, 128]

    // Construct classifier head taking conv3 output shape
    const classifierInput = tf.input({ shape: [8, 8, 128] });
    let x: tf.SymbolicTensor = classifierInput;
    x = model.getLayer("pool3").apply(x) as tf.SymbolicTensor;
    x = model.getLayer("flatten").apply(x) as tf.SymbolicTensor;
    x = model.getLayer("dense1").apply(x) as tf.SymbolicTensor;
    x = model.getLayer("output").apply(x) as tf.SymbolicTensor;
    const classifierModel = tf.model({ inputs: classifierInput, outputs: x });

    const gradFn = tf.grad((features: tf.Tensor4D) => {
      const preds = classifierModel.predict(features) as tf.Tensor2D;
      return preds.slice([0, targetClass], [1, 1]).asScalar();
    });

    const grads = gradFn(conv3Output); // [1, 8, 8, 128]
    const pooledGrads = grads.mean([0, 1, 2]); // [128]

    // Weight conv3 features by pooled gradients
    const convFeatures = conv3Output.squeeze([0]) as tf.Tensor3D; // [8, 8, 128]
    const weighted = convFeatures.mul(pooledGrads); // [8, 8, 128]
    const cam = weighted.sum(-1); // [8, 8]
    const reluCam = tf.relu(cam);
    const maxVal = reluCam.max();
    const normalizedCam = reluCam.div(maxVal.add(1e-10)) as tf.Tensor2D;

    const camData = await normalizedCam.data();
    const camValues: number[][] = [];
    for (let r = 0; r < 8; r++) {
      const row: number[] = [];
      for (let c = 0; c < 8; c++) {
        row.push(camData[r * 8 + c]);
      }
      camValues.push(row);
    }

    // Clean up tensors
    conv3Output.dispose();
    grads.dispose();
    pooledGrads.dispose();
    convFeatures.dispose();
    weighted.dispose();
    cam.dispose();
    reluCam.dispose();
    maxVal.dispose();
    normalizedCam.dispose();

    // 1. Draw raw heatmap into an 8x8 canvas then smoothly upscale to 160x160
    const heatCanvas8 = document.createElement("canvas");
    heatCanvas8.width = 8;
    heatCanvas8.height = 8;
    const hctx8 = heatCanvas8.getContext("2d")!;
    const hdata8 = hctx8.createImageData(8, 8);

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const val = camValues[r][c];
        const [red, green, blue] = turboColormap(val);
        const idx = (r * 8 + c) * 4;
        hdata8.data[idx] = red;
        hdata8.data[idx + 1] = green;
        hdata8.data[idx + 2] = blue;
        hdata8.data[idx + 3] = Math.round(val * 240 + 15);
      }
    }
    hctx8.putImageData(hdata8, 0, 0);

    const upHeatmapCanvas = document.createElement("canvas");
    upHeatmapCanvas.width = 160;
    upHeatmapCanvas.height = 160;
    const upHeatCtx = upHeatmapCanvas.getContext("2d")!;
    upHeatCtx.imageSmoothingEnabled = true;
    upHeatCtx.imageSmoothingQuality = "high";
    upHeatCtx.drawImage(heatCanvas8, 0, 0, 160, 160);
    const heatmapUrl = upHeatmapCanvas.toDataURL("image/png");

    // 2. Composite overlay: Original image + 55% heatmap overlay
    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = 160;
    overlayCanvas.height = 160;
    const octx = overlayCanvas.getContext("2d")!;
    octx.imageSmoothingEnabled = true;
    octx.drawImage(originalCanvas, 0, 0, 160, 160);
    octx.globalAlpha = 0.6;
    octx.drawImage(upHeatmapCanvas, 0, 0, 160, 160);
    octx.globalAlpha = 1.0;
    const overlayUrl = overlayCanvas.toDataURL("image/png");

    return {
      heatmapUrl,
      overlayUrl,
      camValues,
    };
  } catch (err) {
    console.error("Grad-CAM generation error:", err);
    return null;
  }
}

/**
 * Procedurally synthesizes real sample training images for a specific GTSRB class,
 * rendering them with the identical affine transforms, lighting, and noise as during training.
 */
export function generateSampleImagesForClass(classId: number, count = 6): string[] {
  const urls: string[] = [];
  for (let i = 0; i < count; i++) {
    const raw = renderSample(classId); // Float32Array(32*32*3) in [0, 1]
    const cv = document.createElement("canvas");
    cv.width = IMG;
    cv.height = IMG;
    const ctx = cv.getContext("2d")!;
    const idata = ctx.createImageData(IMG, IMG);
    for (let p = 0; p < IMG * IMG; p++) {
      for (let ch = 0; ch < 3; ch++) {
        idata.data[p * 4 + ch] = Math.round(raw[p * 3 + ch] * 255);
      }
      idata.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(idata, 0, 0);

    // Upscale for smooth visual presentation
    const up = document.createElement("canvas");
    up.width = 96;
    up.height = 96;
    const upCtx = up.getContext("2d")!;
    upCtx.imageSmoothingEnabled = false;
    upCtx.drawImage(cv, 0, 0, 96, 96);
    urls.push(up.toDataURL("image/png"));
  }
  return urls;
}
