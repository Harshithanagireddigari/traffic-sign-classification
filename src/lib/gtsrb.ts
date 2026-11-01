/**
 * gtsrb.ts
 * --------
 * The 43 class names of the German Traffic Sign Recognition Benchmark (GTSRB)
 * plus a procedural canvas renderer that draws a stylised version of every
 * sign class. The renderer is used to synthesise training data in the browser
 * so the demo can train a REAL CNN (TensorFlow.js) live — no dummy numbers.
 */

export const CLASS_NAMES: string[] = [
  "Speed limit (20 km/h)",
  "Speed limit (30 km/h)",
  "Speed limit (50 km/h)",
  "Speed limit (60 km/h)",
  "Speed limit (70 km/h)",
  "Speed limit (80 km/h)",
  "End of speed limit (80 km/h)",
  "Speed limit (100 km/h)",
  "Speed limit (120 km/h)",
  "No passing",
  "No passing for vehicles over 3.5 tonnes",
  "Right-of-way at the next intersection",
  "Priority road",
  "Yield",
  "Stop",
  "No vehicles",
  "Vehicles over 3.5 tonnes prohibited",
  "No entry",
  "General caution",
  "Dangerous curve to the left",
  "Dangerous curve to the right",
  "Double curve",
  "Bumpy road",
  "Slippery road",
  "Road narrows to the right",
  "Road work",
  "Traffic signals",
  "Pedestrians",
  "Children crossing",
  "Bicycles crossing",
  "Beware of ice/snow",
  "Wild animals crossing",
  "End of all speed and passing limits",
  "Turn right ahead",
  "Turn left ahead",
  "Ahead only",
  "Go straight or right",
  "Go straight or left",
  "Keep right",
  "Keep left",
  "Roundabout mandatory",
  "End of no passing",
  "End of no passing for vehicles over 3.5 tonnes",
];

const RED = "#d0342c";
const BLUE = "#1f5fbf";
const YELLOW = "#f6c700";
const INK = "#181818";

const rand = (a: number, b: number) => a + Math.random() * (b - a);
function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ------------------------- low-level drawing helpers ------------------------- */

function disc(ctx: CanvasRenderingContext2D, s: number, fill: string, border: string, bw: number) {
  ctx.beginPath();
  ctx.arc(s / 2, s / 2, s * 0.47, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (bw > 0) {
    ctx.lineWidth = bw;
    ctx.strokeStyle = border;
    ctx.stroke();
  }
}

function clipCircle(ctx: CanvasRenderingContext2D, s: number, rFrac = 0.41) {
  ctx.beginPath();
  ctx.arc(s / 2, s / 2, rFrac * s, 0, Math.PI * 2);
  ctx.clip();
}

function warningTriangle(ctx: CanvasRenderingContext2D, s: number) {
  ctx.beginPath();
  ctx.moveTo(s / 2, s * 0.05);
  ctx.lineTo(s * 0.95, s * 0.91);
  ctx.lineTo(s * 0.05, s * 0.91);
  ctx.closePath();
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.lineJoin = "round";
  ctx.lineWidth = s * 0.085;
  ctx.strokeStyle = RED;
  ctx.stroke();
}

function yieldTriangle(ctx: CanvasRenderingContext2D, s: number) {
  ctx.beginPath();
  ctx.moveTo(s * 0.06, s * 0.12);
  ctx.lineTo(s * 0.94, s * 0.12);
  ctx.lineTo(s / 2, s * 0.93);
  ctx.closePath();
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.lineJoin = "round";
  ctx.lineWidth = s * 0.075;
  ctx.strokeStyle = RED;
  ctx.stroke();
}

function octagon(ctx: CanvasRenderingContext2D, s: number, fill: string, border: string, bw: number) {
  const r = s * 0.47;
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const a = Math.PI / 8 + (i * Math.PI) / 4;
    const x = s / 2 + r * Math.cos(a);
    const y = s / 2 + r * Math.sin(a);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = bw;
  ctx.strokeStyle = border;
  ctx.stroke();
}

function diamond(ctx: CanvasRenderingContext2D, s: number, fill: string, border: string, bw: number) {
  ctx.beginPath();
  ctx.moveTo(s / 2, s * 0.07);
  ctx.lineTo(s * 0.93, s / 2);
  ctx.lineTo(s / 2, s * 0.93);
  ctx.lineTo(s * 0.07, s / 2);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = bw;
  ctx.strokeStyle = border;
  ctx.stroke();
}

function squareSign(ctx: CanvasRenderingContext2D, s: number, fill: string, border: string, bw: number) {
  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, s, s);
  ctx.lineWidth = bw;
  ctx.strokeStyle = border;
  ctx.strokeRect(bw / 2, bw / 2, s - bw, s - bw);
}

function txt(ctx: CanvasRenderingContext2D, s: number, str: string, frac: number, color: string, y = 0.53) {
  ctx.fillStyle = color;
  ctx.font = `800 ${s * frac}px Arial, Helvetica, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(str, s / 2, s * y);
}

/** Stroke a polyline. Points are unit coordinates (0..1) scaled by s. */
function poly(
  ctx: CanvasRenderingContext2D,
  s: number,
  pts: Array<[number, number]>,
  wFrac: number,
  color: string,
  close = false,
) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = wFrac * s;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(pts[0][0] * s, pts[0][1] * s);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * s, pts[i][1] * s);
  if (close) {
    ctx.closePath();
    ctx.fill();
  } else {
    ctx.stroke();
  }
}

/** Polyline with an arrow head at the last point. */
function arrow(ctx: CanvasRenderingContext2D, s: number, pts: Array<[number, number]>, wFrac: number, color: string) {
  poly(ctx, s, pts, wFrac, color);
  const [ax, ay] = pts[pts.length - 1];
  const [bx, by] = pts[pts.length - 2];
  const ang = Math.atan2(ay - by, ax - bx);
  const x2 = ax * s;
  const y2 = ay * s;
  const h = wFrac * s * 2.4;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - h * Math.cos(ang - 0.42), y2 - h * Math.sin(ang - 0.42));
  ctx.lineTo(x2 - h * Math.cos(ang + 0.42), y2 - h * Math.sin(ang + 0.42));
  ctx.closePath();
  ctx.fill();
}

function dot(ctx: CanvasRenderingContext2D, s: number, x: number, y: number, r: number, color: string) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x * s, y * s, r * s, 0, Math.PI * 2);
  ctx.fill();
}

function rr(ctx: CanvasRenderingContext2D, s: number, x: number, y: number, w: number, h: number, r: number, fill: string) {
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.roundRect(x * s, y * s, w * s, h * s, r * s);
  ctx.fill();
}

/** Stylised car pictogram centred at (cx, cy) with width wFrac. */
function car(ctx: CanvasRenderingContext2D, s: number, cx: number, cy: number, wFrac: number, color: string) {
  const w = wFrac;
  const h = w * 0.52;
  const x = cx - w / 2;
  const y = cy - h / 2;
  rr(ctx, s, x + w * 0.16, y, w * 0.68, h * 0.42, w * 0.1, color); // cabin
  rr(ctx, s, x, y + h * 0.3, w, h * 0.7, w * 0.14, color); // body
  dot(ctx, s, x + w * 0.22, y + h, 0.045, "#222");
  dot(ctx, s, x + w * 0.78, y + h, 0.045, "#222");
}

/** Stylised truck pictogram centred at (cx, cy). */
function truck(ctx: CanvasRenderingContext2D, s: number, cx: number, cy: number, wFrac: number, color: string) {
  const w = wFrac;
  const h = w * 0.5;
  const x = cx - w / 2;
  const y = cy - h / 2;
  rr(ctx, s, x, y, w * 0.62, h * 0.85, 0.02, color); // trailer
  rr(ctx, s, x + w * 0.64, y + h * 0.35, w * 0.3, h * 0.65, 0.04, color); // cab
  dot(ctx, s, x + w * 0.15, y + h, 0.04, "#222");
  dot(ctx, s, x + w * 0.5, y + h, 0.04, "#222");
  dot(ctx, s, x + w * 0.8, y + h, 0.04, "#222");
}

function person(ctx: CanvasRenderingContext2D, s: number, cx: number, cy: number, k: number, color: string) {
  dot(ctx, s, cx, cy - 0.19 * k, 0.055 * k, color);
  poly(ctx, s, [[cx, cy - 0.12 * k], [cx, cy + 0.1 * k]], 0.06 * k, color);
  poly(ctx, s, [[cx - 0.12 * k, cy - 0.02 * k], [cx + 0.12 * k, cy - 0.02 * k]], 0.045 * k, color);
  poly(ctx, s, [[cx, cy + 0.1 * k], [cx - 0.09 * k, cy + 0.26 * k]], 0.05 * k, color);
  poly(ctx, s, [[cx, cy + 0.1 * k], [cx + 0.1 * k, cy + 0.24 * k]], 0.05 * k, color);
}

/** Diagonal "cancelled" stripe used by the "end of ..." signs. */
function stripe(ctx: CanvasRenderingContext2D, s: number, color = "#b5b5b5") {
  ctx.save();
  clipCircle(ctx, s, 0.42);
  ctx.strokeStyle = color;
  ctx.lineWidth = s * 0.1;
  ctx.beginPath();
  ctx.moveTo(s * 0.12, s * 0.88);
  ctx.lineTo(s * 0.88, s * 0.12);
  ctx.stroke();
  ctx.restore();
}

/* ------------------------------ the 43 sign classes ------------------------------ */

export function drawSign(ctx: CanvasRenderingContext2D, classId: number, s: number): void {
  ctx.save();
  switch (classId) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 7:
    case 8: {
      const labels: Record<number, string> = { 0: "20", 1: "30", 2: "50", 3: "60", 4: "70", 5: "80", 7: "100", 8: "120" };
      const label = labels[classId];
      disc(ctx, s, "#ffffff", RED, s * 0.1);
      txt(ctx, s, label, label.length > 2 ? 0.3 : 0.42, INK);
      break;
    }
    case 6: // end of speed limit 80
      disc(ctx, s, "#f2f2f2", RED, s * 0.1);
      txt(ctx, s, "80", 0.42, "#8f8f8f");
      stripe(ctx, s);
      break;
    case 9:
    case 10: {
      disc(ctx, s, "#ffffff", RED, s * 0.1);
      ctx.save();
      clipCircle(ctx, s);
      if (classId === 9) {
        car(ctx, s, 0.33, 0.52, 0.3, "#c02020");
        car(ctx, s, 0.67, 0.52, 0.3, INK);
      } else {
        car(ctx, s, 0.31, 0.52, 0.26, INK);
        truck(ctx, s, 0.68, 0.52, 0.3, "#c02020");
      }
      ctx.restore();
      break;
    }
    case 41:
    case 42: {
      disc(ctx, s, "#f2f2f2", "#a8a8a8", s * 0.1);
      ctx.save();
      clipCircle(ctx, s);
      if (classId === 41) {
        car(ctx, s, 0.33, 0.52, 0.3, "#9a9a9a");
        car(ctx, s, 0.67, 0.52, 0.3, "#9a9a9a");
      } else {
        car(ctx, s, 0.31, 0.52, 0.26, "#9a9a9a");
        truck(ctx, s, 0.68, 0.52, 0.3, "#9a9a9a");
      }
      ctx.restore();
      stripe(ctx, s, "#bdbdbd");
      break;
    }
    case 11: // right of way at next intersection
      squareSign(ctx, s, "#ffffff", INK, s * 0.03);
      poly(ctx, s, [[0.44, 0.78], [0.44, 0.3]], 0.1, INK);
      poly(ctx, s, [[0.44, 0.55], [0.72, 0.55]], 0.045, INK);
      break;
    case 12: // priority road
      diamond(ctx, s, YELLOW, "#ffffff", s * 0.05);
      break;
    case 13: // yield
      yieldTriangle(ctx, s);
      break;
    case 14: // stop
      octagon(ctx, s, RED, "#ffffff", s * 0.035);
      txt(ctx, s, "STOP", 0.2, "#ffffff");
      break;
    case 15: // no vehicles
      disc(ctx, s, "#ffffff", RED, s * 0.1);
      ctx.save();
      clipCircle(ctx, s);
      car(ctx, s, 0.5, 0.52, 0.42, INK);
      ctx.restore();
      break;
    case 16: // no trucks
      disc(ctx, s, "#ffffff", RED, s * 0.1);
      ctx.save();
      clipCircle(ctx, s);
      truck(ctx, s, 0.5, 0.52, 0.5, INK);
      ctx.restore();
      break;
    case 17: // no entry
      disc(ctx, s, RED, RED, 0);
      rr(ctx, s, 0.26, 0.44, 0.48, 0.12, 0.02, "#ffffff");
      break;
    case 18: // general caution
      warningTriangle(ctx, s);
      poly(ctx, s, [[0.5, 0.36], [0.5, 0.62]], 0.07, INK);
      dot(ctx, s, 0.5, 0.75, 0.045, INK);
      break;
    case 19: // dangerous curve left
      warningTriangle(ctx, s);
      arrow(ctx, s, [[0.66, 0.76], [0.5, 0.6], [0.42, 0.5], [0.4, 0.36]], 0.07, INK);
      break;
    case 20: // dangerous curve right
      warningTriangle(ctx, s);
      arrow(ctx, s, [[0.34, 0.76], [0.5, 0.6], [0.58, 0.5], [0.6, 0.36]], 0.07, INK);
      break;
    case 21: // double curve
      warningTriangle(ctx, s);
      arrow(ctx, s, [[0.36, 0.76], [0.56, 0.64], [0.44, 0.52], [0.62, 0.38]], 0.065, INK);
      break;
    case 22: // bumpy road
      warningTriangle(ctx, s);
      poly(ctx, s, [[0.28, 0.62], [0.4, 0.62], [0.48, 0.48], [0.58, 0.66], [0.72, 0.6]], 0.06, INK);
      break;
    case 23: // slippery road
      warningTriangle(ctx, s);
      car(ctx, s, 0.5, 0.44, 0.34, INK);
      poly(ctx, s, [[0.3, 0.74], [0.4, 0.69], [0.5, 0.74], [0.6, 0.69], [0.7, 0.74]], 0.05, INK);
      break;
    case 24: // road narrows right
      warningTriangle(ctx, s);
      poly(ctx, s, [[0.3, 0.34], [0.52, 0.34], [0.52, 0.46], [0.74, 0.52], [0.52, 0.58], [0.52, 0.7], [0.3, 0.7]], 0.01, INK, true);
      break;
    case 25: // road work
      warningTriangle(ctx, s);
      person(ctx, s, 0.5, 0.56, 1, INK);
      break;
    case 26: // traffic signals
      warningTriangle(ctx, s);
      rr(ctx, s, 0.42, 0.3, 0.16, 0.42, 0.05, "#222222");
      dot(ctx, s, 0.5, 0.38, 0.038, RED);
      dot(ctx, s, 0.5, 0.51, 0.038, YELLOW);
      dot(ctx, s, 0.5, 0.64, 0.038, "#2ecc71");
      break;
    case 27: // pedestrians
      warningTriangle(ctx, s);
      person(ctx, s, 0.5, 0.55, 1.05, INK);
      break;
    case 28: // children crossing
      warningTriangle(ctx, s);
      person(ctx, s, 0.38, 0.58, 0.8, INK);
      person(ctx, s, 0.63, 0.55, 0.9, INK);
      break;
    case 29: { // bicycles crossing
      warningTriangle(ctx, s);
      ctx.strokeStyle = INK;
      ctx.lineWidth = s * 0.045;
      for (const wx of [0.36, 0.64]) {
        ctx.beginPath();
        ctx.arc(wx * s, 0.62 * s, 0.1 * s, 0, Math.PI * 2);
        ctx.stroke();
      }
      poly(ctx, s, [[0.36, 0.62], [0.48, 0.44], [0.64, 0.62], [0.5, 0.62], [0.48, 0.44]], 0.04, INK);
      poly(ctx, s, [[0.48, 0.44], [0.57, 0.37]], 0.04, INK);
      break;
    }
    case 30: { // ice / snow
      warningTriangle(ctx, s);
      for (let k = 0; k < 6; k++) {
        const a = (k * Math.PI) / 3;
        poly(ctx, s, [[0.5, 0.55], [0.5 + 0.17 * Math.cos(a), 0.55 + 0.17 * Math.sin(a)]], 0.045, INK);
      }
      dot(ctx, s, 0.5, 0.55, 0.035, INK);
      break;
    }
    case 31: { // wild animals
      warningTriangle(ctx, s);
      poly(ctx, s, [[0.3, 0.6], [0.42, 0.5], [0.56, 0.52], [0.62, 0.42], [0.68, 0.36], [0.72, 0.4], [0.66, 0.5], [0.72, 0.56], [0.6, 0.58], [0.5, 0.66], [0.44, 0.64]], 0.01, INK, true);
      poly(ctx, s, [[0.44, 0.64], [0.4, 0.78]], 0.035, INK);
      poly(ctx, s, [[0.54, 0.65], [0.56, 0.78]], 0.035, INK);
      poly(ctx, s, [[0.68, 0.36], [0.66, 0.28]], 0.03, INK);
      poly(ctx, s, [[0.71, 0.37], [0.74, 0.29]], 0.03, INK);
      break;
    }
    case 32: { // end of all limits
      disc(ctx, s, "#ffffff", RED, s * 0.1);
      ctx.save();
      clipCircle(ctx, s, 0.42);
      for (const off of [-0.24, 0, 0.24]) {
        ctx.strokeStyle = "#7d7d7d";
        ctx.lineWidth = s * 0.06;
        ctx.beginPath();
        ctx.moveTo(s * (0.1), s * (0.9 + off));
        ctx.lineTo(s * (0.9), s * (0.1 + off));
        ctx.stroke();
      }
      ctx.restore();
      break;
    }
    case 33: // turn right ahead
    case 34: // turn left ahead
    case 35: // ahead only
    case 36: // straight or right
    case 37: // straight or left
    case 38: // keep right
    case 39: // keep left
    case 40: { // roundabout
      disc(ctx, s, BLUE, BLUE, 0);
      ctx.save();
      clipCircle(ctx, s, 0.47);
      if (classId === 33) arrow(ctx, s, [[0.34, 0.76], [0.34, 0.46], [0.64, 0.46]], 0.09, "#ffffff");
      else if (classId === 34) arrow(ctx, s, [[0.66, 0.76], [0.66, 0.46], [0.36, 0.46]], 0.09, "#ffffff");
      else if (classId === 35) arrow(ctx, s, [[0.5, 0.78], [0.5, 0.3]], 0.09, "#ffffff");
      else if (classId === 36) {
        arrow(ctx, s, [[0.36, 0.78], [0.36, 0.32]], 0.075, "#ffffff");
        arrow(ctx, s, [[0.68, 0.78], [0.68, 0.56], [0.86, 0.56]], 0.075, "#ffffff");
      } else if (classId === 37) {
        arrow(ctx, s, [[0.64, 0.78], [0.64, 0.32]], 0.075, "#ffffff");
        arrow(ctx, s, [[0.32, 0.78], [0.32, 0.56], [0.14, 0.56]], 0.075, "#ffffff");
      } else if (classId === 38) arrow(ctx, s, [[0.38, 0.78], [0.44, 0.52], [0.68, 0.42]], 0.09, "#ffffff");
      else if (classId === 39) arrow(ctx, s, [[0.62, 0.78], [0.56, 0.52], [0.32, 0.42]], 0.09, "#ffffff");
      else {
        // roundabout: circular arrow
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = s * 0.055;
        ctx.beginPath();
        ctx.arc(s / 2, s / 2, s * 0.2, 0, Math.PI * 2);
        ctx.stroke();
        for (const aDeg of [-90, 30, 150]) {
          const a = (aDeg * Math.PI) / 180;
          const px = s / 2 + s * 0.2 * Math.cos(a);
          const py = s / 2 + s * 0.2 * Math.sin(a);
          const ta = a + Math.PI / 2; // clockwise tangent
          ctx.fillStyle = "#ffffff";
          ctx.beginPath();
          ctx.moveTo(px + 0.05 * s * Math.cos(ta), py + 0.05 * s * Math.sin(ta));
          ctx.lineTo(px - 0.035 * s * Math.cos(ta - 0.5), py - 0.035 * s * Math.sin(ta - 0.5));
          ctx.lineTo(px - 0.035 * s * Math.cos(ta + 0.5), py - 0.035 * s * Math.sin(ta + 0.5));
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.restore();
      break;
    }
    default:
      disc(ctx, s, "#ffffff", RED, s * 0.1);
  }
  ctx.restore();
}

/* --------------------------- synthetic data generation --------------------------- */

export const IMG = 32; // model input size, same as the Python project
const RENDER = 64; // signs are drawn at 64x64 then downsampled to 32x32

/**
 * Render one augmented sample of a sign class.
 * Returns a normalized Float32Array of length 32*32*3 with values in [0, 1] —
 * exactly the preprocessing used by the Python project (resize 32x32, /255).
 */
export function renderSample(classId: number): Float32Array {
  // 1. draw the sign on a transparent canvas
  const sign = document.createElement("canvas");
  sign.width = sign.height = RENDER;
  const sctx = sign.getContext("2d")!;
  drawSign(sctx, classId, RENDER);

  // 2. composite it onto a random background with a random affine transform
  const cv = document.createElement("canvas");
  cv.width = cv.height = RENDER;
  const ctx = cv.getContext("2d", { willReadFrequently: true })!;
  const bg = `hsl(${rand(0, 360)}, ${rand(12, 40)}%, ${rand(80, 94)}%)`;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, RENDER, RENDER);
  ctx.save();
  ctx.translate(RENDER / 2 + rand(-4, 4), RENDER / 2 + rand(-4, 4));
  ctx.rotate(rand(-0.28, 0.28));
  const sc = rand(0.78, 1.04);
  ctx.scale(sc, sc);
  ctx.drawImage(sign, -RENDER / 2, -RENDER / 2);
  ctx.restore();

  // 3. downsample to 32x32 (this is the "resize" preprocessing step)
  const out = document.createElement("canvas");
  out.width = out.height = IMG;
  const octx = out.getContext("2d", { willReadFrequently: true })!;
  octx.imageSmoothingEnabled = true;
  octx.imageSmoothingQuality = "high";
  octx.drawImage(cv, 0, 0, IMG, IMG);
  const d = octx.getImageData(0, 0, IMG, IMG).data;

  // 4. brightness / contrast jitter + gaussian noise, then normalize to [0, 1]
  const arr = new Float32Array(IMG * IMG * 3);
  const bright = rand(0.8, 1.2);
  const contrast = rand(0.85, 1.15);
  for (let p = 0; p < IMG * IMG; p++) {
    for (let ch = 0; ch < 3; ch++) {
      let v = d[p * 4 + ch];
      v = (v - 128) * contrast + 128;
      v = v * bright + randn() * 8;
      arr[p * 3 + ch] = Math.min(255, Math.max(0, v)) / 255;
    }
  }
  return arr;
}

export interface SyntheticDataset {
  xs: Float32Array; // [total, 32, 32, 3]
  ys: Int32Array; // [total]
  trainCount: number;
  total: number;
}

/** Interleaved class order so a prefix slice is already class-balanced. */
export function generateDataset(trainPerClass: number, valPerClass: number): SyntheticDataset {
  const perClass = trainPerClass + valPerClass;
  const total = CLASS_NAMES.length * perClass;
  const xs = new Float32Array(total * IMG * IMG * 3);
  const ys = new Int32Array(total);
  let n = 0;
  for (let k = 0; k < perClass; k++) {
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      xs.set(renderSample(c), n * IMG * IMG * 3);
      ys[n] = c;
      n++;
    }
  }
  return { xs, ys, trainCount: CLASS_NAMES.length * trainPerClass, total };
}

/** Small PNG thumbnail of the canonical sign for a class (used in the UI). */
export function signThumbnailDataURL(classId: number, px = 96): string {
  const cv = document.createElement("canvas");
  cv.width = cv.height = px;
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = "#f1f5f9";
  ctx.fillRect(0, 0, px, px);
  drawSign(ctx, classId, px * 0.94);
  // re-centre: drawSign draws centered in the given size
  return cv.toDataURL("image/png");
}
