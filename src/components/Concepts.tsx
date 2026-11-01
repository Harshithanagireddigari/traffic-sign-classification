import { SectionHeading } from "./ui";

const CONCEPTS = [
  {
    title: "Convolution (Conv2D)",
    desc: "Small learnable filters (3×3) slide over the image producing feature maps: first edges and corners, deeper layers shapes and whole sign parts.",
    icon: "🧱",
  },
  {
    title: "ReLU activation",
    desc: "f(x) = max(0, x). Keeps positive activations and zeroes the rest, introducing the non-linearity that lets the network learn complex patterns.",
    icon: "📈",
  },
  {
    title: "Max-Pooling",
    desc: "2×2 downsampling that keeps only the strongest activation per window — shrinks the image and makes detection robust to small shifts.",
    icon: "🔽",
  },
  {
    title: "Flatten + Dense",
    desc: "The final 4×4×128 feature maps are flattened into a 2048-value vector and passed through fully-connected (Dense) layers.",
    icon: "🔗",
  },
  {
    title: "Softmax output",
    desc: "The last Dense(43) layer with softmax converts logits into 43 probabilities that sum to 100% — one per traffic-sign class.",
    icon: "🎯",
  },
  {
    title: "Dropout (0.5)",
    desc: "During training, half the neurons are randomly disabled each step. This prevents overfitting and improves generalisation.",
    icon: "🎲",
  },
  {
    title: "Preprocessing",
    desc: "Every image is resized to 32×32 and pixel values are normalized from [0, 255] to [0, 1], matching the training distribution exactly.",
    icon: "🖼️",
  },
  {
    title: "GTSRB dataset",
    desc: "German Traffic Sign Recognition Benchmark: ~39,200 training + ~12,800 test images across 43 classes, with per-class CSV label files read via pandas.",
    icon: "🚧",
  },
  {
    title: "Evaluation",
    desc: "Accuracy/loss curves, a confusion matrix and a precision/recall/F1 classification report computed on a held-out test set — never hardcoded.",
    icon: "📊",
  },
  {
    title: "Adam optimizer",
    desc: "Adaptive learning-rate optimizer used with sparse categorical cross-entropy loss (integer labels 0…42) and early stopping on validation accuracy.",
    icon: "⚙️",
  },
  {
    title: "Model persistence",
    desc: "The best model is checkpointed to models/traffic_sign_cnn.keras during training and reloaded by app.py/predict.py — no retraining on every start.",
    icon: "💾",
  },
  {
    title: "Streamlit UI",
    desc: "A clean Python-only web UI: file uploader, image preview, predict button, top-3 confidence bars and a performance dashboard — all in app.py.",
    icon: "🖥️",
  },
];

export default function Concepts() {
  return (
    <section>
      <SectionHeading
        id="concepts"
        eyebrow="Learn"
        title="ML concepts used in this project"
        desc="Everything a B.Tech AIML student needs to explain this project in a viva — each concept is implemented in train.py and visible in the dashboard above."
      />
      <div className="mx-auto mt-10 grid max-w-6xl gap-4 px-6 sm:grid-cols-2 lg:grid-cols-3">
        {CONCEPTS.map((c) => (
          <div
            key={c.title}
            className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 transition hover:border-teal-400/40 hover:bg-slate-900"
          >
            <div className="text-2xl">{c.icon}</div>
            <h3 className="mt-3 text-base font-bold text-white">{c.title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-slate-400">{c.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
