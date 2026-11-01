import { SectionHeading } from "./ui";

const STEPS = [
  {
    n: "1",
    title: "Install Python 3.10+ (Windows)",
    cmd: ["python --version", "python -m venv .venv", ".venv\\Scripts\\activate"],
    desc: "Download Python from python.org and tick “Add Python to PATH” during setup. A virtual environment keeps dependencies clean.",
  },
  {
    n: "2",
    title: "Download the GTSRB dataset",
    cmd: ["# extract into Traffic-Sign-Classification/dataset/GTSRB"],
    desc: "Official: benchmark.ini.rub.de/gtsrb_dataset.html · Kaggle mirror: kaggle.com/datasets/meowmeowmeowmeowmeowmeow/gtsrb-german-traffic-sign. Layout is documented in dataset/README.md.",
  },
  {
    n: "3",
    title: "Install dependencies",
    cmd: ["pip install -r requirements.txt"],
    desc: "Installs TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pillow and Streamlit.",
  },
  {
    n: "4",
    title: "Train the CNN (run once)",
    cmd: ["python train.py"],
    desc: "Loads ~39k images, preprocesses them (32×32, /255), trains the CNN with early stopping and saves models/traffic_sign_cnn.keras plus graphs and evaluation.json.",
  },
  {
    n: "5",
    title: "Classify an image from the CLI",
    cmd: ["python predict.py path\\to\\sign.png"],
    desc: "Prints the top-3 predictions with class id, sign name and confidence percentage.",
  },
  {
    n: "6",
    title: "Start the Streamlit web app",
    cmd: ["streamlit run app.py"],
    desc: "Opens http://localhost:8501 — upload JPG/JPEG/PNG images, see the 32×32 preprocessed preview, prediction, confidence, top-3 and the model-performance page.",
  },
];

export default function RunLocally() {
  return (
    <section>
      <SectionHeading
        id="run"
        eyebrow="Run locally"
        title="Run the full Python project on Windows"
        desc="The complete mini-project is in Traffic-Sign-Classification/. Follow these steps from inside that folder."
      />

      <div className="mx-auto mt-10 grid max-w-6xl gap-6 px-6 lg:grid-cols-[1fr_1.2fr]">
        {/* project structure */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <h3 className="text-base font-bold text-white">📁 Project structure</h3>
          <pre className="mt-4 overflow-x-auto font-mono text-xs leading-relaxed text-slate-400">
{`Traffic-Sign-Classification/
├── dataset/
│   ├── README.md            # GTSRB download + layout
│   └── GTSRB/               # <- extract dataset here
│       ├── Final_Training/Images/00000..00042/
│       └── Final_Test/Images/ + Test.csv
├── models/                  # created by train.py
│   ├── traffic_sign_cnn.keras
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── evaluation.json
├── sign_classes.py          # 43 class names
├── train.py                 # trains + saves the CNN
├── predict.py               # CLI prediction
├── app.py                   # Streamlit UI
├── requirements.txt
└── README.md`}
          </pre>
          <div className="mt-4 rounded-xl border border-teal-400/30 bg-teal-400/5 p-4 text-sm text-slate-300">
            💡 This web page is a companion demo: the same CNN architecture trains live in your browser (TensorFlow.js)
            on GTSRB-style synthetic signs, so you can try predictions without installing anything.
          </div>
        </div>

        {/* steps */}
        <div className="space-y-4">
          {STEPS.map((s) => (
            <div key={s.n} className="flex gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-teal-400 to-emerald-600 text-sm font-black text-slate-950">
                {s.n}
              </span>
              <div className="min-w-0">
                <h3 className="text-sm font-bold text-white">{s.title}</h3>
                <p className="mt-1 text-xs leading-relaxed text-slate-400">{s.desc}</p>
                <div className="mt-2.5 rounded-lg bg-slate-950 px-3 py-2 font-mono text-[11.5px] text-emerald-300">
                  {s.cmd.map((c, i) => (
                    <div key={i}>{c}</div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
