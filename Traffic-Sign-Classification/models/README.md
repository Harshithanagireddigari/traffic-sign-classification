# Models folder

After running:

```
python train.py
```

the following files are created here automatically:

| File | Description |
|------|-------------|
| `traffic_sign_cnn.keras` | The trained CNN model (loaded by `app.py` / `predict.py`) |
| `training_history.png`   | Training vs validation accuracy & loss curves |
| `confusion_matrix.png`   | Confusion matrix on the official GTSRB test set |
| `evaluation.json`        | Test accuracy/loss, per-epoch history, classification report |

The Streamlit app (`streamlit run app.py`) and the CLI (`python predict.py ...`)
load `traffic_sign_cnn.keras` from this folder, so the model is trained **once**
and reused every time the application starts.
