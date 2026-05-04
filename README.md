# Iris ML Project with Streamlit UI

A small end-to-end machine learning project that:
- Trains a **Random Forest** classifier on the Iris dataset.
- Stores a reusable model artifact in `models/iris_rf.joblib`.
- Provides an interactive **Streamlit UI** for inference.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Train the model

```bash
python src/train.py
```

## 3) Run the UI

```bash
streamlit run app.py
```

## Notes
- If no trained model exists, the UI provides a **Train model** button.
- Default feature values are prefilled so you can quickly test predictions.
