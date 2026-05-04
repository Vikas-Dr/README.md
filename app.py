from pathlib import Path

import joblib
import numpy as np
import streamlit as st

MODEL_PATH = Path("models/iris_rf.joblib")

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris ML Classifier")
st.caption("Random Forest model trained on the Iris dataset")

if not MODEL_PATH.exists():
    st.warning("Model not found yet. Click below to train a model now.")
    if st.button("Train model"):
        from src.train import train_model

        score = train_model(MODEL_PATH)
        st.success(f"Model trained successfully. Accuracy: {score:.2%}")
        st.rerun()
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
target_names = bundle["target_names"]
feature_names = bundle["feature_names"]
accuracy = bundle.get("accuracy")

if accuracy is not None:
    st.info(f"Saved model validation accuracy: {accuracy:.2%}")

st.subheader("Enter flower features")
values = []
col1, col2 = st.columns(2)
for idx, feature in enumerate(feature_names):
    with col1 if idx % 2 == 0 else col2:
        values.append(
            st.number_input(
                feature,
                min_value=0.0,
                max_value=10.0,
                value=5.0 if "length" in feature else 3.0,
                step=0.1,
            )
        )

if st.button("Predict species"):
    sample = np.array(values).reshape(1, -1)
    pred_idx = int(model.predict(sample)[0])
    probs = model.predict_proba(sample)[0]

    st.success(f"Prediction: **{target_names[pred_idx]}**")
    st.write("Class probabilities")
    for i, species in enumerate(target_names):
        st.write(f"- {species}: {probs[i]:.2%}")
