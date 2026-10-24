from __future__ import annotations

import os

import httpx
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👗", layout="centered")
st.title("Fashion-MNIST Classifier")
st.caption("Upload a fashion item image and get a predicted class from the LeNet-5 model.")


def _check_health() -> dict | None:
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


health = _check_health()
if health is None:
    st.error(f"Cannot reach API at {API_URL}. Make sure the prediction service is running.")
elif not health.get("model_loaded"):
    st.warning(
        "The API is reachable but no model is loaded. "
        "Run training first and restart the API with MODEL_RUN_ID set."
    )
else:
    run_id = health.get("model_run_id", "unknown")
    st.success(f"Model loaded (run `{run_id}`)")

st.divider()

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a grayscale or colour image; it will be converted to 28×28 grayscale automatically.",
)

if uploaded is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input image")
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        if st.button("Predict", use_container_width=True, type="primary"):
            uploaded.seek(0)
            with st.spinner("Running inference…"):
                try:
                    resp = httpx.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded, uploaded.type or "image/png")},
                        timeout=30,
                    )
                except httpx.RequestError as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            if resp.status_code == 200:
                data = resp.json()
                st.metric("Predicted class", data["predicted_class"])
                st.metric("Confidence", f"{data['confidence']:.1%}")

                st.subheader("All class probabilities")
                probs = data.get("probabilities", {})
                if probs:
                    import pandas as pd
                    df = (
                        pd.DataFrame(
                            {"Class": list(probs.keys()), "Probability": list(probs.values())}
                        )
                        .sort_values("Probability", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.bar_chart(df.set_index("Class"))
            elif resp.status_code == 503:
                st.warning("Model is not loaded on the server. Please start the API with a valid MODEL_RUN_ID.")
            else:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.error(f"Error {resp.status_code}: {detail}")
