"""EMG → hand gesture decoder (NinaPro preprocessed + Keras models)."""

import os
import sys
from pathlib import Path

# Ensure project root is on path when Streamlit loads this page.
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from src.config import MODEL_DIR, PREPROCESS_PATH
from src.emg_movement.gestures import ALL_GESTURES

id_to_gesture = dict(enumerate(ALL_GESTURES))

for key in ["emg_data", "labels", "model"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.set_page_config(page_title="EMG Hand Movement Decoder", layout="wide")
st.title("EMG Semantic Hand Movement Decoder")

st.sidebar.header("Data Selection")
st.sidebar.radio("Choose data source:", ["Use preprocessed data"])

if not os.path.isdir(PREPROCESS_PATH):
    st.sidebar.error("No preprocessed data directory found.")
    available_datasets = []
else:
    available_datasets = [
        d
        for d in os.listdir(PREPROCESS_PATH)
        if os.path.isdir(os.path.join(PREPROCESS_PATH, d)) and d.isdigit()
    ]

if not available_datasets:
    st.sidebar.error("No preprocessed datasets found.")
else:
    selected_dataset = st.sidebar.selectbox("Choose a dataset", available_datasets)
    dataset_path = os.path.join(PREPROCESS_PATH, selected_dataset)
    x_path = os.path.join(dataset_path, "X.npy")
    y_path = os.path.join(dataset_path, "y.npy")

    if st.sidebar.button("Load dataset"):
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            st.sidebar.error("X.npy or y.npy is missing in the selected folder.")
        else:
            with st.spinner("Loading dataset..."):
                st.session_state.emg_data = np.load(x_path)
                st.session_state.labels = np.load(y_path)
            st.success(f"Dataset '{selected_dataset}' loaded!")
            st.write("**EMG data shape:**", st.session_state.emg_data.shape)
            st.write("**Labels shape:**", st.session_state.labels.shape)

st.sidebar.header("Model Selection")
if not os.path.isdir(MODEL_DIR):
    st.sidebar.error("No models directory found.")
    available_models = []
else:
    available_models = [f for f in os.listdir(MODEL_DIR) if not f.startswith(".")]

if not available_models:
    st.sidebar.error("No models found in src/models/")
else:
    selected_model_name = st.sidebar.selectbox("Choose a model", available_models)
    if selected_model_name and st.sidebar.button("Load Model"):
        with st.spinner(f"Loading model {selected_model_name}..."):
            st.session_state.model = tf.keras.models.load_model(
                os.path.join(MODEL_DIR, selected_model_name)
            )
        st.sidebar.success("Model loaded!")

model = st.session_state.model
emg_data = st.session_state.emg_data
labels = st.session_state.labels

if st.button("Run Inference") and emg_data is not None and model is not None:
    with st.spinner("Running inference..."):
        total = int(len(emg_data) / 100)
        correct = 0
        predictions = []
        progress_bar = st.progress(0)

        for idx in range(total):
            sample = np.expand_dims(emg_data[idx], axis=0)
            probs = model.predict(sample, verbose=0)[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            pred_gesture = id_to_gesture[pred_id]
            true_id = int(labels[idx]) - 1
            true_gesture = id_to_gesture[true_id]
            is_correct = pred_id == true_id
            if is_correct:
                correct += 1

            predictions.append(
                {
                    "Sample": idx,
                    "Predicted Gesture": pred_gesture,
                    "Confidence": confidence,
                    "True Gesture": true_gesture,
                    "Correct": is_correct,
                }
            )

            progress_bar.progress((idx + 1) / total)

        df_pred = pd.DataFrame(predictions)

        accuracy = correct / total * 100
        st.header("Inference metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Accuracy", f"{accuracy:.2f}%")
        col2.metric("Total Samples", total)
        col3.metric("Correct Predictions", correct)

        st.subheader("Per-gesture accuracy")
        per_gesture_acc = df_pred.groupby("Predicted Gesture")["Correct"].mean() * 100
        per_gesture_acc_df = per_gesture_acc.reset_index().rename(
            columns={"Correct": "Accuracy (%)"}
        )

        chart = (
            alt.Chart(per_gesture_acc_df)
            .mark_bar()
            .encode(
                x=alt.X("Predicted Gesture:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Accuracy (%):Q"),
                tooltip=["Predicted Gesture", "Accuracy (%)"],
            )
            .properties(width=800)
        )

        st.altair_chart(chart, use_container_width=True)

        st.subheader("Confidence distribution")
        conf_df = df_pred.groupby("Predicted Gesture")["Confidence"].mean().reset_index()

        chart_conf = (
            alt.Chart(conf_df)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("Predicted Gesture:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Confidence:Q"),
                tooltip=["Predicted Gesture", "Confidence"],
            )
            .properties(width=800)
        )

        st.altair_chart(chart_conf, use_container_width=True)

        st.subheader("Predictions table")
        st.dataframe(df_pred)
