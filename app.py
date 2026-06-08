import streamlit as st
import os
# import shap
import matplotlib.pyplot as plt

from inference import run_full_prediction
from llm_reasoning import extract_symptoms, generate_reasoning
from gejala import symptoms


st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Diagnosis Assistant")
st.markdown("Sistem Prediksi Penyakit berbasis Machine Learning + LLM")


# =============================
# API KEY INPUT
# =============================

st.subheader("🔑 API Key (Opsional untuk Penjelasan AI)")

api_key = st.text_input(
    "Masukkan API Key (Gemini/OpenAI)",
    type="password"
)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


# =============================
# SESSION STATE
# =============================

if "mapped_symptoms" not in st.session_state:
    st.session_state.mapped_symptoms = []

if "final_symptoms" not in st.session_state:
    st.session_state.final_symptoms = []

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "explanation" not in st.session_state:
    st.session_state.explanation = ""


# =============================
# Mapping label
# =============================

label_to_key = {v["id"]: k for k, v in symptoms.items()}
key_to_label = {k: v["id"] for k, v in symptoms.items()}

all_labels = list(label_to_key.keys())


# =============================
# INPUT USER
# =============================

st.subheader("1️⃣ Masukkan Keluhan")

user_input = st.text_area(
    "Contoh: Saya demam dan batuk sejak kemarin",
    height=120
)


# =============================
# DETEKSI GEJALA
# =============================

if st.button("🔍 Deteksi Gejala"):

    if user_input.strip() == "":
        st.warning("Silakan masukkan keluhan.")
    else:

        detected_keys = extract_symptoms(user_input)

        st.session_state.mapped_symptoms = detected_keys
        st.session_state.final_symptoms = detected_keys
        st.session_state.prediction_data = None
        st.session_state.explanation = ""


# =============================
# KONFIRMASI GEJALA
# =============================

if st.session_state.mapped_symptoms:

    st.subheader("2️⃣ Konfirmasi Gejala")

    detected_labels = [
        key_to_label[k] for k in st.session_state.mapped_symptoms
    ]

    selected_labels = st.multiselect(
        "Gejala terdeteksi:",
        options=all_labels,
        default=detected_labels
    )

    selected_keys = [
        label_to_key[label] for label in selected_labels
    ]

    st.session_state.final_symptoms = selected_keys


    # =============================
    # PREDIKSI
    # =============================

    if st.button("🧠 Prediksi Penyakit"):

        prediction_data = run_full_prediction(selected_keys)

        if "error" in prediction_data:

            st.error(prediction_data["error"])

        else:

            explanation = generate_reasoning(
                selected_keys,
                prediction_data["top3"],
                prediction_data["severity_score"],
                api_key
            )

            st.session_state.prediction_data = prediction_data
            st.session_state.explanation = explanation


# =============================
# TAMPILKAN HASIL
# =============================

if st.session_state.prediction_data:

    data = st.session_state.prediction_data

    st.subheader("Hasil Prediksi")

    total_prob = 0

    for r in data["top3"]:

        percent = r["confidence"] * 100
        disease_name = r["disease"].replace("_", " ").title()

        st.markdown(f"### {disease_name} — {percent:.1f}%")
        st.progress(r["confidence"])

        st.write("**Deskripsi Penyakit:**")
        st.write(r["description"])

        st.write("**Penanganan:**")

        for p in r["precautions"]:
            if p:
                st.write(f"- {p}")

        st.markdown("---")

        total_prob += r["confidence"]

    remaining = (1 - total_prob) * 100

    st.info(
        f"Sisa kemungkinan penyakit lain sekitar {remaining:.1f}%"
    )


# =============================
# SEVERITY
# =============================

    st.subheader("📊 Severity Analysis")

    st.write(f"Severity Score: {data['severity_score']}")
    st.write(f"Severity Level: {data['severity_level']}")

    if data["confidence_status"] == "low_confidence":
        st.warning("⚠️ Confidence rendah, konsultasi dokter.")

    if data["emergency_flag"]:
        st.error("🚨 Gejala cukup berat.")

    # =============================
    # SHAP EXPLAINABILITY
    # =============================

    # st.subheader("🔍 Model Explainability (SHAP)")

    # shap_values = data.get("shap_values")

    # if shap_values is not None:

    #     try:
    #         fig = plt.figure()
    #         shap.plots.waterfall(shap_values)
    #         st.pyplot(fig)

    #     except Exception:
    #         st.info("Explainability tidak dapat ditampilkan.")

    # else:
    #     st.info("Explainability tidak tersedia untuk prediksi ini.")
# =============================
# AI EXPLANATION
# =============================

    st.subheader("🧾 Penjelasan AI")

    st.write(st.session_state.explanation)

    st.caption("⚠️ Sistem ini tidak menggantikan diagnosis dokter.")