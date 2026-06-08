import joblib
import numpy as np
import os
import pandas as pd
# import shap

# =====================================================
# CLEANING FUNCTION
# =====================================================
def clean_value(x):

    if pd.isna(x):
        return None

    x = str(x)
    x = x.strip()
    x = x.lower()
    x = x.replace(" ", "_")

    return x if x != "" else None


# =====================================================
# BASE DIRECTORY
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================
# LOAD MODEL & ENCODER
# =====================================================
model_path = os.path.join(BASE_DIR, "disease_model.pkl")
encoder_path = os.path.join(BASE_DIR, "symptom_encoder.pkl")

model = joblib.load(model_path)
mlb = joblib.load(encoder_path)

valid_symptoms = list(mlb.classes_)


# =====================================================
# LOAD DATASETS
# =====================================================
desc_path = os.path.join(BASE_DIR, "symptom_Description.csv")
severity_path = os.path.join(BASE_DIR, "Symptom-severity.csv")
precaution_path = os.path.join(BASE_DIR, "symptom_precaution.csv")

df_desc = pd.read_csv(desc_path)
df_severity = pd.read_csv(severity_path)
df_precaution = pd.read_csv(precaution_path)


# =====================================================
# CLEAN DATASETS
# =====================================================
df_desc["Disease"] = df_desc["Disease"].apply(clean_value)
df_precaution["Disease"] = df_precaution["Disease"].apply(clean_value)
df_severity["Symptom"] = df_severity["Symptom"].apply(clean_value)


# =====================================================
# FIX DUPLICATE SEVERITY
# =====================================================
severity_dict = (
    df_severity
    .drop_duplicates("Symptom")
    .set_index("Symptom")["weight"]
    .to_dict()
)

desc_dict = dict(zip(df_desc["Disease"], df_desc["Description"]))
precaution_dict = df_precaution.set_index("Disease").to_dict(orient="index")


# =====================================================
# CALCULATE TOTAL SEVERITY SCORE
# =====================================================
def calculate_severity(symptom_list):

    total_weight = 0

    for symptom in symptom_list:

        symptom_clean = clean_value(symptom)

        weight = severity_dict.get(symptom_clean, 0)

        try:

            if isinstance(weight, (list, np.ndarray, pd.Series)):
                weight = weight[0]

            total_weight += int(float(weight))

        except Exception:

            total_weight += 0

    return total_weight


# =====================================================
# ENCODE SYMPTOMS
# =====================================================
def encode_symptoms(symptom_list):

    symptom_list_clean = [
        clean_value(s) for s in symptom_list if clean_value(s)
    ]

    encoded = mlb.transform([symptom_list_clean])

    return encoded.astype(float)


# =====================================================
# SHAP EXPLAINABILITY (SAFE)
# =====================================================
# def compute_shap_values(symptom_list):

#     try:

#         encoded = encode_symptoms(symptom_list)

#         explainer = shap.TreeExplainer(model)

#         shap_values = explainer(
#             encoded,
#             check_additivity=False
#         )

#         # ambil hanya sample pertama
#         return shap_values[0]

#     except Exception as e:

#         print("SHAP error:", e)

#         return None


# =====================================================
# MAIN PREDICTION
# =====================================================
def predict_disease(symptom_list):

    symptom_list_clean = [
        clean_value(s) for s in symptom_list if clean_value(s)
    ]

    if not symptom_list_clean:
        return None

    encoded = encode_symptoms(symptom_list_clean)

    probs = model.predict_proba(encoded)[0]

    top3_idx = np.argsort(probs)[-3:][::-1]

    severity_score = calculate_severity(symptom_list_clean)

    results = []

    for idx in top3_idx:

        disease_raw = model.classes_[idx]

        disease = clean_value(disease_raw)

        confidence = float(probs[idx])

        description = desc_dict.get(
            disease,
            "Deskripsi tidak tersedia."
        )

        prec_data = precaution_dict.get(disease)

        if prec_data:

            precautions = [
                p for p in [
                    prec_data.get("Precaution_1"),
                    prec_data.get("Precaution_2"),
                    prec_data.get("Precaution_3"),
                    prec_data.get("Precaution_4")
                ]
                if pd.notna(p) and str(p).strip() != ""
            ]

        else:

            precautions = ["Tidak tersedia"]

        results.append({
            "disease": disease,
            "confidence": confidence,
            "description": description,
            "precautions": precautions
        })

    return {
        "severity_score": severity_score,
        "top3": results
    }


# =====================================================
# CONFIDENCE CHECK
# =====================================================
def apply_confidence_threshold(results_dict):

    if not results_dict:
        return "no_input"

    top_confidence = results_dict["top3"][0]["confidence"]

    if top_confidence < 0.5:
        return "low_confidence"

    return "ok"


# =====================================================
# CLASSIFY SEVERITY
# =====================================================
def classify_severity(severity_score):

    if severity_score <= 7:
        return "Low"

    elif severity_score <= 14:
        return "Medium"

    else:
        return "High"


# =====================================================
# EMERGENCY FLAG
# =====================================================
def check_emergency(severity_level):

    return severity_level == "High"


# =====================================================
# FINAL PIPELINE
# =====================================================
def run_full_prediction(symptom_list):

    results = predict_disease(symptom_list)

    if results is None:

        return {
            "error": "Tidak ada gejala valid yang dimasukkan."
        }

    confidence_status = apply_confidence_threshold(results)

    severity_level = classify_severity(
        results["severity_score"]
    )

    emergency_flag = check_emergency(
        severity_level
    )

    return {
        "severity_score": results["severity_score"],
        "severity_level": severity_level,
        "confidence_status": confidence_status,
        "emergency_flag": emergency_flag,
        "top3": results["top3"]
    }