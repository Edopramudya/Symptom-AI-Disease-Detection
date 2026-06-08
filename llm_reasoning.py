import re
import os
import google.generativeai as genai

from gejala import symptoms


# =========================
# DETEKSI GEJALA
# =========================

def extract_symptoms(user_input):

    text = user_input.lower()
    detected_keys = []

    for key, value in symptoms.items():

        for synonym in value["synonyms"]:

            pattern = r"\b" + re.escape(synonym.lower()) + r"\b"

            if re.search(pattern, text):
                detected_keys.append(key)
                break

    return list(set(detected_keys))


# =========================
# GENERATE AI REASONING
# =========================

def generate_reasoning(symptoms_keys, results, severity_score, api_key):

    if not api_key:
        return "🔑 Penjelasan AI dinonaktifkan. Masukkan API Key Gemini/OpenAI untuk mendapatkan penjelasan medis otomatis."

    try:

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        model = genai.GenerativeModel("gemini-1.5-flash")

        symptom_labels = [
            symptoms[k]["id"] for k in symptoms_keys if k in symptoms
        ]

        disease_names = [
            r["disease"].replace("_", " ").title()
            for r in results
        ]

        prompt = f"""
Anda adalah asisten medis.

Gejala pasien:
{", ".join(symptom_labels)}

Kemungkinan penyakit:
{", ".join(disease_names)}

Severity score: {severity_score}

Jelaskan hubungan gejala dengan penyakit tersebut
dengan bahasa sederhana agar mudah dipahami pasien.
"""

        response = model.generate_content(prompt)

        return response.text

    except Exception:

        return "AI tidak dapat memberikan penjelasan saat ini. Masukan API Key yang valid"