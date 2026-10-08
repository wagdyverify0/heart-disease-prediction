import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "heart_model.joblib"

AGE_MAP = {
    "18-24": 0, "25-29": 0, "30-34": 0,
    "35-39": 1, "40-44": 1, "45-49": 1,
    "50-54": 2, "55-59": 2, "60-64": 2,
    "65-69": 3, "70-74": 3, "75-79": 3,
    "80 or older": 4,
}
GENHEALTH_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
DIABETIC_MAP = {
    "No": 0,
    "No, borderline diabetes": 1,
    "Yes (during pregnancy)": 2,
    "Yes": 3,
}


def bmi_bin(v):
    # same bins as the training notebook
    if v < 16:
        return 0
    if v < 17:
        return 1
    if v < 18.5:
        return 2
    if v < 25:
        return 3
    if v < 30:
        return 4
    if v < 35:
        return 5
    if v < 40:
        return 6
    return 7


def sleep_bin(h):
    if h <= 6:
        return 0
    if h <= 8:
        return 1
    return 2


def days_bin(d):
    # PhysicalHealth / MentalHealth: bad days in the last 30
    if d <= 10:
        return 0
    if d <= 20:
        return 1
    if d <= 25:
        return 2
    return 3


def encode(a, columns):
    """Turn the form answers into the exact feature row the model was trained on."""
    yn = lambda x: 1 if x == "Yes" else 0
    row = {
        "BMI": bmi_bin(a["bmi"]),
        "Smoking": yn(a["smoking"]),
        "AlcoholDrinking": yn(a["alcohol"]),
        "Stroke": yn(a["stroke"]),
        "PhysicalHealth": days_bin(a["physical_days"]),
        "MentalHealth": days_bin(a["mental_days"]),
        "DiffWalking": yn(a["diff_walking"]),
        "Sex": 1 if a["sex"] == "Male" else 0,
        "AgeCategory": AGE_MAP[a["age"]],
        "Diabetic": DIABETIC_MAP[a["diabetic"]],
        "PhysicalActivity": yn(a["activity"]),
        "GenHealth": GENHEALTH_MAP[a["gen_health"]],
        "SleepTime": sleep_bin(a["sleep"]),
        "Asthma": yn(a["asthma"]),
        "KidneyDisease": yn(a["kidney"]),
        "SkinCancer": yn(a["skin_cancer"]),
    }
    return pd.DataFrame([row])[columns]


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["threshold"], bundle["columns"]


st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️")
st.title("❤️ Heart Disease Risk Estimator")
st.caption(
    "Learning project trained on CDC 2020 survey data. "
    "It is NOT a medical tool and cannot diagnose anything."
)

model, threshold, columns = load_model()

with st.form("form"):
    name = st.text_input("Your name (optional)", max_chars=30)
    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Sex", ["Female", "Male"])
        age = st.selectbox("Age group", list(AGE_MAP), index=6)
        height = st.number_input("Height (cm)", 120, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        sleep = st.slider("Sleep (hours per night)", 1, 24, 7)
        gen_health = st.selectbox("General health", list(GENHEALTH_MAP), index=2)
        diabetic = st.selectbox("Diabetes", list(DIABETIC_MAP))
    with c2:
        smoking = st.radio("Smoked 100+ cigarettes in your life?", ["No", "Yes"], horizontal=True)
        alcohol = st.radio("Heavy drinker?", ["No", "Yes"], horizontal=True)
        stroke = st.radio("Ever had a stroke?", ["No", "Yes"], horizontal=True)
        diff_walking = st.radio("Serious difficulty walking/climbing stairs?", ["No", "Yes"], horizontal=True)
        activity = st.radio("Physical activity in the last 30 days?", ["Yes", "No"], horizontal=True)
        asthma = st.radio("Asthma?", ["No", "Yes"], horizontal=True)
        kidney = st.radio("Kidney disease?", ["No", "Yes"], horizontal=True)
        skin_cancer = st.radio("Skin cancer?", ["No", "Yes"], horizontal=True)
    physical_days = st.slider("Days in the last 30 with poor physical health", 0, 30, 0)
    mental_days = st.slider("Days in the last 30 with poor mental health", 0, 30, 0)
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    bmi = weight / ((height / 100) ** 2)
    answers = dict(
        bmi=bmi, smoking=smoking, alcohol=alcohol, stroke=stroke,
        physical_days=physical_days, mental_days=mental_days,
        diff_walking=diff_walking, sex=sex, age=age, diabetic=diabetic,
        activity=activity, gen_health=gen_health, sleep=sleep,
        asthma=asthma, kidney=kidney, skin_cancer=skin_cancer,
    )
    x = encode(answers, columns)
    score = float(model.predict_proba(x)[0, 1])

    clean_name = re.sub(r"[^\w\s\-]", "", name, flags=re.UNICODE).strip()
    st.subheader(f"Hi {clean_name} 👋" if clean_name else "Your result 👋")
    st.metric("Your BMI", f"{bmi:.1f}")
    if score >= threshold:
        st.error(f"Higher-risk profile (model score {score:.2f}, cut-off {threshold:.2f})")
    else:
        st.success(f"Lower-risk profile (model score {score:.2f}, cut-off {threshold:.2f})")
    st.caption(
        "The score is a relative ranking from a class-weighted model, not a real "
        "probability. If you are worried about your health, talk to a doctor."
    )
    if bmi > 43:
        st.warning("BMI above ~43 is outside most of the training data, so the estimate is less reliable.")
