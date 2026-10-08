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


SEX_AR = {"Female": "أنثى", "Male": "ذكر"}
GEN_AR = {"Poor": "ضعيفة", "Fair": "مقبولة", "Good": "جيدة", "Very good": "جيدة جداً", "Excellent": "ممتازة"}
DIAB_AR = {
    "No": "لا",
    "No, borderline diabetes": "لا، سكر على الحدود",
    "Yes (during pregnancy)": "نعم (أثناء الحمل)",
    "Yes": "نعم",
}
YN_AR = {"Yes": "نعم", "No": "لا"}


def bi(opts_ar):
    """Show 'English / عربي' for an option, but keep the English value."""
    return lambda v: f"{v} / {opts_ar[v]}" if v in opts_ar else ("80 أو أكثر / 80 or older" if v == "80 or older" else v)


def yn(label, key_default="No"):
    options = ["No", "Yes"] if key_default == "No" else ["Yes", "No"]
    return st.radio(label, options, format_func=bi(YN_AR), horizontal=True)


st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️")
st.title("❤️ Heart Disease Risk Estimator")
st.subheader("مقدّر خطر أمراض القلب")
st.caption(
    "Learning project trained on CDC 2020 survey data. "
    "It is NOT a medical tool and cannot diagnose anything.\n\n"
    "مشروع تعليمي مبني على بيانات استبيان من CDC لسنة 2020. "
    "ده مش أداة طبية ومينفعش يشخّص أي مرض."
)

model, threshold, columns = load_model()

with st.form("form"):
    name = st.text_input("Your name (optional) / اسمك (اختياري)", max_chars=30)
    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Sex / الجنس", ["Female", "Male"], format_func=bi(SEX_AR))
        age = st.selectbox("Age group / الفئة العمرية", list(AGE_MAP), index=6, format_func=bi({}))
        height = st.number_input("Height (cm) / الطول (سم)", 120, 220, 170)
        weight = st.number_input("Weight (kg) / الوزن (كجم)", 30, 200, 70)
        sleep = st.slider("Sleep (hours per night) / النوم (ساعات في الليلة)", 1, 24, 7)
        gen_health = st.selectbox("General health / صحتك العامة", list(GENHEALTH_MAP), index=2, format_func=bi(GEN_AR))
        diabetic = st.selectbox("Diabetes / السكر", list(DIABETIC_MAP), format_func=bi(DIAB_AR))
    with c2:
        smoking = yn("Smoked 100+ cigarettes in your life? / دخّنت 100 سيجارة أو أكتر في حياتك؟")
        alcohol = yn("Heavy drinker? / بتشرب كحول بكثرة؟")
        stroke = yn("Ever had a stroke? / جالك جلطة أو سكتة دماغية قبل كده؟")
        diff_walking = yn("Serious difficulty walking/climbing stairs? / صعوبة كبيرة في المشي أو طلوع السلم؟")
        activity = yn("Physical activity in the last 30 days? / مارست نشاط بدني في آخر 30 يوم؟", "Yes")
        asthma = yn("Asthma? / ربو؟")
        kidney = yn("Kidney disease? / مرض في الكلى؟")
        skin_cancer = yn("Skin cancer? / سرطان جلد؟")
    physical_days = st.slider(
        "Days in the last 30 with poor physical health / عدد الأيام في آخر 30 يوم كانت صحتك الجسدية فيها سيئة", 0, 30, 0)
    mental_days = st.slider(
        "Days in the last 30 with poor mental health / عدد الأيام في آخر 30 يوم كانت نفسيتك فيها سيئة", 0, 30, 0)
    submitted = st.form_submit_button("Estimate risk / احسب الخطر")

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
    st.subheader(f"Hi {clean_name} 👋 / أهلاً {clean_name}" if clean_name else "Your result 👋 / نتيجتك")
    st.metric("Your BMI / مؤشر كتلة جسمك", f"{bmi:.1f}")
    detail = f"(model score {score:.2f}, cut-off {threshold:.2f} / درجة الموديل {score:.2f}، حد الفصل {threshold:.2f})"
    if score >= threshold:
        st.error(f"Higher-risk profile / ملفك الصحي أعلى خطورة {detail}")
    else:
        st.success(f"Lower-risk profile / ملفك الصحي أقل خطورة {detail}")
    st.caption(
        "The score is a relative ranking, not a real probability. "
        "If you are worried about your health, talk to a doctor.\n\n"
        "الدرجة دي ترتيب نسبي بين الناس، مش احتمال حقيقي للإصابة. "
        "لو قلقان على صحتك كلّم دكتور."
    )
    if bmi > 43:
        st.warning(
            "BMI above ~43 is outside most of the training data, so the estimate is less reliable. / "
            "مؤشر كتلة الجسم فوق 43 تقريباً خارج أغلب بيانات التدريب، فالتقدير أقل دقة."
        )
