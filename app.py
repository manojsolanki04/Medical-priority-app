import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Appointment Priority", page_icon="🏥", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.result-card {padding: 1.4rem; border-radius: 12px; text-align: center; color: white;
              font-size: 1.6rem; font-weight: 700;}
.High {background:#d62828;} .Medium {background:#f77f00;} .Low {background:#2a9d8f;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("medical_appointment_priority_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python train_model.py <dataset.csv>` first.")
    st.stop()

# ---------- Header ----------
st.title("🏥 Medical Appointment Priority System")
st.caption("ML-based decision-support tool. It assists staff and does not replace clinical judgement.")

tab_predict, tab_about = st.tabs(["Predict", "About"])

with tab_predict:
    with st.form("patient_form"):
        st.subheader("Patient details")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 0, 120, 40)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        arrival = c3.selectbox("Arrival mode", ["Walk-in", "Ambulance", "Referral"])

        c1, c2 = st.columns(2)
        symptoms = c1.selectbox("Symptoms", [
            "Fever", "Shortness of Breath", "Back Pain", "Headache", "Dizziness",
            "Chest Pain", "High Blood Sugar", "Fracture", "Abdominal Pain", "Cough"])
        diagnosis = c2.selectbox("Diagnosis", [
            "Viral Infection", "Pneumonia", "Muscle Strain", "Migraine", "Hypertension",
            "Heart Attack", "Diabetes", "Bone Fracture", "Gastritis", "Bronchitis"])

        st.subheader("Vitals")
        c1, c2, c3, c4 = st.columns(4)
        heart_rate = c1.number_input("Heart rate (bpm)", 30, 250, 80)
        temperature = c2.number_input("Temperature (°C)", 30.0, 45.0, 37.0, step=0.1)
        oxygen = c3.slider("Oxygen saturation (%)", 50, 100, 98)
        pain = c4.slider("Pain scale", 0, 10, 5)
        c1, c2 = st.columns(2)
        systolic = c1.number_input("Systolic BP", 50, 250, 120)
        diastolic = c2.number_input("Diastolic BP", 30, 150, 80)

        st.subheader("Visit details")
        c1, c2, c3 = st.columns(3)
        emergency = c1.selectbox("Emergency level", ["Low", "Moderate", "High", "Critical"])
        specialty = c2.selectbox("Doctor specialty", [
            "General Medicine", "Pulmonology", "Orthopedics", "Neurology",
            "Cardiology", "Endocrinology", "Gastroenterology"])
        disposition = c3.selectbox("Disposition", ["Observation", "Admitted", "Discharged", "Referred"])

        c1, c2 = st.columns(2)
        waiting = c1.number_input("Waiting time (minutes)", 0, 1000, 20)
        appt_date = c2.date_input("Appointment date")

        submitted = st.form_submit_button("Predict priority", type="primary", use_container_width=True)

    if submitted:
        patient = pd.DataFrame([{
            "Age": age, "Gender": gender, "Symptoms": symptoms, "Diagnosis": diagnosis,
            "Heart_Rate": heart_rate, "Temperature_C": temperature,
            "Oxygen_Saturation": oxygen, "Pain_Scale": pain,
            "Emergency_Level": emergency, "Waiting_Time_Minutes": waiting,
            "Arrival_Mode": arrival, "Doctor_Specialty": specialty,
            "Disposition": disposition, "Systolic_BP": systolic, "Diastolic_BP": diastolic,
            "Appointment_Day": appt_date.day, "Appointment_Month": appt_date.month,
            "Appointment_Weekday": appt_date.weekday(),
        }])

        prediction = model.predict(patient)[0]
        probs = pd.Series(model.predict_proba(patient)[0], index=model.classes_)

        left, right = st.columns([1, 2])
        with left:
            st.markdown(f'<div class="result-card {prediction}">Priority: {prediction}</div>',
                        unsafe_allow_html=True)
            st.metric("Confidence", f"{probs.max():.0%}")
        with right:
            st.write("**Class probabilities**")
            st.bar_chart(probs)

with tab_about:
    st.markdown("""
**Model:** Random Forest pipeline (imputation, scaling, one-hot encoding), tuned with GridSearchCV.  
**Target:** `Priority_Label` (Low / Medium / High).  
**Use:** decision support only; final triage decisions belong to clinicians.
""")
