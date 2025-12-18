

import pandas as pd
import streamlit as st
import numpy as np
import joblib
from explain_with_biogpt import explain_prediction

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


scaler = joblib.load("scaler.pkl")
# Load trained ML model
model = joblib.load("disease_model.pkl")
alz_model = joblib.load("alzheimer_model.pkl")
alz_scaler = joblib.load("alzheimer_scaler.pkl")


# Label mapping (must match preprocessing)
label_map = {
    0: "Alzheimer",
    1: "Diabetes",
    2: "Healthy",
    3: "Heart"
}

st.set_page_config(page_title="Age-Related Disease Prediction", layout="centered")

st.title("🧠 Age-Related Disease Prediction System")
st.subheader("Hybrid ML + BioGPT (Explainable AI)")

st.markdown("""
This system predicts the **risk of age-related diseases** using clinical data  
and provides **AI-based medical explanations**.

⚠️ *This is not a medical diagnosis.*
""")

st.divider()

# ================= USER INPUT =================
age = st.number_input("Age", min_value=1, max_value=120, value=60)

bp = st.number_input("Blood Pressure", value=120.0)
sugar = st.number_input("Blood Sugar", value=90.0)
cholesterol = st.number_input("Cholesterol", value=180.0)
memory_score = st.number_input("Memory Score (MMSE)", value=28.0)

symptoms = st.text_area(
    "Symptoms",
    placeholder="e.g., chest pain, fatigue, memory loss"
)

# ================= PREDICTION =================
if st.button("🔍 Predict Disease"):
    
    # # ===== ALZHEIMER BINARY CHECK =====
    # alz_input = pd.DataFrame(
    # [[age, memory_score, bp, cholesterol]],
    # columns=["age", "memory_score", "bp", "cholesterol"]
    # )

    # alz_scaled = alz_scaler.transform(alz_input)
    # alz_prob = alz_model.predict_proba(alz_scaled)[0][1]
  

    # input_df = pd.DataFrame(
    # [[age, bp, sugar, cholesterol, memory_score]],
    # columns=["age", "bp", "sugar", "cholesterol", "memory_score"]
    # )

    # input_data_scaled = scaler.transform(input_df)


    # # Predict probabilities
    # probs = model.predict_proba(input_data_scaled)[0]

    # pred_label = np.argmax(probs)
    # disease = label_map[pred_label]
    # confidence = probs[pred_label]

    # # Risk scoring
    # if disease == "Healthy":
    #     risk = "Low"
    # else:
    #     if confidence >= 0.7:
    #         risk = "High"
    #     elif confidence >= 0.4:
    #         risk = "Medium"
    #     else:
    #         risk = "Low"
    
    
    # ===== ALZHEIMER BINARY CHECK =====
    alz_input = pd.DataFrame(
        [[age, memory_score, bp, cholesterol]],
        columns=["age", "memory_score", "bp", "cholesterol"]
    )

    alz_scaled = alz_scaler.transform(alz_input)
    alz_prob = alz_model.predict_proba(alz_scaled)[0][1]

    # ===== DECISION LOGIC =====
    if alz_prob >= 0.6:
        disease = "Alzheimer"
        confidence = alz_prob
        risk = "High"
    else:
        # ===== MULTICLASS MODEL =====
        input_df = pd.DataFrame(
            [[age, bp, sugar, cholesterol, memory_score]],
            columns=["age", "bp", "sugar", "cholesterol", "memory_score"]
        )

        input_data_scaled = scaler.transform(input_df)
        probs = model.predict_proba(input_data_scaled)[0]
        pred_label = np.argmax(probs)

        disease = label_map[pred_label]
        confidence = probs[pred_label]

        if disease == "Healthy":
            risk = "Low"
        elif confidence >= 0.7:
            risk = "High"
        elif confidence >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

    # st.success(f"**Predicted Disease:** {disease}")
    # st.info(f"**Risk Level:** {risk}")
    # st.write(f"**Prediction Confidence:** {confidence:.2f}")
    
    CONFIDENCE_THRESHOLD = 0.40

    st.write(f"Prediction Confidence: {confidence:.2f}")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Prediction inconclusive due to low confidence. Further clinical evaluation is recommended.")
    else:
        st.success(f"Predicted Disease: {disease}")
        st.info(f"Risk Level: {risk}")


    # ================= BIOGPT EXPLANATION =================
    # with st.spinner("Generating medical explanation..."):
    #     explanation = explain_prediction(
    #         age=age,
    #         disease=disease,
    #         probability=confidence,
    #         symptoms=symptoms
    #     )

    # st.subheader("🧾 Explanation")
    # st.write(explanation)
    
     # ===== BIOGPT EXPLANATION =====
    with st.spinner("Generating medical explanation..."):
            # explanation = explain_prediction(
            #     age=age,
            #     disease=disease,
            #     probability=confidence,
            #     symptoms=symptoms
            # )
        explanation = explain_prediction(
            age=age,
            disease=disease,
            risk=risk,
            probability=confidence,
            symptoms=symptoms
        )

        # ===== COLUMN FORMAT OUTPUT =====
    st.subheader("🧾 Explanation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Age**")
        st.markdown("**Predicted Disease**")
        st.markdown("**Risk Level**")
        st.markdown("**Confidence**")
        st.markdown("**Symptoms**")

    with col2:
        st.write(age)
        st.write(disease)
        st.write(risk)
        st.write(f"{confidence:.2f}")
        st.write(symptoms if symptoms else "Not provided")

    st.markdown("### 🧠 AI Medical Reasoning")
    st.write(explanation)

        # ===== DOWNLOAD REPORT =====
        # ===== PDF REPORT GENERATION =====
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(pdf_buffer)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>AI-Based Age-Related Disease Prediction Report</b>", styles["Title"]))
    content.append(Paragraph("<br/>", styles["Normal"]))

    content.append(Paragraph(f"<b>Age:</b> {age}", styles["Normal"]))
    content.append(Paragraph(f"<b>Predicted Disease:</b> {disease}", styles["Normal"]))
    content.append(Paragraph(f"<b>Risk Level:</b> {risk}", styles["Normal"]))
    content.append(Paragraph(f"<b>Prediction Confidence:</b> {confidence:.2f}", styles["Normal"]))

    content.append(Paragraph("<br/><b>Symptoms:</b>", styles["Normal"]))
    content.append(Paragraph(symptoms if symptoms else "Not provided", styles["Normal"]))

    content.append(Paragraph("<br/><b>AI Medical Explanation:</b>", styles["Normal"]))
    content.append(Paragraph(explanation, styles["Normal"]))

    content.append(Paragraph(
        "<br/><b>Disclaimer:</b> This report is generated by an AI system and is for educational purposes only.",
        styles["Normal"]
    ))

    doc.build(content)
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="disease_prediction_report.pdf",
        mime="application/pdf"
    )



    st.warning("⚠️ This system is for educational purposes only and is not a medical diagnosis.")
