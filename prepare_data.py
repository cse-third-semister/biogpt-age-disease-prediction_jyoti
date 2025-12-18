import pandas as pd
import numpy as np

# ================= LOAD DATASETS =================
heart = pd.read_csv("data/heart.csv")
diabetes = pd.read_csv("data/diabetes.csv")
alz = pd.read_csv("data/alzheimer.csv")
print(alz["Diagnosis"].value_counts())


# ================= HEART DATA =================
heart_df = pd.DataFrame()

heart_df["age"] = heart["age"]
heart_df["bp"] = heart["resting_bp_s"]
heart_df["sugar"] = heart["fasting_blood_sugar"]
heart_df["cholesterol"] = heart["cholesterol"]
heart_df["memory_score"] = 0
heart_df["symptoms_text"] = "chest pain"

heart_df["disease"] = heart["target"].apply(
    lambda x: "Heart" if x == 1 else "Healthy"
)

# ================= DIABETES DATA =================
diab_df = pd.DataFrame()

diab_df["age"] = diabetes["Age"]
diab_df["bp"] = diabetes["BloodPressure"]
diab_df["sugar"] = diabetes["Glucose"]
diab_df["cholesterol"] = diabetes["BMI"]
diab_df["memory_score"] = 0
diab_df["symptoms_text"] = "high blood sugar"

diab_df["disease"] = diabetes["Outcome"].apply(
    lambda x: "Diabetes" if x == 1 else "Healthy"
)

# ================= ALZHEIMER DATA =================
# ================= ALZHEIMER DATA =================
alz_df = pd.DataFrame()

alz_df["age"] = alz["Age"]
alz_df["bp"] = alz["SystolicBP"]
alz_df["sugar"] = alz["Diabetes"]
alz_df["cholesterol"] = alz["CholesterolTotal"]
alz_df["memory_score"] = alz["MMSE"]
alz_df["symptoms_text"] = "memory loss and confusion"

# IMPORTANT: Diagnosis is numeric (0/1)
alz_df["disease"] = alz["Diagnosis"].apply(
    lambda x: "Alzheimer" if x == 1 else "Healthy"
)


# ================= UNIFY ALL =================
unified_df = pd.concat([heart_df, diab_df, alz_df], ignore_index=True)

# Replace missing values with 0
unified_df.fillna(0, inplace=True)

# Save unified dataset
unified_df.to_csv("data/unified_data.csv", index=False)

print("✅ unified_data.csv created successfully!")
print("\nSample rows:")
print(unified_df.head())
print("\nClass distribution:")
print(unified_df["disease"].value_counts())
