import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load unified dataset
df = pd.read_csv("data/unified_data.csv")

# ---------------- FEATURES & LABEL ----------------
X = df.drop("disease", axis=1)
y = df["disease"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label mapping (important for later)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Normalize numeric features
numeric_cols = ["age", "bp", "sugar", "cholesterol", "memory_score"]
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Combine back
processed_df = X.copy()
processed_df["label"] = y_encoded

# Save processed dataset
processed_df.to_csv("data/processed_data.csv", index=False)

print("✅ processed_data.csv created successfully!")
print(processed_df.head())




joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as scaler.pkl")
