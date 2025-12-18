import pandas as pd

# Load unified data
df = pd.read_csv("data/unified_data.csv")

# Keep only Alzheimer and Healthy
df_binary = df[df["disease"].isin(["Alzheimer", "Healthy"])].copy()

# Binary label
df_binary["label"] = df_binary["disease"].map({
    "Healthy": 0,
    "Alzheimer": 1
})

# Select important features for Alzheimer
features = ["age", "memory_score", "bp", "cholesterol"]
df_binary = df_binary[features + ["label"]]

# Save
df_binary.to_csv("data/alzheimer_binary.csv", index=False)

print("✅ Alzheimer vs Healthy dataset created")
print(df_binary["label"].value_counts())
