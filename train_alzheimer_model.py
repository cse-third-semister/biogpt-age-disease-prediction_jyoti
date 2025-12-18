import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/alzheimer_binary.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accs.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}")

print("\nMean Accuracy:", np.mean(accs))
print("Std Dev:", np.std(accs))

# Train final model
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

final_model.fit(X_scaled, y)

# Save model & scaler
joblib.dump(final_model, "alzheimer_model.pkl")
joblib.dump(scaler, "alzheimer_scaler.pkl")

print("✅ Alzheimer binary model saved")
