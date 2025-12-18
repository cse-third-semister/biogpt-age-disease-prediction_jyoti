import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import joblib
# Load processed data
df = pd.read_csv("data/processed_data.csv")

# Separate features and label
X = df.drop("label", axis=1)
y = df["label"]

# Remove text feature for ML model
X_numeric = X.drop("symptoms_text", axis=1)

# Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_numeric, y, test_size=0.2, random_state=42, stratify=y
# )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_numeric, y), 1):
    X_train, X_test = X_numeric.iloc[train_idx], X_numeric.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}")


print("\nCross-Validation Results:")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
print(f"Std Deviation: {np.std(accuracies):.4f}")


final_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

final_model.fit(X_numeric, y)

joblib.dump(final_model, "disease_model.pkl")
print("✅ Final model trained on full data and saved as disease_model.pkl")



''' 
Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)


model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



joblib.dump(model, "disease_model.pkl")
print("✅ Model saved as disease_model.pkl")
'''