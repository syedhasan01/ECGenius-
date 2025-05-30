import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

# === STEP 1: Load the datasets ===
df = pd.read_excel("model data.xlsx")

# === STEP 3: Prepare features and labels ===
X = df[['Min', 'Max', 'Mean', 'Std']]  # Features
y = df['label']  # Target: 0 = Normal, 1 = Abnormal

# === STEP 4: Split into training and test sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 5: Train Random Forest model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === STEP 6: Predict and evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# === STEP 7: Show metrics ===
print("==== Model Evaluation ====")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# === STEP 8: Save the trained model ===
joblib.dump(model, "ecg_random_forest_model.pkl")
print("\nModel saved as 'ecg_random_forest_model.pkl'")