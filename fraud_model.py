import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Select useful columns
df = df[[
    "step",
    "type",
    "amount",
    "isFlaggedFraud",
    "isFraud"
]]

# Convert transaction type to numbers
encoder = LabelEncoder()

df["type"] = encoder.fit_transform(df["type"])

# Features and target
X = df.drop("isFraud", axis=1)

y = df["isFraud"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "models/paysim_model.pkl")

# Save encoder
joblib.dump(encoder, "models/type_encoder.pkl")

print("Model Saved Successfully!")
