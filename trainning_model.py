# ============================================================
# BANK MARKETING PREDICTION — ANN TRAINING SCRIPT (WITH EPOCHS)
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATASETS
# ============================================================

train_df = pd.read_csv("train (11).csv", sep=";")
test_df = pd.read_csv("test (4).csv", sep=";")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

for dataset in [train_df, test_df]:
    dataset.columns = (
        dataset.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

# ============================================================
# 3. COMBINE & DEDUPLICATE
# ============================================================

df = pd.concat([train_df, test_df], ignore_index=True)

print("\nDuplicates before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after:", df.duplicated().sum())

df.to_csv("bank.csv", sep=";", index=False)
print("\nCombined dataset saved as bank.csv")

# ============================================================
# 4. CONVERT TARGET
# ============================================================

df["y"] = df["y"].astype(str).str.strip().str.lower()
df["y"] = df["y"].map({"yes": 1, "no": 0})

if df["y"].isna().sum() > 0:
    print("Removing invalid target values")
    df = df.dropna(subset=["y"])

df["y"] = df["y"].astype(int)

print("\nTarget Distribution:")
print(df["y"].value_counts())

# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("y", axis=1)
y = df["y"]

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical Columns:", categorical_columns)
print("\nNumerical Columns:", numerical_columns)

# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============================================================
# 7. BUILD ANN PIPELINE (MLPClassifier with StandardScaler)
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)

# Artificial Neural Network (Multi-Layer Perceptron) with verbose=True to show epochs/loss
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    verbose=True  # <--- Prints epoch loss updates to the terminal
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ============================================================
# 8. TRAIN AND EVALUATE
# ============================================================

print("\nTraining Artificial Neural Network (ANN) Pipeline...")

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("ANN MODEL RESULTS")
print("======================================")
print(f"Accuracy: {round(accuracy * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# 9. SAVE PIPELINE
# ============================================================

joblib.dump(pipeline, "model_ann.pkl")

print("\n======================================")
print("SUCCESS!")
print("======================================")
print("model_ann.pkl (ANN pipeline) created successfully!")