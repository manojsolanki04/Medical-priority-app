"""Train the model once and save it as a .pkl file.
Run:  python train_model.py "path/to/Medical_Appointment_Priority_Dataset.csv"
"""
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

csv_path = sys.argv[1] if len(sys.argv) > 1 else "Medical_Appointment_Priority_Dataset.csv"

df = pd.read_csv(csv_path).drop_duplicates()

# Blood pressure -> two numeric columns
bp = df["Blood_Pressure"].str.split("/", expand=True)
df["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
df["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")
df = df.drop(columns=["Blood_Pressure", "Patient_ID"])

# Date -> day / month / weekday (the model can't use a raw datetime column)
df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"], errors="coerce")
df["Appointment_Day"] = df["Appointment_Date"].dt.day
df["Appointment_Month"] = df["Appointment_Date"].dt.month
df["Appointment_Weekday"] = df["Appointment_Date"].dt.weekday
df = df.drop(columns=["Appointment_Date"])

X = df.drop(columns=["Priority_Label"])
y = df["Priority_Label"]

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

grid = GridSearchCV(
    pipeline,
    {"model__n_estimators": [100, 200],
     "model__max_depth": [10, None],
     "model__min_samples_split": [2, 5]},
    cv=5, scoring="f1_weighted", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print(classification_report(y_test, grid.best_estimator_.predict(X_test)))

joblib.dump(grid.best_estimator_, "medical_appointment_priority_model.pkl")
print("Saved medical_appointment_priority_model.pkl")
