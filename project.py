"""Synthetic clinic appointment exercise. Run from this file's folder."""
from pathlib import Path
import json
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
SITE = BASE / "site"
OUT.mkdir(exist_ok=True)
SITE.mkdir(exist_ok=True)
df = pd.read_json(BASE / "appointments.json")
print("Raw rows:", len(df))
df = df.drop_duplicates(subset="appointment_id", keep="first")
print("Clean rows:", len(df), "Missing values:", df.isna().sum().to_dict())



with sqlite3.connect(":memory:") as conn:
    df.to_sql("appointments", conn, index=False)
    summary = pd.read_sql_query("SELECT reminder_sent, COUNT(*) AS appointments, AVG(missed) AS missed_rate FROM appointments GROUP BY reminder_sent", conn)
print(summary)
summary.to_csv(OUT / "reminder_summary.csv", index=False)
summary.to_csv(SITE / "reminder_summary.csv", index=False)
summary.plot.bar(x="reminder_sent", y="missed_rate", legend=False)
plt.ylabel("Fraction of appointments missed")
plt.title("Synthetic appointment outcomes by reminder group")
plt.tight_layout()
plt.savefig(OUT / "reminder_chart.png")
plt.savefig(SITE / "reminder_chart.png")
plt.close()


features = ["lead_days", "prior_missed", "reminder_sent"]
X, y = df[features], df["missed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
model.fit(X_train, y_train)
baseline = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="most_frequent"))
baseline.fit(X_train, y_train)

imputer = model.named_steps["simpleimputer"]
scaler = model.named_steps["standardscaler"]
classifier = model.named_steps["logisticregression"]
model_parameters = {
    "features": features,
    "imputation_values": imputer.statistics_.tolist(),
    "means": scaler.mean_.tolist(),
    "scales": scaler.scale_.tolist(),
    "coefficients": classifier.coef_[0].tolist(),
    "intercept": float(classifier.intercept_[0]),
}
(SITE / "model.json").write_text(json.dumps(model_parameters, indent=2))


pred = model.predict(X_test)
base_pred = baseline.predict(X_test)
report = "Baseline accuracy: " + str(accuracy_score(y_test, base_pred))
report += "\nBaseline report\n" + classification_report(y_test, base_pred, zero_division=0)
report += "\nModel report\n" + classification_report(y_test, pred, zero_division=0)
report += "\nConfusion matrix [rows actual 0,1; columns predicted 0,1]\n"
report += str(confusion_matrix(y_test, pred, labels=[0,1]))
print(report)
(OUT / "metrics.txt").write_text(report)
(SITE / "metrics.txt").write_text(report)


new = pd.DataFrame([{"lead_days": 30, "prior_missed": 2, "reminder_sent": 0}])
probability = float(model.predict_proba(new)[0, 1])
result = {"input": new.to_dict("records")[0], "missed_score": probability,
          "prediction_at_0_5": int(probability >= 0.5)}
(OUT / "new_prediction.json").write_text(json.dumps(result, indent=2))
(SITE / "new_prediction.json").write_text(json.dumps(result, indent=2))
print("New appointment:", result)
