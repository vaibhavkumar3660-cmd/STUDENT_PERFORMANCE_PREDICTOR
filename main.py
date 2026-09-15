"""
Student Score Prediction — Linear Regression Project
======================================================
Internship Task 1

Pipeline:
 1. Load data
 2. Clean data (nulls, duplicates, types)
 3. Explore data (basic stats + visualizations)
 4. Train/test split
 5. Simple linear regression: Hours_Studied -> Exam_Score
 6. Evaluate (MAE, MSE, RMSE, R2) + prediction plot
 7. BONUS: Polynomial regression vs linear
 8. BONUS: Multi-feature regression (add/remove features) vs simple model

All plots are saved as PNG files into the outputs/ folder so they can be
viewed outside a notebook too.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
# Using the real Kaggle "Student Performance Factors" dataset.
# NOTE: update this path to wherever the CSV lives on YOUR computer.
# If main.py and the CSV are in the same folder, just use the filename:
DATA_PATH = "StudentPerformanceFactors.csv"
df = pd.read_csv(DATA_PATH)

# The real dataset has extra columns (categorical) beyond what this task needs.
# We keep only the numeric columns relevant to this regression task.
FEATURE_COLUMNS = ["Hours_Studied", "Attendance", "Sleep_Hours",
                    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
df = df[FEATURE_COLUMNS + ["Exam_Score"]]

print("=" * 60)
print("STEP 1: RAW DATA OVERVIEW")
print("=" * 60)
print(df.shape)
print(df.head())
print(df.info())

# ----------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# 2a. Check missing values
print("Missing values per column:\n", df.isnull().sum())

# 2b. Handle missing values -> fill numeric columns with the column median
# (median is robust to outliers, better than mean for skewed data)
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {col} nulls with median = {median_val}")

# 2c. Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate row(s). New shape: {df.shape}")

# 2d. Sanity-check ranges (basic outlier screen)
df = df[(df["Exam_Score"] >= 0) & (df["Exam_Score"] <= 100)]
print("Final cleaned shape:", df.shape)

print("\nSummary statistics after cleaning:")
print(df.describe())

# ----------------------------------------------------------------------
# 3. BASIC VISUALIZATION / EDA
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY VISUALIZATION")
print("=" * 60)

# 3a. Distribution of Exam Scores
plt.figure(figsize=(7, 5))
plt.hist(df["Exam_Score"], bins=20, color="#4C72B0", edgecolor="white")
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(f"{OUT}/01_exam_score_distribution.png", dpi=120)
plt.close()

# 3b. Scatter: Hours Studied vs Exam Score (the core relationship)
plt.figure(figsize=(7, 5))
plt.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.5, color="#55A868")
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied per Week")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.savefig(f"{OUT}/02_hours_vs_score_scatter.png", dpi=120)
plt.close()

# 3c. Correlation heatmap (using matplotlib only, no seaborn dependency needed)
corr = df.corr(numeric_only=True)
plt.figure(figsize=(7, 6))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_heatmap.png", dpi=120)
plt.close()

print("Correlation of each feature with Exam_Score:")
print(corr["Exam_Score"].sort_values(ascending=False))

# ----------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT (simple model: Hours_Studied only)
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: TRAIN/TEST SPLIT")
print("=" * 60)

X_simple = df[["Hours_Studied"]]
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 5. SIMPLE LINEAR REGRESSION
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: SIMPLE LINEAR REGRESSION (Hours_Studied -> Exam_Score)")
print("=" * 60)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

print(f"Intercept: {lin_model.intercept_:.3f}")
print(f"Coefficient (Hours_Studied): {lin_model.coef_[0]:.3f}")
print(f"Equation: Exam_Score = {lin_model.intercept_:.2f} + "
      f"{lin_model.coef_[0]:.2f} * Hours_Studied")

y_pred_lin = lin_model.predict(X_test)

# ----------------------------------------------------------------------
# 6. EVALUATION + VISUALIZATION
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"MAE  : {mae:.3f}")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

print("\n" + "=" * 60)
print("STEP 6: EVALUATION - SIMPLE LINEAR MODEL")
print("=" * 60)
metrics_simple = evaluate(y_test, y_pred_lin, "Simple Linear Regression")

# 6a. Regression line over full data
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, alpha=0.5, label="Actual (test set)", color="#4C72B0")
x_line = np.linspace(X_simple.min().values[0], X_simple.max().values[0], 100).reshape(-1, 1)
y_line = lin_model.predict(pd.DataFrame(x_line, columns=["Hours_Studied"]))
plt.plot(x_line, y_line, color="red", linewidth=2, label="Linear fit")
plt.title("Linear Regression Fit: Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_linear_regression_fit.png", dpi=120)
plt.close()

# 6b. Predicted vs Actual scatter (how good are predictions?)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lin, alpha=0.6, color="#C44E52")
lims = [min(y_test.min(), y_pred_lin.min()), max(y_test.max(), y_pred_lin.max())]
plt.plot(lims, lims, "k--", label="Perfect prediction")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Predicted vs Actual (Simple Linear Model)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/05_predicted_vs_actual.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 7. BONUS: POLYNOMIAL REGRESSION
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 (BONUS): POLYNOMIAL REGRESSION")
print("=" * 60)

results_poly = {}
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, alpha=0.4, color="gray", label="Actual (test set)")

for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    m = evaluate(y_test, y_pred_poly, f"Polynomial Regression (degree={degree})")
    results_poly[degree] = m

    x_line_poly = poly.transform(x_line)
    y_line_poly = poly_model.predict(x_line_poly)
    plt.plot(x_line, y_line_poly, linewidth=2, label=f"Degree {degree} fit")

plt.plot(x_line, y_line, color="red", linewidth=2, linestyle="--", label="Linear fit (degree 1)")
plt.title("Linear vs Polynomial Regression Fits")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/06_polynomial_vs_linear.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 8. BONUS: DIFFERENT FEATURE COMBINATIONS (MULTIPLE LINEAR REGRESSION)
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 (BONUS): FEATURE COMBINATION EXPERIMENTS")
print("=" * 60)

feature_sets = {
    "Hours only": ["Hours_Studied"],
    "Hours + Attendance": ["Hours_Studied", "Attendance"],
    "Hours + Sleep": ["Hours_Studied", "Sleep_Hours"],
    "Hours + Previous_Scores": ["Hours_Studied", "Previous_Scores"],
    "All features": ["Hours_Studied", "Attendance", "Sleep_Hours",
                      "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"],
}

comparison_rows = []
for name, feats in feature_sets.items():
    X = df[feats]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression()
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    comparison_rows.append({"Feature set": name, "R2": round(r2, 3),
                             "MAE": round(mae, 3), "RMSE": round(rmse, 3)})
    print(f"{name:28s} | R2={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(f"{OUT}/feature_comparison.csv", index=False)

# Bar chart comparing R2 across feature sets
plt.figure(figsize=(8, 5))
plt.bar(comparison_df["Feature set"], comparison_df["R2"], color="#8172B2")
plt.title("R² Score by Feature Combination")
plt.ylabel("R² Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/07_feature_comparison_r2.png", dpi=120)
plt.close()

print("\nAll outputs (plots + CSV) saved in:", OUT)
print("\nDONE.")