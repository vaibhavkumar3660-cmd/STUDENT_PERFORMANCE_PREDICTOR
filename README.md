# Task 1: Student Score Prediction — Explanation

## 1. Goal
Predict a student's **Exam Score** from their **study habits**, mainly **Hours Studied**, using linear regression. Then go further with polynomial regression and multiple features.

## 2. About the dataset
This now runs on the **real Kaggle "Student Performance Factors" dataset** (6,607 students, 20 columns). The dataset also has categorical columns (`Parental_Involvement`, `Motivation_Level`, `School_Type`, `Peer_Influence`, `Gender`, etc.) that aren't used in this task — this project focuses on the numeric features most relevant to regression:
`Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity` → predicting `Exam_Score`.

(`generate_dataset.py` is kept in this folder only as a fallback — it creates a synthetic version with the same columns, useful if you ever need to test the pipeline without the real file. `main.py` is already pointed at the real CSV.)

## 3. Step-by-step walkthrough

### Step 1 — Load the data
Straightforward `pd.read_csv()`, then `.shape`, `.head()`, `.info()` to get a first look at size, columns, and data types.

### Step 2 — Data cleaning
Real datasets are never perfectly clean. Here we:
- **Check for missing values** with `df.isnull().sum()`.
- **Fill missing numeric values with the median** of that column (median is used instead of mean because it's less sensitive to outliers).
- **Drop duplicate rows** with `df.drop_duplicates()`.
- **Sanity-check the target range** (Exam_Score should be between 0–100). The real dataset actually has one row with `Exam_Score = 101`, clearly a data-entry glitch — this step catches and removes it.

This is a minimal but standard cleaning routine — enough to make the model trustworthy without over-engineering it. On the real data, there were no missing values in the numeric features used, one exact duplicate row, and the one out-of-range score mentioned above.

### Step 3 — Exploratory Data Analysis (EDA)
Three visualizations are generated:
1. **Histogram of Exam Scores** — shows the overall spread/shape of the target variable.
2. **Scatter plot: Hours Studied vs Exam Score** — this is the key relationship the whole task is about. If there's a visible upward trend, linear regression is a reasonable choice.
3. **Correlation heatmap** — shows how strongly every numeric feature relates to every other, especially to `Exam_Score`. This tells us which features are worth including later.

In the real data, the two strongest predictors are `Attendance` (correlation ≈ **0.58**) and `Hours_Studied` (correlation ≈ **0.45**) — attendance actually edges out study hours as the single best predictor here. `Sleep_Hours` and `Physical_Activity` barely correlate with score at all (close to 0). The task asks us to focus on `Hours_Studied`, so that's still our main feature for the simple model, but this is a good real-world reminder to always check correlations before assuming which feature matters most.

### Step 4 — Train/test split
`train_test_split(X, y, test_size=0.2, random_state=42)` holds back 20% of the data as a test set the model never sees during training. This is how we get an honest estimate of performance on "new" students — `random_state=42` just makes the split reproducible.

### Step 5 — Simple Linear Regression
We fit `Exam_Score = intercept + coefficient × Hours_Studied` using scikit-learn's `LinearRegression`. The fitted line on the real data is:

```
Exam_Score ≈ 61.47 + 0.29 × Hours_Studied
```

Meaning: every extra hour of studying per week is associated with about **+0.29 points** on the exam score, on average, in this simple one-feature model. That's a smaller slope than you might expect — a hint that hours studied alone doesn't tell the whole story (see Step 8).

### Step 6 — Evaluation metrics
Four standard regression metrics are computed on the **held-out test set**:

| Metric | What it means |
|---|---|
| **MAE** (Mean Absolute Error) | Average size of the error, in the same units as the score (points) |
| **MSE** (Mean Squared Error) | Like MAE but penalizes large errors more heavily |
| **RMSE** (Root MSE) | Same units as the score, easier to interpret than MSE |
| **R²** (R-squared) | Fraction of the variance in exam scores explained by the model (0 = no better than guessing the mean, 1 = perfect) |

Our results for the simple model on the real data:
- MAE ≈ 2.48, RMSE ≈ 3.36, **R² ≈ 0.21**

That means hours studied alone only explains about 21% of the variation in exam scores. The MAE tells a more forgiving story though — predictions are typically off by only about 2.5 points, which is because most students in this dataset score in a fairly narrow band (mostly 60–75). R² looks modest mainly because there just isn't a huge spread to explain with one feature — this is exactly why the bonus experiments below matter.

Two plots support this:
- **Regression line over the actual test data** — visually shows the fit.
- **Predicted vs Actual scatter** — points closer to the diagonal dashed line mean better predictions.

### Step 7 (Bonus) — Polynomial Regression
We also fit **degree-2 and degree-3 polynomial regressions** (still using `LinearRegression`, but on polynomial-expanded features via `PolynomialFeatures`). This lets the model capture slight curvature instead of forcing a straight line.

Result: R² barely moved at all (0.214 → 0.215 → 0.215 for degree 2 and 3). This tells us the relationship between hours studied and score is **essentially a straight line** in this dataset — adding curve-fitting complexity doesn't help, which is a useful and common finding (not every project needs a complex model, and it's good practice to check rather than assume).

### Step 8 (Bonus) — Feature combination experiments
We compared several multiple-linear-regression models, each using a different set of input features:

| Feature set | R² | MAE | RMSE |
|---|---|---|---|
| Hours only | 0.214 | 2.48 | 3.36 |
| **Hours + Attendance** | **0.520** | **1.56** | **2.63** |
| Hours + Sleep | 0.215 | 2.47 | 3.36 |
| Hours + Previous Scores | 0.240 | 2.42 | 3.30 |
| **All features** | **0.574** | **1.34** | **2.47** |

This is the most important finding in the whole project: **adding `Attendance` alone more than doubles R²** (0.21 → 0.52), while adding `Sleep_Hours` does almost nothing (0.21 → 0.215). Using every available feature together gives the best result overall (R² ≈ 0.57, MAE ≈ 1.34 — predictions typically off by just over a point).

**Lesson:** not all features are equally useful. `Attendance` and `Hours_Studied` are the two features doing almost all the work; `Sleep_Hours` and `Physical_Activity` add very little in this dataset. This is exactly why it matters to test feature combinations with real metrics instead of assuming "more features = automatically better," and it's a good talking point for an internship review — it shows you understand *why* a model improves, not just that it does.

## 4. Files produced
```
student_score_project/
├── generate_dataset.py          # creates a synthetic fallback dataset (not used by main.py anymore)
├── main.py                      # the full pipeline: cleaning → EDA → modeling → evaluation
├── data/
│   └── StudentPerformanceFactors_real.csv   # the real Kaggle dataset you provided
└── outputs/
    ├── 01_exam_score_distribution.png
    ├── 02_hours_vs_score_scatter.png
    ├── 03_correlation_heatmap.png
    ├── 04_linear_regression_fit.png
    ├── 05_predicted_vs_actual.png
    ├── 06_polynomial_vs_linear.png
    ├── 07_feature_comparison_r2.png
    └── feature_comparison.csv
```

## 5. How to explain this in an interview / report
If asked to summarize this project in one paragraph:

> "I built a linear regression model to predict exam scores from study hours using the real Kaggle 'Student Performance Factors' dataset (6,600+ students). After cleaning the data (removing a duplicate row and one invalid score above 100) and exploring correlations, I found attendance and study hours were the two strongest predictors. A simple linear model using only hours studied explained about 21% of the variance in scores (R² = 0.21). Polynomial regression didn't improve on this, showing the relationship is essentially linear. The interesting finding came from testing feature combinations: adding attendance alone more than doubled R² to 0.52, and using all available features pushed it to 0.57 with an average error of about 1.3 points — while features like sleep hours barely helped at all. This shows that picking the *right* features matters more than just adding more of them."

## 6. Possible next steps (if you want to extend this further)
- Try `Ridge`/`Lasso` regression to handle potential multicollinearity between features.
- Encode and include categorical features from the real dataset (e.g. `Parental_Involvement`, `Motivation_Level`) using `pd.get_dummies()`.
- Use `cross_val_score` for more robust evaluation instead of a single train/test split.
- Try a non-linear model (e.g. `RandomForestRegressor`) as a benchmark against linear regression.
