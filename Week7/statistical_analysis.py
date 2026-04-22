import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------------
# LOAD DATA
# -------------------------------
sales_df = pd.read_csv("sales_data.csv")
churn_df = pd.read_csv("customer_churn.csv")

sales_df.columns = sales_df.columns.str.strip().str.upper()
churn_df.columns = churn_df.columns.str.strip().str.upper()

# -------------------------------
# DESCRIPTIVE STATISTICS
# -------------------------------
print("\n===== DESCRIPTIVE STATISTICS =====")

mean_sales = sales_df["TOTAL_SALES"].mean()
median_sales = sales_df["TOTAL_SALES"].median()
std_sales = sales_df["TOTAL_SALES"].std()

print(f"Mean Sales: {mean_sales}")
print(f"Median Sales: {median_sales}")
print(f"Standard Deviation: {std_sales}")

# -------------------------------
# DISTRIBUTION (HISTOGRAM)
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(sales_df["TOTAL_SALES"], kde=True)
plt.title("Sales Distribution")
plt.savefig("sales_distribution.png")
plt.show()

# -------------------------------
# CORRELATION ANALYSIS
# -------------------------------
print("\n===== CORRELATION =====")

corr = sales_df.select_dtypes(include='number').corr()
print(corr)

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# -------------------------------
# HYPOTHESIS TEST 1 (T-TEST)
# Example: Compare two regions
# -------------------------------
regions = sales_df["REGION"].unique()

if len(regions) >= 2:
    group1 = sales_df[sales_df["REGION"] == regions[0]]["TOTAL_SALES"]
    group2 = sales_df[sales_df["REGION"] == regions[1]]["TOTAL_SALES"]

    t_stat, p_value = stats.ttest_ind(group1, group2)

    print("\n===== T-TEST RESULT =====")
    print("T-statistic:", t_stat)
    print("P-value:", p_value)

# -------------------------------
# HYPOTHESIS TEST 2 (NORMALITY TEST)
# -------------------------------
stat, p = stats.shapiro(sales_df["TOTAL_SALES"].sample(50))
print("\n===== NORMALITY TEST =====")
print("P-value:", p)

# -------------------------------
# HYPOTHESIS TEST 3 (ANOVA)
# -------------------------------
groups = [sales_df[sales_df["REGION"] == r]["TOTAL_SALES"] for r in regions]

if len(groups) > 2:
    f_stat, p_val = stats.f_oneway(*groups)
    print("\n===== ANOVA TEST =====")
    print("F-stat:", f_stat)
    print("P-value:", p_val)

# -------------------------------
# CONFIDENCE INTERVAL
# -------------------------------
confidence = 0.95
n = len(sales_df["TOTAL_SALES"])
mean = mean_sales
se = std_sales / np.sqrt(n)

margin = stats.t.ppf((1 + confidence) / 2, n-1) * se

print("\n===== CONFIDENCE INTERVAL =====")
print(f"{mean - margin:.2f} to {mean + margin:.2f}")

# -------------------------------
# REGRESSION ANALYSIS
# -------------------------------
if "QUANTITY" in sales_df.columns:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        sales_df["QUANTITY"], sales_df["TOTAL_SALES"]
    )

    print("\n===== REGRESSION =====")
    print("R-squared:", r_value**2)

# -------------------------------
# SAVE RESULTS
# -------------------------------
with open("hypothesis_tests_results.txt", "w") as f:
    f.write("Statistical Analysis Results\n")
    f.write(f"Mean Sales: {mean_sales}\n")
    f.write(f"Confidence Interval: {mean - margin} to {mean + margin}\n")

print("\n✅ Analysis Completed Successfully")