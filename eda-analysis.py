import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

sns.set_theme(style="whitegrid")

# Create graphs folder if it does not exist
os.makedirs("graphs", exist_ok=True)

print("=" * 60)
print("SWYNEX - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# 2. LOAD THE RAW KAGGLE DATASET
# --------------------------------------------------

print("\nLoading train.csv...")

df = pd.read_csv("train.csv")

print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 3. BASIC INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# --------------------------------------------------
# 4. CHECK DUPLICATES
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"\nNumber of duplicate rows: {duplicates}")


# --------------------------------------------------
# 5. CHECK MISSING VALUES
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df) * 100
)

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percentage
})

print("\nMissing values:")
print(missing_table)


# --------------------------------------------------
# 6. MISSING VALUE VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Heatmap")
plt.tight_layout()

plt.savefig(
    "graphs/missing_values_heatmap.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 7. CLEAN THE DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Fill Age missing values with median
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked missing values with mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

# Fill Fare missing values with median
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Fill Cabin missing values with "Unknown"
if "Cabin" in df.columns:
    df["Cabin"] = df["Cabin"].fillna("Unknown")

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# --------------------------------------------------
# 8. SAVE CLEANED DATASET
# --------------------------------------------------

df.to_csv(
    "Titanic-Cleaned.csv",
    index=False
)

print("\nTitanic-Cleaned.csv created successfully!")


# --------------------------------------------------
# 9. DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())


# --------------------------------------------------
# 10. SURVIVAL ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("SURVIVAL ANALYSIS")
print("=" * 60)

print("\nSurvival counts:")
print(df["Survived"].value_counts())

print("\nSurvival percentages:")
print(
    df["Survived"].value_counts(normalize=True) * 100
)


# --------------------------------------------------
# 11. SURVIVAL VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    "graphs/survival_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 12. GENDER ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("GENDER ANALYSIS")
print("=" * 60)

print(
    df.groupby("Sex")["Survived"].mean()
)

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="Sex",
    y="Survived"
)

plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "graphs/survival_by_gender.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 13. CLASS ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("PASSENGER CLASS ANALYSIS")
print("=" * 60)

print(
    df.groupby("Pclass")["Survived"].mean()
)

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="Pclass",
    y="Survived"
)

plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "graphs/survival_by_class.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 14. AGE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    kde=True
)

plt.title("Age Distribution")

plt.tight_layout()

plt.savefig(
    "graphs/age_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 15. AGE VS SURVIVAL
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Survived",
    y="Age"
)

plt.title("Age Distribution by Survival")

plt.tight_layout()

plt.savefig(
    "graphs/age_vs_survival.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 16. FARE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Fare",
    kde=True
)

plt.title("Fare Distribution")

plt.tight_layout()

plt.savefig(
    "graphs/fare_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 17. CORRELATION ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

correlation = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation)


# --------------------------------------------------
# 18. CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "graphs/correlation_heatmap.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 19. OUTLIER ANALYSIS
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Fare"
)

plt.title("Fare Outlier Analysis")

plt.tight_layout()

plt.savefig(
    "graphs/fare_outliers.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 20. FINAL DATASET CHECK
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("\nFinal shape:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nFirst 5 cleaned rows:")
print(df.head())


print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)