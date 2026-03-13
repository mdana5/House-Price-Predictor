import pandas as pd

# ===============================
# 1. Load Original Dataset
# ===============================
df = pd.read_csv("House price datasets/train.csv")

# ===============================
# 2. Create Age Feature
# ===============================
df["Age"] = 2026 - df["YearBuilt"]

# ===============================
# 3. Create Total Bathrooms
# ===============================
df["TotalBathrooms"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)

# ===============================
# 4. Create Location Column
# ===============================
df["Location"] = df["Neighborhood"]

# ===============================
# 5. Select Final Features
# ===============================
df_features = df[
    [
        "GrLivArea",
        "BedroomAbvGr",
        "Age",
        "OverallQual",
        "TotalBathrooms",
        "GarageCars",
        "TotalBsmtSF",
        "Location",
        "SalePrice"
    ]
]

# ===============================
# 6. Handle Missing Values
# ===============================
df_features = df_features.fillna(0)

# ===============================
# 7. Encode Location
# ===============================
df_features = pd.get_dummies(df_features, columns=["Location"], drop_first=True)

# ===============================
# 8. Convert Bool to Int
# ===============================
df_features = df_features.astype(int)

# ===============================
# 9. Save Processed Dataset
# ===============================
df_features.to_csv("housingfinalprocessed.csv", index=False)

print("Dataset successfully processed and saved!")
print(df_features.head())