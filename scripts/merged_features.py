import pandas as pd

# Load cleaned datasets
vg_sales = pd.read_csv("cleaned_vgsales.csv")
top_grossing = pd.read_csv("cleaned_topgrossing.csv")

# Merge on Game name
merged = pd.merge(vg_sales, top_grossing[['Game', 'Units_Sold']], on='Game', how='left')

# Flag: Top Grossing Game (1 if match found, else 0)
merged['Top_Grossing_Flag'] = merged['Units_Sold'].notnull().astype(int)

# Fill missing Units_Sold with 0 for non-top grossing games
merged['Units_Sold'] = merged['Units_Sold'].fillna(0)

# Optional: drop original Platform and Genre after encoding
categorical_columns = ['Platform', 'Genre']

# One-hot encode Platform and Genre
merged_encoded = pd.get_dummies(merged, columns=categorical_columns, prefix=categorical_columns)

# Save the resulting feature set
merged_encoded.to_csv("feature_set_1.csv", index=False)

print("✅ Feature Set 1 saved as 'feature_set_1.csv' with one-hot encoding applied to Platform and Genre.")
