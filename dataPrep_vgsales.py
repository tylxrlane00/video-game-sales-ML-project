import pandas as pd

# Load the dataset
vg_sales = pd.read_csv("C:/Users/mfunm/Downloads/archive/vgsales.csv")

# Rename columns for consistency
vg_sales.rename(columns={
    "Name": "Game",
    "Year_of_Release": "Year"
}, inplace=True)

# Drop rows with missing values in key fields
vg_sales.dropna(subset=["Game", "Platform", "Genre", "Year"], inplace=True)

# Fill missing publisher with 'Unknown'
vg_sales["Publisher"] = vg_sales["Publisher"].fillna("Unknown")

# Normalize casing and trim whitespace
vg_sales["Game"] = vg_sales["Game"].str.lower().str.strip()
vg_sales["Platform"] = vg_sales["Platform"].str.upper().str.strip()

# Remove duplicate rows
vg_sales.drop_duplicates(inplace=True)

# Create derived fields
vg_sales["Total_Regional_Sales"] = vg_sales[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum(axis=1)
vg_sales["Era"] = pd.cut(vg_sales["Year"], bins=[1980, 1990, 2000, 2010, 2020],
                         labels=["80s", "90s", "00s", "10s"])

# Save cleaned file in current directory
vg_sales.to_csv("cleaned_vgsales.csv", index=False)

print("✅ Cleaned data saved as 'cleaned_vgsales.csv'. Rows with missing Year or Genre have been removed.")
