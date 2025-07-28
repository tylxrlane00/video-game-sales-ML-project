import pandas as pd

# Load the dataset from the new path
df = pd.read_csv("C:/Users/mfunm/Downloads/archive(1)/Top_Selling_Games.csv")

# Rename columns to standardized names
df.rename(columns={
    "Game Title": "Game",
    "Platform(s)": "Platform",
    "Units Sold (millions)": "Units_Sold",
    "Publisher(s)[b]": "Publisher",
    "Developer(s)[b]": "Developer"
}, inplace=True)

# Drop rows with missing critical data
df.dropna(subset=["Game", "Platform", "Units_Sold"], inplace=True)

# Normalize text fields
df["Game"] = df["Game"].str.lower().str.strip()
df["Platform"] = df["Platform"].str.upper().str.strip()

# Fill optional fields with 'Unknown'
df["Publisher"] = df["Publisher"].fillna("Unknown")
df["Developer"] = df["Developer"].fillna("Unknown")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Save cleaned dataset to current directory
df.to_csv("cleaned_topgrossing.csv", index=False)

print("✅ Cleaned dataset saved as 'cleaned_topgrossing.csv'")
