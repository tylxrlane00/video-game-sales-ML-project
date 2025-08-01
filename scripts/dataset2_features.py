import pandas as pd
from datetime import datetime

# Load the top grossing dataset
df = pd.read_csv("cleaned_topgrossing.csv")

# Create Franchise_Flag: 1 if Series is not null
df['Franchise_Flag'] = df['Series'].notnull().astype(int)

# Create Platform_Group (basic keyword mapping)
def map_platform(platform):
    platform = platform.lower()
    if "pc" in platform or "windows" in platform:
        return "PC"
    elif "multi" in platform:
        return "MULTI"
    else:
        return "CONSOLE"

df['Platform_Group'] = df['Platform'].apply(map_platform)

# Parse release date to compute Years_Since_Release (if applicable)
try:
    df['Initial_Release'] = pd.to_datetime(df['Initial release date'], errors='coerce')
    current_year = datetime.now().year
    df['Years_Since_Release'] = current_year - df['Initial_Release'].dt.year
except:
    df['Years_Since_Release'] = None

# Save feature set
df.to_csv("feature_set_2.csv", index=False)

print("✅ Feature Set 2 created and saved as 'feature_set_2.csv'")
