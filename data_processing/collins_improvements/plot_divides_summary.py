import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load summarized CSV
df = pd.read_csv("multidate_FIRA_nextgen_timeseries.csv")
print(df.head())

# Pick a specific time to plot
chosen_time = "2004-01-01 00:00:00"
df_time = df[df["time"] == chosen_time]

divides = gpd.read_file("C:/Users/colli/Downloads/hi_nextgen.gpkg", layer="divides")
print(divides.head())

# Merge on divide_id
merged = divides.merge(df_time, on="divide_id")
print(merged.head())

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
merged.plot(column="FIRA", cmap="Blues", legend=True, ax=ax)#can add ,edgecolor="black"
ax.set_title(f"FIRA on {chosen_time}")
plt.tight_layout()
plt.show()
