import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import geopandas as gpd
import os

# Load JSONL Data
def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Load scraped data
filename = "scraped_data.jsonl"
data = load_jsonl(filename)

# Convert to DataFrame
df = pd.DataFrame(data)

# Print basic info
print("Total records:", len(df))
print("Columns:", df.columns.tolist())

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Add text length column
df['text_length'] = df['output'].apply(lambda x: len(x.split()))

# Print statistics
print("Text length statistics:")
print(df['text_length'].describe())

# Plot distribution of text lengths
plt.figure(figsize=(10,5))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.xlabel("Text Length (words)")
plt.ylabel("Frequency")
plt.title("Distribution of Article Lengths")
plt.show()

# Load U.S. cities dataset
us_cities = pd.read_csv("uscities.csv")[["city", "state_name"]]
city_state_dict = {row["city"].lower(): row["state_name"] for _, row in us_cities.iterrows()}
major_cities = set(["new york", "los angeles", "chicago", "las vegas", "san francisco", "miami", "orlando", "washington", "boston", "seattle"])

# Extract city names efficiently
city_counts = Counter()
city_content_length = Counter()
city_data = []

for text in df['output']:
    words = set(re.findall(r'\b[A-Za-z]+\b', text.lower()))
    matched_cities = {city for city in major_cities if re.search(rf"\b{re.escape(city)}\b", text.lower())}

    for city in matched_cities:
        state = city_state_dict[city]
        city_counts[city.title()] += 1
        city_content_length[city.title()] += len(text.split())
        city_data.append((state, city.title()))

# Ensure only valid city names are used
valid_city_df = pd.DataFrame(city_data, columns=["State", "City"]).drop_duplicates()
valid_city_df = valid_city_df[valid_city_df["City"].isin(city_counts.keys())]

# Print total cities mentioned
print("Total unique U.S. cities mentioned:", len(city_counts))

# Save to CSV with error handling
output_filename = "us_cities_travel_data.csv"
try:
    valid_city_df.to_csv(output_filename, index=False)
    print(f"City data saved to {output_filename}")
except PermissionError:
    print(f"Permission denied: Unable to write to {output_filename}. Trying a new filename.")
    output_filename = "us_cities_travel_data_new.csv"
    valid_city_df.to_csv(output_filename, index=False)
    print(f"City data saved to {output_filename}")

# Convert city counts to DataFrame for better visualization
city_plot_df = pd.DataFrame({
    "City": list(city_counts.keys()),
    "Count": list(city_counts.values()),
    "Content_Length": [city_content_length[city] for city in city_counts.keys()]
})

# Filter for only top cities with valid state names
city_plot_df = city_plot_df.merge(valid_city_df, on="City").drop_duplicates()

# Plot city frequency using a horizontal bar chart for better clarity
plt.figure(figsize=(12, 8))
sns.barplot(data=city_plot_df.sort_values(by="Count", ascending=False).head(10), 
            y="City", x="Count", palette="Blues_r")
plt.xlabel("Mention Count")
plt.ylabel("City")
plt.title("Top 10 Mentioned U.S. Cities in Travel Data")
plt.show()

# Print the top 10 most frequently mentioned cities
print("Most frequently mentioned cities:")
top_cities_mentioned = city_plot_df.sort_values(by="Count", ascending=False).head(10)
print(top_cities_mentioned[["State", "City", "Count"]])

# Print the top 10 cities with the most content along with their state
print("Top 10 cities with the most content:")
top_cities_content = city_plot_df.sort_values(by="Content_Length", ascending=False).head(10)
print(top_cities_content[["State", "City", "Content_Length"]])

# Print dataset insights
print("Insights:")
print(f"Average text length: {df['text_length'].mean():.2f} words")
print(f"Shortest article: {df['text_length'].min()} words")
print(f"Longest article: {df['text_length'].max()} words")