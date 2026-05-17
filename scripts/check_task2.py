import pandas as pd

df = pd.read_csv("data/raw/task2_output.csv")

print("\n--- HEAD ---")
print(df.head())

print("\n--- SENTIMENT DISTRIBUTION ---")
print(df["sentiment_label"].value_counts())

print("\n--- THEME DISTRIBUTION ---")
print(df["identified_theme"].value_counts())