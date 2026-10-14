
import pandas as pd

# Load the dataset
df = pd.read_csv("sentiment_human_vs_llm.csv")

# Explore the dataset
print(df.head())

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

# Define sentiment columns
sentiment_columns = ["Human", "GPT", "Claude", "Gemini", "Llama", "Copilot"]

# Correct known spelling mistakes and abbreviations
label_mapping = {
    "POS": "POSITIVE",
    "POSTIVE": "POSITIVE",
    "NUETAL": "NEUTRAL",
    "NEUTAL": "NEUTRAL",
    "NUETRAL": "NEUTRAL"
}

# Clean every sentiment column
for column in sentiment_columns:
    df[column] = df[column].str.strip().str.upper()
    df[column] = df[column].replace(label_mapping)

# Display cleaned sentiment labels
print("\nCleaned sentiment counts:")

for column in sentiment_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())

# Verify unique labels
print("\nUnique labels in each column:")

for column in sentiment_columns:
    print(column, ":", df[column].unique())
