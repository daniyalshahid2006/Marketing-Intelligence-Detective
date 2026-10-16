
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
DATA_FILE = Path(__file__).parent / "sentiment_human_vs_llm.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

sentiment_columns = ["Human", "GPT", "Claude", "Gemini", "Llama", "Copilot"]

# Standardize sentiment labels
label_mapping = {
    "POS": "POSITIVE",
    "POSTIVE": "POSITIVE",
    "NEUTAL": "NEUTRAL",
    "NUETAL": "NEUTRAL",
    "NUETRAL": "NEUTRAL"
}

for column in sentiment_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(label_mapping)
    )

valid_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Keep rows with valid human judgments
df = df[df["Human"].isin(valid_labels)].copy()

# Compare each AI model against human judgments
results = []

for model in sentiment_columns:
    if model == "Human":
        continue

    valid = df[model].isin(valid_labels)
    actual = df.loc[valid, "Human"]
    predicted = df.loc[valid, model]

    correct = (actual == predicted).sum()
    total = len(actual)
    accuracy = correct / total * 100 if total else 0

    results.append({
        "Model": model,
        "Correct": int(correct),
        "Total": total,
        "Accuracy (%)": round(accuracy, 2)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Accuracy (%)", ascending=False
)

# Display results
print("\nDATASET OVERVIEW")
print("----------------")
print("Comments:", len(df))
print("Columns:", len(df.columns))

print("\nHUMAN SENTIMENT DISTRIBUTION")
print("----------------------------")
print(df["Human"].value_counts())

print("\nAI MODEL ACCURACY")
print("-----------------")
print(results_df.to_string(index=False))

# Save reports
results_df.to_csv(
    OUTPUT_DIR / "model_accuracy.csv", index=False
)

df.to_csv(
    OUTPUT_DIR / "cleaned_sentiment_data.csv", index=False
)

# Chart 1: Model accuracy
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy (%)"])
plt.ylabel("Agreement with human labels (%)")
plt.title("AI Model vs Human Sentiment Agreement")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_accuracy.png", dpi=200)
plt.close()

# Chart 2: Human sentiment distribution
sentiment_counts = (
    df["Human"]
    .value_counts()
    .reindex(valid_labels, fill_value=0)
)

plt.figure(figsize=(7, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.ylabel("Number of comments")
plt.title("Human-Labeled Sentiment Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=200)
plt.close()

print("\nFiles generated in the outputs folder:")
print("- model_accuracy.csv")
print("- cleaned_sentiment_data.csv")
print("- model_accuracy.png")
print("- sentiment_distribution.png")

print("\nAnalysis completed successfully!")
outputs/cleaned_sentiment_data.csv