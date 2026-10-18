
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import re

# Project paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "sentiment_human_vs_llm.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

sentiment_columns = ["Human", "GPT", "Claude", "Gemini", "Llama", "Copilot"]
valid_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

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

# Keep rows with valid human labels
df = df[df["Human"].isin(valid_labels)].copy()

# 1. Compare AI models against human labels
results = []

for model in sentiment_columns:
    if model == "Human":
        continue

    valid = df[model].isin(valid_labels)
    actual = df.loc[valid, "Human"]
    predicted = df.loc[valid, model]

    correct = int((actual == predicted).sum())
    total = len(actual)
    accuracy = correct / total * 100 if total else 0

    results.append({
        "Model": model,
        "Correct": correct,
        "Total": total,
        "Accuracy (%)": round(accuracy, 2)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Accuracy (%)", ascending=False
)

# 2. Analyze human-labeled sentiment
sentiment_counts = (
    df["Human"]
    .value_counts()
    .reindex(valid_labels, fill_value=0)
)

total_comments = len(df)
negative_count = int(sentiment_counts["NEGATIVE"])
positive_count = int(sentiment_counts["POSITIVE"])
neutral_count = int(sentiment_counts["NEUTRAL"])

negative_pct = negative_count / total_comments * 100 if total_comments else 0
positive_pct = positive_count / total_comments * 100 if total_comments else 0
neutral_pct = neutral_count / total_comments * 100 if total_comments else 0

sentiment_report = pd.DataFrame({
    "Sentiment": valid_labels,
    "Comments": [positive_count, negative_count, neutral_count],
    "Percentage (%)": [
        round(positive_pct, 2),
        round(negative_pct, 2),
        round(neutral_pct, 2)
    ]
})

# 3. Extract common words from comments
stop_words = {
    "the", "and", "for", "are", "but", "not", "you", "your",
    "this", "that", "with", "was", "were", "have", "has",
    "had", "from", "they", "their", "its", "it's", "into",
    "about", "just", "very", "can", "could", "would", "will",
    "what", "when", "where", "who", "how", "all", "also",
    "been", "being", "than", "then", "there", "here", "our",
    "out", "get", "got", "too", "much", "more", "some",
    "really", "only", "like", "his", "her", "she", "him",
    "them", "we", "i", "me", "my", "it", "is", "a", "an",
    "to", "of", "in", "on", "at", "as", "or", "if", "be",
    "do", "so", "no", "yes"
}

word_counts = Counter()

if "Comment" in df.columns:
    for comment in df["Comment"].dropna():
        words = re.findall(r"\b[a-zA-Z]{3,}\b", str(comment).lower())
        word_counts.update(
            word for word in words if word not in stop_words
        )

top_words = word_counts.most_common(15)

word_report = pd.DataFrame(
    top_words, columns=["Word", "Frequency"]
)

# 4. Generate marketing recommendations
recommendations = []

if negative_pct >= 30:
    recommendations.append({
        "Finding": f"{negative_pct:.1f}% of comments are negative",
        "Recommendation": (
            "Investigate customer complaints, identify recurring issues, "
            "and prioritize improvements."
        ),
        "Priority": "High"
    })
elif negative_count > 0:
    recommendations.append({
        "Finding": f"{negative_pct:.1f}% of comments are negative",
        "Recommendation": (
            "Review negative comments and monitor whether the same issues "
            "appear repeatedly."
        ),
        "Priority": "Medium"
    })
else:
    recommendations.append({
        "Finding": "No negative comments were identified",
        "Recommendation": (
            "Continue monitoring feedback and verify that the dataset "
            "contains a representative sample."
        ),
        "Priority": "Medium"
    })

if positive_count > 0:
    recommendations.append({
        "Finding": f"{positive_pct:.1f}% of comments are positive",
        "Recommendation": (
            "Examine positive comments to identify strengths that could "
            "be highlighted in marketing campaigns."
        ),
        "Priority": "Medium"
    })

if neutral_count > 0:
    recommendations.append({
        "Finding": f"{neutral_pct:.1f}% of comments are neutral",
        "Recommendation": (
            "Review neutral comments for questions or information needs "
            "that clearer product descriptions or FAQs could address."
        ),
        "Priority": "Medium"
    })

best_model = results_df.iloc[0] if not results_df.empty else None

if best_model is not None:
    recommendations.append({
        "Finding": (
            f"{best_model['Model']} had the highest agreement "
            f"with human labels ({best_model['Accuracy (%)']:.2f}%)."
        ),
        "Recommendation": (
            "Consider it a candidate for further testing, but validate "
            "it on a larger, independent dataset before relying on it."
        ),
        "Priority": "Low"
    })

recommendations_df = pd.DataFrame(recommendations)

# 5. Save reports
results_df.to_csv(OUTPUT_DIR / "model_accuracy.csv", index=False)
df.to_csv(OUTPUT_DIR / "cleaned_sentiment_data.csv", index=False)
sentiment_report.to_csv(OUTPUT_DIR / "sentiment_report.csv", index=False)
word_report.to_csv(OUTPUT_DIR / "common_words.csv", index=False)
recommendations_df.to_csv(
    OUTPUT_DIR / "marketing_recommendations.csv", index=False
)

# 6. Chart: AI model agreement
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy (%)"])
plt.ylabel("Agreement with human labels (%)")
plt.title("AI Model vs Human Sentiment Agreement")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_accuracy.png", dpi=200)
plt.close()

# 7. Chart: sentiment distribution
plt.figure(figsize=(7, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.ylabel("Number of comments")
plt.title("Human-Labeled Sentiment Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=200)
plt.close()

# 8. Chart: most common words
if not word_report.empty:
    chart_words = word_report.sort_values("Frequency")
    plt.figure(figsize=(9, 6))
    plt.barh(chart_words["Word"], chart_words["Frequency"])
    plt.xlabel("Frequency")
    plt.title("Most Common Words in Comments")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "common_words.png", dpi=200)
    plt.close()

# 9. Print results
print("\nMARKETING INTELLIGENCE DETECTIVE")
print("=" * 38)
print("Comments analyzed:", total_comments)

print("\nHUMAN SENTIMENT DISTRIBUTION")
print(sentiment_report.to_string(index=False))

print("\nAI MODEL AGREEMENT")
print(results_df.to_string(index=False))

print("\nTOP WORDS IN COMMENTS")
if word_report.empty:
    print("No Comment column or usable comment text was found.")
else:
    print(word_report.to_string(index=False))

print("\nMARKETING RECOMMENDATIONS")
print(recommendations_df.to_string(index=False))

print("\nReports saved in:", OUTPUT_DIR)
print("Analysis completed successfully!")
