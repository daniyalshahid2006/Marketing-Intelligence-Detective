
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import re

# Project paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "social_media_sentiment_raw.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place social_media_sentiment_raw.csv beside main.py."
    )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

# Check required columns
required = [
    "post_id", "timestamp", "platform", "text", "hashtags",
    "country", "likes", "retweets_shares", "comments", "sentiment"
]

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean data
starting_rows = len(df)

df = df.drop_duplicates(subset=["post_id"]).copy()

for column in ["likes", "retweets_shares", "comments"]:
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    df[column] = df[column].clip(lower=0)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Standardize platform names
df["platform"] = (
    df["platform"]
    .astype("string")
    .str.strip()
    .str.title()
)

# Standardize countries and sentiment
df["country"] = df["country"].astype("string").str.strip()

df["sentiment"] = (
    df["sentiment"]
    .astype("string")
    .str.strip()
    .str.upper()
)

sentiment_mapping = {
    "POS": "POSITIVE",
    "POSTIVE": "POSITIVE",
    "NEG": "NEGATIVE",
    "NEUT": "NEUTRAL"
}

df["sentiment"] = df["sentiment"].replace(sentiment_mapping)

valid_sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Calculate engagement
df["engagement"] = (
    df["likes"] + df["retweets_shares"] + df["comments"]
)

df["day_of_week"] = df["timestamp"].dt.day_name()
df["hour"] = df["timestamp"].dt.hour

# Identify Pakistan posts
pakistan = df[
    df["country"].str.contains(
        r"pakistan|^pk$", case=False, na=False, regex=True
    )
].copy()

# 1. Dataset overview
print("\nPAKISTANI SOCIAL MEDIA CONTENT DETECTIVE")
print("=" * 45)
print("Original rows:", starting_rows)
print("Unique posts analyzed:", len(df))
print("Duplicate rows removed:", starting_rows - len(df))
print("Platforms:", df["platform"].nunique())
print("Countries:", df["country"].nunique())
print("Posts from Pakistan:", len(pakistan))
print("Posts with missing timestamps:", df["timestamp"].isna().sum())
print("Posts with missing sentiment:", df["sentiment"].isna().sum())

# 2. Platform performance
platform_report = (
    df.dropna(subset=["platform"])
    .groupby("platform")
    .agg(
        Posts=("post_id", "count"),
        Average_Likes=("likes", "mean"),
        Average_Shares=("retweets_shares", "mean"),
        Average_Comments=("comments", "mean"),
        Average_Engagement=("engagement", "mean"),
        Total_Engagement=("engagement", "sum")
    )
    .round(2)
    .sort_values("Average_Engagement", ascending=False)
)

platform_report.to_csv(OUTPUT_DIR / "platform_performance.csv")

# 3. Sentiment and engagement
# Missing or unrecognized sentiment labels are excluded only here.
sentiment_df = df[df["sentiment"].isin(valid_sentiments)].copy()

sentiment_report = (
    sentiment_df.groupby("sentiment")
    .agg(
        Posts=("post_id", "count"),
        Average_Likes=("likes", "mean"),
        Average_Shares=("retweets_shares", "mean"),
        Average_Comments=("comments", "mean"),
        Average_Engagement=("engagement", "mean")
    )
    .round(2)
    .sort_values("Average_Engagement", ascending=False)
)

sentiment_report.to_csv(OUTPUT_DIR / "sentiment_engagement.csv")

# 4. Country performance
country_report = (
    df.dropna(subset=["country"])
    .groupby("country")
    .agg(
        Posts=("post_id", "count"),
        Average_Engagement=("engagement", "mean"),
        Total_Engagement=("engagement", "sum")
    )
    .round(2)
    .sort_values("Average_Engagement", ascending=False)
)

country_report.to_csv(OUTPUT_DIR / "country_performance.csv")

# 5. Hashtag analysis
# Extract hashtags from both "#tag" and plain "tag" formats.
hashtag_posts = {}

for _, row in df.iterrows():
    raw_hashtags = row["hashtags"]

    if pd.isna(raw_hashtags):
        continue

    tags = re.findall(r"#?([A-Za-z0-9_]+)", str(raw_hashtags))

    # Count each hashtag at most once per post.
    for tag in set(tag.lower() for tag in tags):
        if tag:
            hashtag_posts.setdefault(tag, []).append(row["engagement"])

hashtag_rows = []

for tag, engagements in hashtag_posts.items():
    hashtag_rows.append({
        "Hashtag": tag,
        "Post_Count": len(engagements),
        "Average_Engagement": round(
            sum(engagements) / len(engagements), 2
        )
    })

hashtag_report = pd.DataFrame(
    hashtag_rows,
    columns=["Hashtag", "Post_Count", "Average_Engagement"]
)

if not hashtag_report.empty:
    hashtag_report = hashtag_report.sort_values(
        ["Average_Engagement", "Post_Count"],
        ascending=[False, False]
    )

hashtag_report.to_csv(
    OUTPUT_DIR / "hashtag_performance.csv", index=False
)

# 6. Top-performing posts
top_columns = [
    "post_id", "platform", "country", "text", "hashtags",
    "sentiment", "likes", "retweets_shares", "comments", "engagement"
]

top_posts = df.nlargest(20, "engagement")[top_columns]
top_posts.to_csv(OUTPUT_DIR / "top_posts.csv", index=False)

# 7. Posting day and hour patterns
timed_df = df.dropna(subset=["timestamp"]).copy()

day_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

day_report = (
    timed_df.groupby("day_of_week")
    .agg(
        Posts=("post_id", "count"),
        Average_Engagement=("engagement", "mean")
    )
    .reindex(day_order)
    .dropna(subset=["Posts"])
    .round(2)
)

hour_report = (
    timed_df.groupby("hour")
    .agg(
        Posts=("post_id", "count"),
        Average_Engagement=("engagement", "mean")
    )
    .round(2)
)

day_report.to_csv(OUTPUT_DIR / "day_performance.csv")
hour_report.to_csv(OUTPUT_DIR / "hour_performance.csv")

# 8. Generate marketing recommendations
recommendations = []

if not platform_report.empty:
    best_platform = platform_report.index[0]
    best_value = platform_report.iloc[0]["Average_Engagement"]
    platform_posts = int(platform_report.iloc[0]["Posts"])

    recommendations.append({
        "Finding": f"{best_platform} has the highest average engagement",
        "Evidence": (
            f"Average engagement: {best_value:.2f}; "
            f"posts analyzed: {platform_posts}"
        ),
        "Recommendation": (
            "Test additional content on this platform. Compare similar "
            "content and audience sizes before increasing investment."
        ),
        "Priority": "High"
    })

if not sentiment_report.empty:
    best_sentiment = sentiment_report.index[0]
    best_value = sentiment_report.iloc[0]["Average_Engagement"]

    recommendations.append({
        "Finding": f"{best_sentiment} posts have the highest average engagement",
        "Evidence": (
            f"Average engagement: {best_value:.2f}; "
            f"posts analyzed: {int(sentiment_report.iloc[0]['Posts'])}"
        ),
        "Recommendation": (
            "Review examples in this sentiment category and test similar "
            "messaging. This association does not prove sentiment causes engagement."
        ),
        "Priority": "Medium"
    })

if not hashtag_report.empty:
    best_tag = hashtag_report.iloc[0]

    recommendations.append({
        "Finding": f"#{best_tag['Hashtag']} has the highest average engagement",
        "Evidence": (
            f"Posts: {int(best_tag['Post_Count'])}; "
            f"average engagement: {best_tag['Average_Engagement']:.2f}"
        ),
        "Recommendation": (
            "Review the posts using this hashtag and test it only where "
            "it is relevant to the brand and audience."
        ),
        "Priority": "Medium"
    })

if not day_report.empty:
    best_day = day_report["Average_Engagement"].idxmax()
    best_value = day_report.loc[best_day, "Average_Engagement"]
    day_posts = int(day_report.loc[best_day, "Posts"])

    recommendations.append({
        "Finding": f"{best_day} has the highest average engagement",
        "Evidence": (
            f"Average engagement: {best_value:.2f}; posts: {day_posts}"
        ),
        "Recommendation": (
            "Run a posting-time experiment across several weeks before "
            "treating this day as the best time to publish."
        ),
        "Priority": "Medium"
    })

if not hour_report.empty:
    best_hour = hour_report["Average_Engagement"].idxmax()
    best_value = hour_report.loc[best_hour, "Average_Engagement"]

    recommendations.append({
        "Finding": f"Hour {best_hour:02d}:00 has the highest average engagement",
        "Evidence": f"Average engagement: {best_value:.2f}",
        "Recommendation": (
            "Validate this time using the target audience's local time zone "
            "and test it against alternative posting times."
        ),
        "Priority": "Low"
    })

if len(pakistan) == 0:
    recommendations.append({
        "Finding": "No posts were labelled as being from Pakistan",
        "Evidence": "No matching country entries were found in the dataset",
        "Recommendation": (
            "Present the current findings as international dataset results. "
            "Obtain Pakistan-specific data before making claims about Pakistani audiences."
        ),
        "Priority": "High"
    })
else:
    recommendations.append({
        "Finding": f"{len(pakistan)} posts are labelled as from Pakistan",
        "Evidence": (
            f"{len(pakistan) / len(df) * 100:.1f}% of all analyzed posts"
            if len(df) else "No posts"
        ),
        "Recommendation": (
            "Analyze Pakistani posts separately and disclose the sample size "
            "and any limitations."
        ),
        "Priority": "High"
    })

recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv(
    OUTPUT_DIR / "marketing_recommendations.csv", index=False
)

# 9. Generate charts
if not platform_report.empty:
    platform_report["Average_Engagement"].plot(kind="bar")
    plt.title("Average Engagement by Platform")
    plt.xlabel("Platform")
    plt.ylabel("Average likes + shares + comments")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "platform_engagement.png", dpi=200)
    plt.close()

if not sentiment_report.empty:
    sentiment_report["Average_Engagement"].plot(kind="bar")
    plt.title("Average Engagement by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average engagement")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_engagement.png", dpi=200)
    plt.close()

if not hashtag_report.empty:
    chart_data = hashtag_report.head(10).sort_values(
        "Average_Engagement"
    )

    chart_data.plot(
        x="Hashtag",
        y="Average_Engagement",
        kind="barh",
        legend=False
    )
    plt.title("Top Hashtags by Average Engagement")
    plt.xlabel("Average engagement")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hashtag_engagement.png", dpi=200)
    plt.close()

if not day_report.empty:
    day_report["Average_Engagement"].plot(kind="bar")
    plt.title("Average Engagement by Day")
    plt.xlabel("Day of week")
    plt.ylabel("Average engagement")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "day_engagement.png", dpi=200)
    plt.close()

# 10. Save an overall summary
summary = pd.DataFrame([{
    "Posts_Analyzed": len(df),
    "Platforms": df["platform"].nunique(),
    "Countries": df["country"].nunique(),
    "Pakistan_Posts": len(pakistan),
    "Valid_Sentiment_Posts": len(sentiment_df),
    "Missing_Timestamps": int(df["timestamp"].isna().sum())
}])

summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

# 11. Print results
print("\nPLATFORM PERFORMANCE")
print(platform_report.to_string())

print("\nSENTIMENT AND ENGAGEMENT")
print(sentiment_report.to_string())

print("\nTOP 10 HASHTAGS BY AVERAGE ENGAGEMENT")
if hashtag_report.empty:
    print("No usable hashtags were found.")
else:
    print(hashtag_report.head(10).to_string(index=False))

print("\nDAY PERFORMANCE")
print(day_report.to_string())

print("\nMARKETING RECOMMENDATIONS")
print(recommendations_df.to_string(index=False))

print("\nReports and charts saved to:", OUTPUT_DIR)
print("Analysis completed successfully!")
