from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "pakistani_brand_posts_sample.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = {"brand", "platform", "followers", "post_id", "caption", "likes", "comments", "saves"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing columns: {sorted(missing)}")

for col in ["followers", "likes", "comments", "saves"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["brand", "followers", "likes", "comments"])
df["visible_engagement"] = df["likes"] + df["comments"]
df["engagement_per_1000_followers"] = df["visible_engagement"] / df["followers"] * 1000

summary = df.groupby("brand", as_index=False).agg(
    posts=("post_id", "count"),
    followers=("followers", "first"),
    avg_likes=("likes", "mean"),
    avg_comments=("comments", "mean"),
    avg_saves=("saves", "mean"),
    avg_visible_engagement=("visible_engagement", "mean"),
    avg_engagement_per_1000_followers=("engagement_per_1000_followers", "mean"),
)
summary = summary.sort_values("avg_engagement_per_1000_followers", ascending=False)
summary.to_csv(OUTPUT_DIR / "pakistani_brand_comparison.csv", index=False)

df.sort_values("visible_engagement", ascending=False).to_csv(
    OUTPUT_DIR / "top_pakistani_posts.csv", index=False
)

plt.figure(figsize=(8, 5))
plt.bar(summary["brand"], summary["avg_likes"])
plt.title("Average Likes by Pakistani Fashion Brand")
plt.xlabel("Brand")
plt.ylabel("Average likes per post")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pakistani_brand_average_likes.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(summary["brand"], summary["avg_engagement_per_1000_followers"])
plt.title("Average Visible Engagement per 1,000 Followers")
plt.xlabel("Brand")
plt.ylabel("Likes + comments per 1,000 followers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pakistani_brand_engagement_rate.png", dpi=200)
plt.close()

top_brand = summary.iloc[0]["brand"]
top_likes_brand = summary.sort_values("avg_likes", ascending=False).iloc[0]["brand"]
report = [
    "PAKISTANI FASHION BRAND INSTAGRAM ANALYSIS",
    "",
    f"Posts analyzed: {len(df)}",
    f"Brands analyzed: {df['brand'].nunique()}",
    "",
    "Average metrics by brand:",
    summary.round(2).to_string(index=False),
    "",
    f"Highest average engagement per 1,000 followers in this sample: {top_brand}.",
    f"Highest average likes per post in this sample: {top_likes_brand}.",
    "",
    "Limitations:",
    "- Results describe only the posts included in this CSV; they may not represent all posts or all Pakistani audiences.",
    "- Likes + comments are used as visible engagement. Saves are reported separately because they may not be publicly visible.",
    "- If any records are illustrative/sample data rather than collected observations, label the dataset as sample data and do not present the results as real-world findings."
]
(OUTPUT_DIR / "pakistani_brand_report.txt").write_text("\n".join(report), encoding="utf-8")

print("Analysis complete.")
print(f"Posts analyzed: {len(df)}")
print(f"Brands analyzed: {df['brand'].nunique()}")
print("\nBrand comparison:")
print(summary.round(2).to_string(index=False))
print(f"\nOutputs saved in: {OUTPUT_DIR}")
print("Files: pakistani_brand_comparison.csv, top_pakistani_posts.csv, two PNG charts, pakistani_brand_report.txt")
