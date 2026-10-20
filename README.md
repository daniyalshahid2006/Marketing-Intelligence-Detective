# Pakistani Social Media Content Detective

## Project Overview

The Pakistani Social Media Content Detective is a data analysis project that investigates social-media engagement patterns and compares the performance of three Pakistani fashion brands: Khaadi, Sapphire, and Gul Ahmed.

The objective is to identify engagement trends, compare content performance, and develop data-driven recommendations for social-media marketing.

## Research Questions

- Which social-media platforms show the highest average engagement in the international dataset?
- How does engagement vary with sentiment, hashtags, and posting day?
- How do likes, comments, and saves compare across the three Pakistani fashion brands?
- Which brand has the highest average engagement relative to its follower count?
- What insights can inform future content-marketing decisions?

## Datasets

### 1. International Social-Media Dataset
File: `social_media_sentiment_raw.csv`

This dataset contains social-media posts from multiple platforms and countries. It is used to analyze platform performance, sentiment, hashtags, and posting-day patterns.

The dataset contains 2,200 unique posts after duplicate removal. No posts are labelled as originating from Pakistan, so these findings are treated as international results rather than evidence about Pakistani audiences.

### 2. Pakistani Fashion Brand Dataset
File: `pakistani_brand_posts_sample.csv`

This dataset contains 30 selected post records from three Pakistani fashion brands:

- Khaadi
- Sapphire
- Gul Ahmed

The dataset includes follower counts, likes, comments, saves, captions, and post identifiers. The figures were collected and verified for this analysis.

The findings describe the selected posts and should not automatically be generalized to every post published by these brands.

## Tools and Technologies

- Python
- Pandas
- Matplotlib
- CSV data processing
- Exploratory data analysis
- Data visualization

## Methodology

1. Load and inspect the datasets.
2. Remove duplicate records and clean relevant fields.
3. Standardize platform and sentiment values.
4. Calculate average engagement metrics.
5. Analyze engagement by platform, sentiment, hashtag, and posting day.
6. Compare average likes, comments, and saves across the three Pakistani fashion brands.
7. Calculate visible engagement and engagement per 1,000 followers.
8. Generate comparison tables, charts, and marketing recommendations.

For the Pakistani brand comparison, visible engagement is calculated as likes plus comments. Saves are analyzed separately. Engagement per 1,000 followers is calculated using visible engagement divided by follower count, multiplied by 1,000.

## Key Findings

### International Dataset

- TikTok recorded the highest average engagement at 60.50.
- Positive-sentiment posts had the highest average engagement at 53.81.
- The hashtag `#ai` recorded the highest average engagement among the analyzed hashtags, at 49.38.
- Sunday recorded the highest average engagement by posting day, at 48.51.

These results reflect the international dataset and do not establish equivalent trends among Pakistani audiences.

### Pakistani Fashion Brand Comparison

Based on the 30 selected posts:

- **Khaadi** recorded the highest average likes, at approximately 6,430 per post.
- **Sapphire** recorded the highest average saves, at approximately 318 per post.
- **Gul Ahmed** recorded the highest average visible engagement per 1,000 followers in this sample, at approximately 1.99.

These findings provide a comparison of the selected posts, not a definitive ranking of the brands' overall social-media performance.

## Generated Outputs

The analysis scripts generate files in the `outputs` directory, including:

- Pakistani brand comparison table
- Top-performing post report
- Brand average-likes chart
- Engagement-rate comparison chart
- Text summary of Pakistani brand findings
- International dataset reports and visualizations

## How to Run the Project

1. Install the required libraries:

   `pip install pandas matplotlib`

2. Place the CSV datasets in the same directory as the Python scripts.
3. Run `main.py` to analyze the international dataset.
4. Run `analyze_pakistani_brands.py` to compare the three Pakistani fashion brands.
5. Review the generated reports and charts in the `outputs` directory.

## Limitations

- The international dataset contains no Pakistan-labelled posts.
- The Pakistani brand analysis covers 30 selected posts, with 10 posts per brand.
- Engagement can vary with post format, publication date, audience size, and other factors.
- The results describe the selected posts and do not prove that the observed differences apply to all content published by each brand.
- The analysis identifies associations and patterns; it does not establish causation.

## Conclusion

This project demonstrates how Python and data visualization can be used to investigate social-media engagement and compare brand performance. Combining international platform analysis with selected Pakistani fashion-brand posts provides a practical foundation for evidence-based content-marketing recommendations.

Further research using a larger collection of verified posts across different dates and content formats could provide a more comprehensive understanding of Pakistani fashion-brand engagement.
