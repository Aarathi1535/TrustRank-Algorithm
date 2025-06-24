import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# App Title
st.title("🧠 TrustRank – Review Scoring Algorithm with Evaluation")

# Sample Review Data
reviews = pd.DataFrame({
    'review_id': list(range(1, 21)),
    'product_id': [101, 101, 102, 102, 103, 103, 104, 104, 105, 105,
                   106, 106, 107, 107, 108, 108, 109, 109, 110, 110],
    'rating': [5, 4, 3, 2, 5, 5, 4, 1, 3, 2,
               4, 5, 2, 3, 4, 4, 5, 1, 3, 5],
    'review_text': [
        "Great product, loved the quality.", "Worked well, but battery drains fast.",
        "Average experience, not as expected.", "Poor build, broke easily.",
        "Perfect for my needs.", "Amazing features and usability.",
        "Good value for money.", "Not worth the price.",
        "Fair quality, okay for casual use.", "Too small and weak.",
        "Works flawlessly, very happy.", "Excellent design and feel.",
        "Doesn’t match the listing.", "Satisfactory performance.",
        "Quite comfortable and durable.", "Stylish and functional.",
        "Highly recommend it.", "Terrible purchase.",
        "Meets basic requirements.", "Exceeded expectations."
    ],
    'helpful_votes': [9, 3, 4, 0, 8, 10, 5, 1, 3, 0, 6, 7, 0, 2, 5, 6, 9, 0, 4, 10],
    'total_votes': [10, 5, 6, 2, 8, 12, 6, 3, 5, 1, 7, 7, 1, 3, 6, 8, 10, 1, 6, 11],
    'review_date': pd.to_datetime([
        '2024-06-01', '2024-05-15', '2024-04-20', '2024-04-01', '2024-03-15',
        '2024-03-01', '2024-02-20', '2024-02-01', '2024-01-15', '2024-01-01',
        '2023-12-15', '2023-12-01', '2023-11-20', '2023-11-01', '2023-10-15',
        '2023-10-01', '2023-09-15', '2023-09-01', '2023-08-15', '2023-08-01']
    )
})

st.subheader("📦 Sample Dataset")
st.dataframe(reviews)

# TrustRank Scoring
analyzer = SentimentIntensityAnalyzer()
current_date = datetime.strptime("2025-06-23", "%Y-%m-%d")

def trust_rank(row):
    sentiment = analyzer.polarity_scores(row['review_text'])['compound']
    helpfulness = row['helpful_votes'] / row['total_votes'] if row['total_votes'] > 0 else 0
    days_old = (current_date - row['review_date']).days
    time_decay = np.exp(-days_old / 365)
    return row['rating'] * 0.4 + sentiment * 5 * 0.2 + helpfulness * 5 * 0.3 + time_decay * 5 * 0.1

reviews['trust_rank'] = reviews.apply(trust_rank, axis=1)

product_scores = reviews.groupby('product_id').agg(
    avg_rating=('rating', 'mean'),
    trust_rank_score=('trust_rank', 'mean')
).reset_index()

st.subheader("📊 Product Score Comparison")
st.dataframe(product_scores)

# Rank Correlation
spearman_corr, _ = spearmanr(product_scores['avg_rating'], product_scores['trust_rank_score'])
st.markdown(f"**🔁 Spearman Rank Correlation**: {spearman_corr:.4f}")

# Top-K Overlap
top_avg = set(product_scores.sort_values('avg_rating', ascending=False).head(3)['product_id'])
top_trust = set(product_scores.sort_values('trust_rank_score', ascending=False).head(3)['product_id'])
overlap = len(top_avg.intersection(top_trust))
st.markdown(f"**✅ Top-3 Overlap**: {overlap}/3 products match between Avg Rating and TrustRank")

# Proof: Disagreement
product_scores['rank_diff'] = product_scores['trust_rank_score'].rank(ascending=False) - product_scores['avg_rating'].rank(ascending=False)
st.subheader("🚨 Rank Disagreement (TrustRank vs Avg Rating)")
st.dataframe(product_scores[product_scores['rank_diff'].abs() >= 3])

# Plot: Rank Difference
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.barplot(x='product_id', y='rank_diff', data=product_scores, palette='coolwarm', ax=ax1)
ax1.axhline(0, color='gray', linestyle='--')
ax1.set_title('Rank Difference: TrustRank vs Avg Rating')
ax1.set_ylabel('Rank Difference')
ax1.set_xlabel('Product ID')
st.pyplot(fig1)

# Plot: Bar Chart
fig2, ax2 = plt.subplots(figsize=(10, 5))
index = np.arange(len(product_scores))
bar_width = 0.35
ax2.bar(index, product_scores['avg_rating'], bar_width, label='Avg Rating')
ax2.bar(index + bar_width, product_scores['trust_rank_score'], bar_width, label='TrustRank')
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(product_scores['product_id'])
ax2.legend()
ax2.set_title("TrustRank vs Avg Rating")
st.pyplot(fig2)

# Plot: Scatter
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.scatterplot(x='avg_rating', y='trust_rank_score', data=product_scores, ax=ax3)
ax3.set_title("Scatter: Avg Rating vs TrustRank")
st.pyplot(fig3)

# Plot: Boxplot
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.boxplot(data=product_scores[['avg_rating', 'trust_rank_score']], ax=ax4)
ax4.set_title("Boxplot Comparison")
st.pyplot(fig4)

# Final Table
st.subheader("🏁 Final Rankings by TrustRank")
st.dataframe(product_scores.sort_values(by='trust_rank_score', ascending=False))
