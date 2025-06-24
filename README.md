# 🧠 TrustRank – A Smarter Review Scoring Algorithm for E-Commerce

### 📍 Author: Aarathisree Balla  
🚧 Publication-in-progress: Targeting submission to IJRTE / IJERT

---

## 🚀 Project Overview

Online reviews often influence product choices but suffer from fake ratings, outdated opinions, and unhelpful content. **TrustRank** is a custom-designed algorithm that provides a more trustworthy and insightful ranking mechanism by combining:

- ⭐ Rating
- 💬 Review Sentiment (via NLP)
- 👍 Helpfulness votes
- ⏳ Recency (Time Decay)

📊 Compared to simple average ratings, **TrustRank produces rankings that are more aligned with actual product quality and user trust**.

---

## 🔍 Why TrustRank Over Average Ratings?

| Feature                 | Avg Rating | TrustRank |
|------------------------|------------|-----------|
| Considers sentiment    | ❌ No      | ✅ Yes     |
| Weighs helpful votes   | ❌ No      | ✅ Yes     |
| Penalizes outdated reviews | ❌ No  | ✅ Yes     |
| Detects fake or biased trends | ❌ No | ✅ Yes  |

---

## 💡 Key Features

- 🧮 Computes TrustRank using:
  - Sentiment polarity from review text (VADER)
  - Helpfulness ratio
  - Time decay factor
- 📊 Evaluation Metrics:
  - **Spearman rank correlation**
  - **Top-k ranking overlap**
  - **Standard deviation comparison**
- 📉 Visual Proof:
  - Bar chart (TrustRank vs Avg Rating)
  - Scatter plot
  - Rank difference plot
  - Boxplot analysis
- 🧠 CLI and Streamlit versions available

---

## 🔬 Formula

```python
TrustRank = 
    0.4 * Rating +
    0.2 * Sentiment +
    0.3 * Helpfulness Ratio +
    0.1 * Time Decay
