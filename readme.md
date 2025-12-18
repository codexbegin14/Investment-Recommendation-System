# Hybrid Investment Recommendation System

An AI-powered investment recommendation system that generates personalized and diversified asset portfolios using a hybrid recommender approach. The system combines user behavior, asset characteristics, and demographic risk profiles to support informed investment decisions.

---

## Key Features

Hybrid Recommendation Engine
• Collaborative Filtering using Mean-Centered SVD
• Content-Based Filtering using Cosine Similarity on asset features
• Demographic Risk Matching based on user and asset risk alignment

Adaptive Weighting
• Automatically adjusts model importance based on user interaction history
• Handles cold-start and sparse data scenarios effectively

Portfolio Diversification
• Sector-aware post-processing to prevent over-concentration

Dynamic User Profiling
• Interactive questionnaire for updating risk tolerance and investment capacity

Interactive Dashboard
• Displays ROI metrics, profitability analysis, and sector allocation visualizations

---

## Technology Stack

Frontend: Streamlit
Backend: Python
Machine Learning: Scikit-Learn
Visualization: Plotly
Data Processing: Pandas, NumPy

---

## Installation

```bash
git clone https://github.com/your-username/hybrid-investment-system.git
cd hybrid-investment-system
pip install -r requirements.txt
streamlit run app.py
```


## Evaluation Metrics

Recall@K
nDCG@K
Average ROI@K
Sector diversification checks

---
