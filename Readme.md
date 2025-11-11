# 🎯 Vibe Matcher - AI Product Recommendation System

## 📊 Project Overview
A sophisticated recommendation engine that understands subjective customer "vibes" and matches them with relevant fashion products using semantic AI and cosine similarity.

## 🚀 Features
- **Vibe-Based Matching**: Goes beyond traditional filters to understand emotional preferences
- **Local Embeddings**: Uses sentence-transformers for cost-effective, private processing
- **Cosine Similarity**: Semantic matching using sklearn's cosine_similarity
- **Performance Analytics**: Comprehensive evaluation with metrics and visualization
- **Interactive Demo**: Real-time query testing
- **Edge Case Handling**: Fallback recommendations for poor matches

## 🛠️ Technical Implementation
- **Embeddings**: SentenceTransformer ('all-MiniLM-L6-v2')
- **Similarity**: Cosine Similarity (sklearn)
- **Visualization**: Matplotlib & Seaborn
- **Data**: 10 fashion products with detailed descriptions and vibe tags

## 📈 Performance Metrics
- Average Query Latency: < 0.1 seconds
- Good Match Rate (>0.4 similarity): 75%
- Scalable architecture ready for production

## 🎮 Quick Start
```bash
pip install -r requirements.txt
jupyter notebook vibe_matcher.ipynb