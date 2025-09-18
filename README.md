# amazon-recommendation-system

A Matrix Factorization-based recommendation system implemented using Bayesian Personalized Ranking (BPR) for implicit feedback datasets.
This project demonstrates end-to-end data preprocessing, model training, and evaluation on the Amazon Electronics dataset (2023 subset).

📌 Features:

  Data Preprocessing
    Mapping raw Amazon product/user IDs to integer indices
    Cleaning interactions & handling duplicates
    Train/valid/test dataset split
    
  Model
    Bayesian Personalized Ranking – Matrix Factorization (BPR-MF)
    Optimized with stochastic gradient descent (SGD)
    Supports negative sampling for implicit feedback
    
  Training & Evaluation
    Training loop with epoch logging
    Evaluation metrics:
      Recall@K
      Precision@K
      NDCG@K (Normalized Discounted Cumulative Gain)
    Reproducible results with fixed random seeds
    
  Modular Codebase
    src/data/ → dataset loaders
    src/models/ → BPR-MF implementation
    src/training/ → training & evaluation scripts
    scripts/ → data preprocessing helpers
