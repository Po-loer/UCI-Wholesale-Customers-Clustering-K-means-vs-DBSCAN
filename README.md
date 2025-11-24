# UCI-Wholesale-Customers-Clustering-K-means-vs-DBSCAN
This repository contains a Jupyter notebook implementing clustering analysis on the UCI Wholesale Customers Dataset using K-means and DBSCAN algorithms. The project addresses the Group 2 assignment tasks, including data preprocessing, exploratory data analysis (EDA), feature engineering, clustering modeling, parameter selection, statistical inference, and visualization/reflection.
Dataset

- Source: UCI Wholesale Customers Dataset (available from UCI ML Repository or Kaggle).
- Features: Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen.
- Size: 440 samples.
- Included in Repo: Wholesale customers data.csv (download from links above and place in the root if not already present).

Key Findings

1. Preprocessing: No missing values; no duplicates removed (shape unchanged at 440x8). RobustScaler used for scaling due to outliers.
2. EDA: High skewness in spending categories (e.g., Fresh, Frozen); log-transformation reduces it. Strong correlations (e.g., Grocery-Detergents_Paper ~0.92) suggest co-purchasing.
3. Feature Engineering: Added TotalSpend (sum of categories) and ProportionFresh (Fresh/TotalSpend) for better cluster interpretation.
4. Clustering:
    -K-means: Optimal k=3 (highest Silhouette ~0.35, lowest DB index ~1.12). Clusters: 134, 174, 132 sizes.
    -DBSCAN: Optimal eps=1.2 (from k-distance plot), min_samples=5. Clusters: 3 (sizes 8, 7, 373), 52 noise points (~12%).
    -Comparison: K-means produces balanced clusters; DBSCAN identifies more noise and fewer/smaller clusters.

5. Inference: Significant differences in TotalSpend across clusters (Kruskal-Wallis p<0.001). Channel/Region associated with clusters (Chi-square p<0.001).
6. Visualization: PCA shows compact K-means clusters vs. DBSCAN's dense core with noise.
7. Reflection: K-means assumes spherical shapes (may miss outliers); DBSCAN handles arbitrary shapes but sensitive to eps/min_samples. Recommendations: Target high-TotalSpend clusters (e.g., Cluster 0) with promotions on correlated items like Grocery/Detergents.

Requirements

1. Python 3.8+
2. Libraries:
    -pandas, numpy, matplotlib, seaborn (for data handling and viz)
    -scikit-learn (for scaling, clustering, PCA, metrics)
    -scipy (for statistical tests)


Install via:
textpip install pandas numpy matplotlib seaborn scikit-learn scipy

Usage

1. Clone the repository:
  textgit clone https://github.com/yourusername/uci-wholesale-customers-clustering.git
  cd uci-wholesale-customers-clustering
2. Download the dataset (Wholesale customers data.csv) from UCI/Kaggle and place it in the root directory (update the file path in the notebook if needed).
3. Open the notebook:textjupyter notebook             Group_Two_Wholesale_Customers_Clustering.ipynb
  -Run cells sequentially for preprocessing, EDA, clustering, etc.
  -For reproducibility, use the provided parameters (e.g., random_state=42).


Notebook Structure

Part A: Data Loading & Preprocessing: Load CSV, handle missing/duplicates, scale with RobustScaler.
Part B: First EDA: Stats, boxplots (outliers: e.g., Fresh has 3>50k), log-transform histograms, correlation heatmap.
Part C: Feature Engineering: Add TotalSpend and ProportionFresh; justify for spend patterns.
Part D: Clustering & Parameter Selection: K-means elbow/Silhouette/DB plots; DBSCAN k-distance for eps; comparison table.
Part E: Second EDA & Inference: Centroids table, Kruskal-Wallis/Chi-square tests.
Part F: Visualization & Reflection: PCA scatterplots, discussion on assumptions/sensitivity, domain recommendations.

Results

-Optimal Models:
  -K-means (k=3): Silhouette=0.35, DB=1.12.
  -DBSCAN (eps=1.2, min_samples=5): 3 clusters + 12% noise.

-Statistical Tests:
  -TotalSpend differences: p=1.2e-50 (K-means), p=4.3e-12 (DBSCAN).
  -Channel association: p=1.4e-34 (K-means), p=0.002 (DBSCAN).

-Recommendations: Segment customers by spend (e.g., high Grocery/Detergents cluster for retail promotions; noise points as outliers for further investigation).

Contributing
Fork and submit pull requests for improvements, such as additional clustering algorithms (e.g., Hierarchical) or hyperparameter tuning.

License
MIT License - Free to use and modify.

Acknowledgments
-Dataset from UCI Machine Learning Repository.
-Inspired by clustering techniques for customer segmentation in retail.
