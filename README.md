# Capstone-Project: osrs-virtual-economy

This project analyzes the virtual economy of Old School RuneScape (OSRS) using historical Grand Exchange market data collected between:

February 1, 2025 – August 30, 2025

The provided parquet files within the DATA folder reproduce the exact dataset used in the final paper. The API data scrapper does not need to be run to view the project. Initial data collection exported to Excel format. During _rich data_ collection, the 6H time-series dataset exceeded Excel’s row limit (1,048,576 rows). The pipeline was refactored to use Parquet format, eliminating row constraints and improving I/O performance. This design change allowed full-scale dataset persistence without artificial partitioning.

**Objective:**

The goal of the project was to model price dynamics, detect structural changes, and identify clustering behavior across high-value in-game assets using time-series and unsupervised learning methods. The project contains the following:


API-based data scraper (OSRS Grand Exchange)

Processed datasets in Parquet format

Time-series feature engineering pipeline

PCA-based dimensionality reduction

K-Means clustering analysis

Change-point detection (ruptures)

Final submitted capstone paper (PDF)



**Methods:**

Principal Component Analysis (PCA)

K-Means Clustering

Silhouette Score Evaluation

Autocorrelation Function (ACF)

Change-Point Detection (ruptures)

Time-Series Visualization

Volatility and Return Engineering
