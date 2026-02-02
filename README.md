# Capstone-Project: osrs-virtual-economy
OSRS Virtual Economy Modeling
Master’s Capstone Project – Data Science

This project analyzes the virtual economy of Old School RuneScape (OSRS) using historical Grand Exchange market data collected between:

February 1, 2025 – August 30, 2025

Objectives:
The goal of the project was to model price dynamics, detect structural changes, and identify clustering behavior across high-value in-game assets using time-series and unsupervised learning methods.

This repository contains:
-API-based data scraper (OSRS Grand Exchange)
-Processed datasets in Parquet format
-Time-series feature engineering pipeline
-PCA-based dimensionality reduction
-K-Means clustering analysis
=Change-point detection (ruptures)
-Final submitted capstone paper (PDF)

*The included Parquet files reproduce the exact dataset used in the final paper to ensure methodological consistency.

Methods:

-Principal Component Analysis (PCA)
-K-Means Clustering
-Silhouette Score Evaluation
-Autocorrelation Function (ACF)
-Change-Point Detection (ruptures)
-Time-Series Visualization
-Volatility and Return Engineering

Main Requirements:

Python version 3.10.19 Enviornment 
Tensorflow 3.12.1 

numpy==1.24.3
pandas==2.3.3
matplotlib==3.10.7
seaborn==0.13.2
scikit-learn==1.7.2
statsmodels==0.14.5
ruptures==1.1.10

For replication purposes, please utilize the provided parquet data files to match the data shown in the PDF document. API scraper is executable but not encourged to run--long processing time to collect data. 
