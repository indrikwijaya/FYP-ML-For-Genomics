# FYP-ML-For-Genomics
This repo contains relevant files for my final year project in NUS (2018), 'Machine Learning For Genomics'.

*Abstract*

This report will explore the performance of different unsupervised learning algorithms, particularly on clustering, on short time-series data from gene expression values. 
Many biological data are in the form of short time-series, yet there are not many studies done on this area. 
Many standard machine learning algorithms normally work well on longer time-series. 
These algorithms tend to fail to separate different short time-series data into meaningful clusters as the data are not long enough to develop distinct and clear patterns. 
As such, data that are not supposed to be clustered together, may be clustered together. 
In this report, we will explore a few algorithms: Short Time-series Expression Miner (STEM), Gaussian Mixture Model, K-means and Hierarchical Clustering. 
STEM was specifically developed to address the problem of clustering short time-series data, 
whereas the other three algorithms are the standard machine learning algorithms that are still widely used to cluster time-series data.

[STEM](https://link.springer.com/article/10.1186/1471-2105-7-191) is a widely-cited algorithm for short time-series data.

**Datasets**
1. Unlabelled gene expression values
2. Short time-series, 7 time points
3. N ~ 10k data points, 2 sets: Bulk mitochondria and Crude mitochondria
4. Special subset: Mitochondria genes (~ 1.1k data points)

**STEM as benchmark**
1. The only algorithm that gives optimal number of clusters using statistical test
2. Takes into account the sequential nature of time-series
3. Generates profiles independent of data

**Findings**
1. STEM may exclude many genes (or remove noises)
2. Algorithms with using correlation coefficient as distance measure perform better 
3. Use STEM to initialize number of clusters
4. Euclidean distance is not a good distance measure

