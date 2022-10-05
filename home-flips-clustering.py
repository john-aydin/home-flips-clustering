**Welcome to my experiment on Single-Family Home Flips by Census Tract!**

In this exploration, we will be sorting the information provided by the data into clusters and evaluating those clusters' accuracy based on similarity of data points within each cluster and distance between neighboring data points. Background information on the home flips data is provided below, including a link that may be helpful to reference for understanding of the individual facets of the data being analyzed.

**Our Question Today:** How many different groups (or, in a sense, "tiers" of home flipping) can the data be effectively divided into?

Dataset: **Single-Family Home Flips by Census Tract**

"Displacement risk indicator showing the number of property transactions of single-family homes recorded by the King County Assessor that can be classified as "flips" (meaning that the home had previously been sold within the past year and that the sale price had increased between sales at least twice as fast as the increase in regional housing Consumer Price Index during the same time period). Summarized at the census tract level; available for every year from 2004 through the most recent year of available data."

[*SOURCE*](https://catalog.data.gov/dataset/single-family-home-flips-by-census-tract-948c1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time

!pip install umap-learn
import umap

from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

# Loading the dataset into the dataframe:
  # Single-Family Home Flips by Census Tract

df = pd.read_csv("https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::single-family-home-flips-by-census-tract.csv?outSR=%7B%22latestWkid%22%3A4326%2C%22wkid%22%3A4326%7D")

"""# **Exploratory Data Analysis:**"""

df.info()
df.head(20)

df.describe()

df.corr()

# Preprocessing the dataset to prepare it for dimensionality reduction and clustering:

X = df.drop(labels=['OBJECTID', 'YEAR', 'FLIPS'], axis=1) # data
y = df['FLIPS'] # target

# Replace missing values (marked by `?`) with a `0`
X = X.replace(to_replace='?', value=0)

# Binarize `y` so that `1` means yes and `0` means no
y = np.where(y > 0.95, 1, 0)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

"""# **Applying Dimensionality Reduction techniques to visualize the observations:**"""

# Defining a tool to assess speed:
time_start = time.time()

"""Principal Component Analysis (PCA):"""

# We just want the first two principal components
pca = PCA(n_components=2)

# We get the components by calling fit_transform method with our data
pca_components = pca.fit_transform(X)

print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.scatter(pca_components[:, 0], pca_components[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

# Let's try testing for more than 2 principal components when visualizing the clusters of data:

# 3 principal components:
pca = PCA(n_components=3)
pca_components = pca.fit_transform(X)

plt.figure(figsize=(10,5))
plt.title("PCA with 3 principal components")
plt.scatter(pca_components[:, 0], pca_components[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
print("------------------------------------------------")

# Greater than 3 principal components is out of range for PCA solver and leads to execution error

# PCA appears to group similar data points slightly more closely together with 3 principal components 
  # rather than with 2 principal components

"""t-Distributed Stochastic Neighbor Embedding (t-SNE):"""

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

# Testing a range of perplexity values for clustering visualization:

tsne = TSNE(n_components=3, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("t-SNE with perplexity=10")
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

tsne = TSNE(n_components=3, verbose=1, perplexity=20, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("t-SNE with perplexity=20")
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

tsne = TSNE(n_components=3, verbose=1, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("t-SNE with perplexity=30")
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("t-SNE with perplexity=40")
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

tsne = TSNE(n_components=3, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("t-SNE with perplexity=50")
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

tsne = TSNE(n_components=3, verbose=1, perplexity=60, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("t-SNE with perplexity=60")
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

# t-SNE appears to perform best with higher perplexity, 
  # such as when perplexity=60

"""Uniform Manifold Approximation and Projection (UMAP):"""

umap_results = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(X)

print('UMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
colours = ["r","b","g","c","m","y","k","r","burlywood","chartreuse"]
for i in range(umap_results.shape[0]):
    plt.text(umap_results[i, 0], umap_results[i, 1], y[i],
             color=colours[int(y[i])],
             fontdict={'weight': 'bold', 'size': 50}
        )

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

plt.figure(figsize=(10,5))
plt.scatter(umap_results[:, 0], umap_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

umap_results = umap.UMAP(n_neighbors=3,
                      min_dist=0.3,
                      metric='correlation').fit_transform(X)

print('UMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("Umap with n_neighbors=3 and min_dist=0.3")
plt.scatter(umap_results[:, 0], umap_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

umap_results = umap.UMAP(n_neighbors=3,
                      min_dist=1,
                      metric='correlation').fit_transform(X)

print('UMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("Umap with n_neighbors=3 and min_dist=1")
plt.scatter(umap_results[:, 0], umap_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

umap_results = umap.UMAP(n_neighbors=7,
                      min_dist=0.3,
                      metric='correlation').fit_transform(X)

print('UMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("Umap with n_neighbors=7 and min_dist=0.3")
plt.scatter(umap_results[:, 0], umap_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

umap_results = umap.UMAP(n_neighbors=7,
                      min_dist=1,
                      metric='correlation').fit_transform(X)

print('UMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(10,5))
plt.title("Umap with n_neighbors=7 and min_dist=1")
plt.scatter(umap_results[:, 0], umap_results[:, 1])
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

"""# **Applying Clustering Techniques to group similar observations:**"""

# Using %timeit to assess speed.

"""K-means:"""

# Commented out IPython magic to ensure Python compatibility.
# Run k-means by setting n_clusters=3, because we have three classes of comparison types in the data. 

kmeans_cluster = KMeans(n_clusters=3, random_state=123)

# Fit model
# %timeit kmeans_cluster.fit(X_std)
y_pred = kmeans_cluster.predict(X_std)

# Next, visualizing the predictions and the true labels of the observations:

pca = PCA(n_components=3).fit_transform(X_std)

plt.figure(figsize=(10,5))
colours = 'rbg'
for i in range(pca.shape[0]):
    plt.text(pca[i, 0], pca[i, 1], str(y_pred[i]),
             color=colours[y[i]],
             fontdict={'weight': 'bold', 'size': 50}
        )

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

# In the plot below, the numbers show the cluster that each observation has been assigned to by the algorithm. 
  # The colors denote the true classes.

# Commented out IPython magic to ensure Python compatibility.
# Defining the mini-batch k-means
minikmeans_cluster = MiniBatchKMeans(
    init='random',
    n_clusters=3,
    batch_size=50)

# Fit model
# %timeit minikmeans_cluster.fit(X_std)
minikmeans_cluster = minikmeans_cluster.predict(X_std)

plt.figure(figsize=(10,5))
colours = 'rbg'
for i in range(pca.shape[0]):
    plt.text(pca[i, 0], pca[i, 1], str(minikmeans_cluster[i]),
             color=colours[y[i]],
             fontdict={'weight': 'bold', 'size': 50}
        )

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

# Defining the k-means

cluster_numbers = [2, 3, 4, 5, 6]
kmeans_clusters = []
for i in cluster_numbers:
    k_means = KMeans(n_clusters=i, random_state=123)
    kmeans_clusters.append(k_means.fit_predict(X_std))

# Commented out IPython magic to ensure Python compatibility.
pca = PCA(n_components=2).fit_transform(X_std)

colours = 'rbg'
for i, solution in enumerate(kmeans_clusters):
#     %timeit kmeans_clusters 
    plt.figure(figsize=(10,5))
    plt.text(np.mean(pca[:,0]), np.max(pca[:, 1]), "K-means with k = {}".format(cluster_numbers[i]),
                 fontdict={'weight': 'bold', 'size': 50})
    for i in range(pca.shape[0]):
        plt.text(pca[i, 0], pca[i, 1], str(solution[i]),
                 color=colours[y[i]],
                 fontdict={'weight': 'bold', 'size': 50}
            )

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

# Reduce it to two components for visualization
X_pca = PCA(2).fit_transform(X_std)

# Calculate predicted values.
y_pred = KMeans(n_clusters=2, random_state=123).fit_predict(X_std)

# Plot the solution.
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.show()

# Check the solution against the data using cross-tabs/contingency tables.
print('Comparing k-means clusters against the data:')
print(pd.crosstab(y_pred, y))

# Evaluating performance metrics for the algorithm:
print("------------------------------------------------")
print("ARI score of k-means with k=2: {}".format(
    metrics.adjusted_rand_score(y, y_pred)))

print("Silhouette score of k-means with k=2: {}".format(
    metrics.silhouette_score(X_std, y_pred, metric='euclidean')))

"""Hierarchical:"""

plt.figure(figsize=(20,10))
plt.title("Dendrogram with linkage method: Complete")
dendrogram(linkage(X_std, method='complete'))
plt.show()

plt.figure(figsize=(20,10))
plt.title("Dendrogram with linkage method: Average")
dendrogram(linkage(X_std, method='average'))
plt.show()

plt.figure(figsize=(20,10))
plt.title("Dendrogram with linkage method: Ward")
dendrogram(linkage(X_std, method='ward'))
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# Defining the agglomerative clustering
agg_cluster = AgglomerativeClustering(linkage="average", 
                                      affinity='euclidean',
                                      n_clusters=3)

# Fit model
clusters = agg_cluster.fit_predict(X_std)
# %timeit clusters 

print("ARI score of linkage method average: {}".format(
    metrics.adjusted_rand_score(y, clusters)))

print("Silhouette score of linkage method average: {}".format(
    metrics.silhouette_score(X_std, clusters, metric='euclidean')))
print("------------------------------------------------")

# Defining the agglomerative clustering
agg_cluster = AgglomerativeClustering(linkage="complete", 
                                      affinity='euclidean',
                                      n_clusters=3)

# Fit model
clusters = agg_cluster.fit_predict(X_std)
# %timeit clusters 

print("ARI score of linkage method complete: {}".format(
    metrics.adjusted_rand_score(y, clusters)))

print("Silhouette score of linkage method complete: {}".format(
    metrics.silhouette_score(X_std, clusters, metric='euclidean')))
print("------------------------------------------------")

# Defining the agglomerative clustering
agg_cluster = AgglomerativeClustering(linkage="ward", 
                                      affinity='euclidean',
                                      n_clusters=3)

# Fit model
clusters = agg_cluster.fit_predict(X_std)
# %timeit clusters 

print("ARI score of linkage method ward: {}".format(
    metrics.adjusted_rand_score(y, clusters)))

print("Silhouette score of linkage method ward: {}".format(
    metrics.silhouette_score(X_std, clusters, metric='euclidean')))

"""DBSCAN:"""

# Commented out IPython magic to ensure Python compatibility.
# Defining the agglomerative clustering
dbscan_cluster = DBSCAN(eps=0.5, min_samples=1, metric="euclidean")

# Fit model
clusters = dbscan_cluster.fit_predict(X_std)
# %timeit clusters

pca = PCA(n_components=2).fit_transform(X_std)

plt.figure(figsize=(10,5))
colours = 'rbg'
for i in range(pca.shape[0]):
    plt.text(pca[i, 0], pca[i, 1], str(clusters[i]),
             color=colours[y[i]],
             fontdict={'weight': 'bold', 'size': 50}
        )

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

# Commented out IPython magic to ensure Python compatibility.
print("Number of clusters when min_samples=1 is: {}".format(len(np.unique(clusters))))

min_samples_list = range(2,10)

for i in range(2,10):
    dbscan_cluster = DBSCAN(eps=1, min_samples=i, metric="euclidean")
    # Fit model
    clusters = dbscan_cluster.fit_predict(X_std)
#     %timeit clusters
    print("Number of clusters when min_samples={} is: {}".format(i, len(np.unique(clusters))))

# Commented out IPython magic to ensure Python compatibility.
print("Number of clusters when eps=1 is: {}".format(len(np.unique(clusters))))

for i in [0.1,0.5,1,2,3,4,5,6,7,8,9,10]:
    dbscan_cluster = DBSCAN(eps=i, min_samples=1, metric="euclidean")
    # Fit model
    clusters = dbscan_cluster.fit_predict(X_std)
#     %timeit clusters
    print("Number of clusters when eps={} is: {}".format(i, len(np.unique(clusters))))

# Commented out IPython magic to ensure Python compatibility.
# Defining the agglomerative clustering with trial and error:
dbscan_cluster = DBSCAN(eps=3, min_samples=420, metric="euclidean")

# Fit model:
clusters = dbscan_cluster.fit_predict(X_std)
# %timeit clusters

print("Adjusted Rand Index of the DBSCAN solution: {}"
      .format(metrics.adjusted_rand_score(y, clusters)))

print("The silhouette score of the DBSCAN solution: {}"
      .format(metrics.silhouette_score(X_std, clusters, metric='euclidean')))

"""Gaussian:"""

# Commented out IPython magic to ensure Python compatibility.
# Defining the agglomerative clustering
gmm_cluster = GaussianMixture(n_components=3, random_state=123)

# Fit model
clusters = gmm_cluster.fit_predict(X_std)
# %timeit clusters

pca = PCA(n_components=2).fit_transform(X_std)

plt.figure(figsize=(10,5))
colours = 'rbg'
for i in range(pca.shape[0]):
    plt.text(pca[i, 0], pca[i, 1], str(clusters[i]),
             color=colours[y[i]],
             fontdict={'weight': 'bold', 'size': 50}
        )

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

probs = gmm_cluster.predict_proba(X_std)

size = 50 * probs.max(1) ** 2  # Squaring emphasizes differences

plt.figure(figsize=(10,5))
colours = 'rbg'
for i in range(pca.shape[0]):
    plt.text(pca[i, 0], pca[i, 1], str(clusters[i]),
             color=colours[y[i]],
             fontdict={'weight': 'bold', 'size': size[i]}
        )

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# Defining the agglomerative clustering
gmm_cluster = GaussianMixture(n_components=3, random_state=123, covariance_type="full")

# Fit model
clusters = gmm_cluster.fit_predict(X_std)
# %timeit clusters

print("ARI score with covariance_type=full: {}".format(metrics.adjusted_rand_score(y, clusters)))

print("Silhouette score with covariance_type=full: {}".format(
    metrics.silhouette_score(X_std, clusters, metric='euclidean')))
print("------------------------------------------------------")

# Defining the agglomerative clustering
gmm_cluster = GaussianMixture(n_components=3, random_state=123, covariance_type="tied")

# Fit model
clusters = gmm_cluster.fit_predict(X_std)
# %timeit clusters

print("ARI score with covariance_type=tied: {}".format(metrics.adjusted_rand_score(y, clusters)))

print("Silhouette score with covariance_type=tied: {}".format(
    metrics.silhouette_score(X_std, clusters, metric='euclidean')))
print("------------------------------------------------------")

# Defining the agglomerative clustering
gmm_cluster = GaussianMixture(n_components=3, random_state=123, covariance_type="diag")

# Fit model
clusters = gmm_cluster.fit_predict(X_std)
# %timeit clusters

print("ARI score with covariance_type=diag: {}".format(
    metrics.adjusted_rand_score(y, clusters)))

print("Silhouette score with covariance_type=diag: {}".format(
    metrics.silhouette_score(X_std, clusters, metric='euclidean')))
print("------------------------------------------------------")


# Defining the agglomerative clustering
gmm_cluster = GaussianMixture(n_components=3, random_state=123, covariance_type="spherical")

# Fit model
clusters = gmm_cluster.fit_predict(X_std)
# %timeit clusters

print("ARI score with covariance_type=spherical: {}".format(metrics.adjusted_rand_score(y, clusters)))

print("Silhouette score with covariance_type=spherical: {}".format(metrics.silhouette_score(X_std, clusters, metric='euclidean')))
print("------------------------------------------------------")

"""# **Choosing the best-performing methods: a discussion on dimensionality reduction and clustering algorithms and how they enable us to gain insights regarding the data.**

*Dimensionality Reduction:*


*   **Separating the data into distinct clusters:** PCA with three principal components (yielded two clusters most distinguishably in its visualization).
  * It appears that PCA produced the most clear result when using a variety of techniques for dimensionality reduction to visualize the data into separate clusters, and that increasing the dimensions from two to three principal components slightly improves the result. 
      * In contrast, UMAP seems to provide the worst-performing representation of the distance between data points and neighbors, as well as the similarity within each neighborhood of data in the dataset, since it produced a visualization that resembles a scatter plot rather than a distinct set of as few clusters as possible. 
  * Nevertheless, setting t-SNE with a high value for its perplexity parameter may yield a more interesting result, as it appears to perform better as the perplexity value increases. At the moment, PCA performs best in reducing dimensionality due to clarity in its visualization, and it may be valuable to test t-SNE further before comparing these methods again.

*   **Speed:** PCA with two principal components (executed in just over 0.02 seconds) far outdid t-SNE and UMAP.
  * After reviewing the amount of time that elapsed over the course of executing each method, it is evident that PCA of two main components completes the task in the shortest duration of time. While t-SNE and UMAP each required between about 26.64 and 113.28 seconds, respectively, at their fastest times, these methods are still much slower than the quickest time that PCA reduced dimensions to visualize the data with (again, about 0.02 seconds).

*Clustering algorithms:*


*   **Clustering:** Hierarchical clustering with `ward` linkage or DBSCAN with `eps=3` and `min_samples=420`. 
  * After running several methods for clustering the data, it seems that Hierarchical clustering with `ward` linkage performs the best. Specifically, the fact that this method creates the most clearly distinct clusters is evidenced by its ARI and Silhouette scores. It attains a much higher ARI score than almost every other method (aside from Gaussian clustering with either `diag` or `spherical` covariance type) and a silhouette score that is greater than all Gaussian methods' and at the same time not incredibly far from the higher end of silhouette scores (about 0.175 points lower than the highest, which is attained by DBSCAN with `eps=3` and `min_samples=420`). 
      * K-means with `k=2` appears to also be a well-performing method here, as it is the most visually appealing representation of the data and its algorithm yields ARI and Silhouette scores that are very close and only just under those of hierarchical clustering with `ward` linkage. Nevertheless, the contingency table provided unfortunately shows that more than half of the data points are placed inaccurately when comparing the clusters against the data. 

  * DBSCAN with `eps=3` and `min_samples=420` also provides a well-clustered image of the data and actually yields the highest silhouette score, so it creates the separation of the data that most accurately matches the objects in the dataset with their own clusters (in other words, this method does not match objects to neighboring clusters as much as other tested methods do). However, DBSCAN produced an extremely low ARI scores (only about 0.001 points greater than the lowest score, which was yielded by Hierarchical clustering with either `average` or `complete` linkage), so it is unlikely that the data is recovered well in the process of utilizing techniques of DBSCAN.

*   **Speed:** Gaussian mixture with `covariance_type=diag`.
  * Gaussian clustering with `covariance_type=diag` performs the task the quickest, as the `timeit` method recorded a score of about 31.1 nanoseconds. This technique also, however, yielded a much lower Silhouette score in its results than other methods that performed at similar speeds. In effect, it may be unreliable to evaluate the performance of clustering methods on speed, especially considering the fact that other techniques used over the course of this exploration produced much better performance metrics while executing with the same amount of clusters.
      * With the exceptions of the initial K-means execution with `n_clusters=3` (at about 42 milliseconds) and K-means minibatching with k=3 clusters (at about 11 milliseconds), most (or rather, practically all) of the clustering tools used performed at almost the same speed (between about 31 to 33 nanoseconds).

*Insights and Best-performing Methods:*


*   **Clarity/Accuracy:** Considering that the data points tend to scatter in inconsistent ways in the visual representations of each clustering technique, it is possible that the parameters set for each method in this study were not ideal for separating the data.
  * Conversely, it is also possible that the `['TRACT']` column in the dataset contributed to the inconsistent clustering, as it may be the case that its extremely high standard deviation value is indicative of extreme (perhaps too random) variation that prevents the data from being divided consistently enough to engender distinct clusters.

*   **Principal Components prevalent in the data:** Most of the clustering techniques employed appear to exhibit three main components in the data. While there are visualizations of the data that include more than three principal components, these results still tend to have only two or three dominating clusters in which the data is sorted into. 
  * According to several t-SNE-generated representations of the data, there may be value in attempting to reduce dimensionality on a larger scale of `perplexity` in order to more confidently determine a clear amount of main components prevalent in the data. Nevertheless, it is again possible that these results would improve by dropping the `['TRACT']` label from our `X` value during the Exploratory Data Analysis step of this study.

Thank you for your time and attention over the course of this project!
"""
