# K_means 

K-means is a prototype-based clustering method, i.e. the clusters are represented by prototypes (eg center). Normally one looks at data from the n-dimensional continuous space.
The goal is to divide given data points into K clusters. 

Given
  * Input: K, set of points x1,...xn
  * Place centroids c1, ..cn at random locations  
  
Steps: 
  * Choose K points as starting centers repeat
  * Map each point to the cluster closest to its center
  * Recalculate Cluster Centers
  * until the centers do not change anymore

In this scripts, I implement it using numpy library to classify numbers into two clusters.  
The purpose of this script is to better understand the functioning of k-means and structure of formed clusters. 
