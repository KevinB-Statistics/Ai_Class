#Clustering Algorithm
'''
Three clustering algorithms;
* Kmeans - cluster data into K different groups
1. Assigns centers of the clusters randomly in space
2. Assign each point in the data to the closest cluster
3. Move the center to the center of all the points associated with it
4. Repeat until the centers stop moving


* Hierarchial Clustering - 
Two kinds: 
Agglomerative (Starts with eery data point in their cluster and creates new clusters until one left)
Divisive (starts with one cluster and divides clusters until none left)


* DBScan - uses 2 parameters and 3 types of data points
Parameter 1 - epsilon (distance metrics; if the distance between two points is less than epsilon then they are neighbors)
Paramater 2 - minimum points (minimum number of neighbors within epsilon readius; generally = dimensions + 1)

Core point: A data point that has more points than minimum points in an epsilon radius)
Border point: not a core point (fewer than minpts within eps) but a neighbor of a core point
Noise : not a core or border point

Finds all neighbor pts within eps, find core points
For each core point, make a new cluster if not already there

Normalization: comparing distance on different scales can be tricky, normalization puts all data on a scale of 0 to 1 
Value - minimum/(max-min) (min-max scaling)

'''
