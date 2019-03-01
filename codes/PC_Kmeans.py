import numpy as np


class PC_Kmeans:
    def __init__(self, k, ml, cl, neighborhoods, y,w=1, tolerance=0.00001, max_iterations = 300):
        self.k = k
        self.ml = ml  # is transitive graph
        self.cl = cl  # is graph
        self.neighborhoods = neighborhoods
        self.y = y
        self.w = w
        self.tolerance = tolerance
        self.max_iterations = max_iterations


    def fit(self,data):

        # Initialize centroids
        self.centroids = {}
        cluster_centers = self.init_centers(data, self.neighborhoods,self.y)
        for i in range(len(cluster_centers)):
            self.centroids[i] = cluster_centers[i]

        # Repeat until convergence
        for i in range(self.max_iterations):
            # print("iteration :",i)
            self.clusters = {}  # {0:[] , 1:[] , ... , k:[]}
            for i in range(self.k):
                self.clusters[i] = set()

            # Assign clusters
            alter = self.assign_clusters(data, cluster_centers, self.ml, self.cl, self.w)

            if alter == "empty":
                return "Cluster Not Found"

            # average the cluster data points to re-calculate the centroids
            previous_centers = cluster_centers
            cluster_centers = self.compute_mean_cluster_centers(data)

            isOptimal = True
            for centert_index in range(len(cluster_centers)):
                original_centroid = previous_centers[centert_index]
                curr = cluster_centers[centert_index]
                diff = curr - original_centroid
                if np.sum(diff / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
            if isOptimal:
                break


    def assign_clusters(self,data,cluster_centers,ml,cl,w):
        self.is_clustered = [-1] * len(data)
        for x_index in range(len(data)):
            h = [self.objective_function(data, x_index, cluster_centers, cluster_index, self.is_clustered, ml, cl, w) for cluster_index in range(self.k)]
            center_index = np.argmin(h)
            self.is_clustered[x_index] = center_index
            self.clusters[center_index].add(x_index)

        # Handle empty clusters
        empty_cluster_flag = False
        for i in range(self.k):
            if i not in self.is_clustered:
                empty_cluster_flag = True
                break
        if empty_cluster_flag:
            return "empty"

    def objective_function(self, data, x_index, centroids, cluster_index, is_clustered, ml, cl, w):

        distance = 1 / 2 * np.sum((data[x_index] - centroids[cluster_index]) ** 2)

        ml_penalty = 0
        for i in ml[x_index]:
            if is_clustered[i] != -1 and is_clustered[i] != cluster_index:
                ml_penalty += w

        cl_penalty = 0
        for i in cl[x_index]:
            if is_clustered[i] == cluster_index:
                cl_penalty += w

        return distance + ml_penalty + cl_penalty



    def init_centers(self, data, neighborhoods,y):
        neighborhoods = sorted(neighborhoods, key=len,reverse=True)  # srot neighborhoods base on size
        neighborhood_centers = np.array([data[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        #neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        if len(neighborhoods) > self.k:
            # Select K largest neighborhoods' centroids
            # cluster_centers1 = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.k:]]
            cluster_centers = neighborhood_centers[:self.k]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, data.shape[1]))

            # look for a point that is connected by cannot-links to every neighborhood set
            for x_index in range(len(data)):
                find_flag = 1
                for neighborhood_set in neighborhoods:
                    if y[x_index] == y[neighborhood_set[0]]:
                        find_flag = 0
                        break
                if find_flag == 1:  # find a point that is connected by cannot-links to every neighborhood set
                    np.append(cluster_centers, [data[x_index]], axis=0)
                    break


            if len(neighborhoods) < self.k:
                remaining_cluster_centers = data[np.random.choice(len(data), self.k - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers

    def compute_mean_cluster_centers(self, data):
        for _center in self.clusters:
            lst = []
            for index_value in self.clusters[_center]:
                lst.append(data[index_value])
            avgArr = np.array(lst)

            if len(lst) != 0:
                self.centroids[_center] = np.average(avgArr, axis=0)
        clusters_center = []
        for key,value in self.centroids.items():
            clusters_center.append(value)

        return np.array(clusters_center)



