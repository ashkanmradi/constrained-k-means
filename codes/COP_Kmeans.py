import random
import numpy as np


class COP_KMeans:
    def __init__(self, k, ml, cl, tolerance=0.00001, max_iterations=300):
        self.k = k
        self.ml = ml
        self.cl = cl
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data , initial_method='random'):

        self.centroids = {}

        # initialize the centroids, the random 'k' elements in the dataset will be our initial centroids
        self.centroids = self.initialize_centers(data,self.k, initial_method)  #return dic


        # begin iterations
        for i in range(self.max_iterations):
            # print('for :',i)
            self.is_clustered = [-1] * len(data)

            self.clusters = {}   # {0:[] , 1:[] , ... , k:[]}
            for i in range(self.k):
                self.clusters[i] = set()

            # find the distance between the point and cluster; choose the nearest centroid
            for x_index in range(len(data)):
                distances = {center_index: np.linalg.norm(data[x_index] - self.centroids[center_index])for center_index in self.centroids}
                sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])
                empty_flag = True

                for center_index,dis_value in sorted_distances:  # ??????
                    vc_result = self.violate_constraints(x_index, center_index, self.ml, self.cl)  # boolean
                    # vc_result = False
                    if not vc_result:
                        self.clusters[center_index].add(x_index)
                        self.is_clustered[x_index] = center_index

                        for j in self.ml[x_index]:
                            self.is_clustered[j] = center_index

                        empty_flag = False
                        break
                if empty_flag:
                    return "Clustering Not Found Exception"


            previous = dict(self.centroids)

            # average the cluster data points to re-calculate the centroids
            for _center in self.clusters:
                lst = []
                for index_value in self.clusters[_center]:
                    lst.append(data[index_value])
                avgArr = np.array(lst)

                if len(lst) != 0:
                    self.centroids[_center] = np.average(avgArr, axis = 0)  # ????????????

            isOptimal = True
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False


            if isOptimal:
                break

    def violate_constraints(self, data_index, cluster_index , ml, cl):
        for i in ml[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] != cluster_index :
                return True

        for i in cl[data_index]:
            if self.is_clustered[i] != -1 and self.is_clustered[i] == cluster_index :
                return True
            
        return False

    def decoder(self,x_data,y_data):
        # for i, j in zip(d.values(), y):
        #     if i != j:
        #         sum += 1
        print(self.clusters)

    def initialize_centers(self, dataset, k, method):
        if method == 'random':
            # print(method)
            c = {}
            index_list = list(range(len(dataset)))
            random.shuffle(index_list)
            for i in range(k):
                c[i] = dataset[index_list[i]]  # center is index of data point in datas {0:data[25] , 1:72 , 2:124}

        elif method == 'km++':
            chances = [1] * len(dataset)
            centers = []
            c = {}

            for i in range(k):
                chances = [x / sum(chances) for x in chances]
                r = random.random()
                acc = 0.0
                for index, chance in enumerate(chances):
                    if acc + chance >= r:
                        break
                    acc += chance
                centers.append(dataset[index])
                c[i] = dataset[index]
                for index, point in enumerate(dataset):
                    # cids, distances = closest_clusters(centers, point)
                    distances = {center_index: np.linalg.norm(point - c[center_index]) for center_index in c}
                    sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])
                    # chances[index] = distances[cids[0]]
                    chances[index] = sorted_distances[0][1]

        return c

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

