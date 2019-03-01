import numpy as np

def dist(i, neighborhood, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in neighborhood])
    return distances.min()
