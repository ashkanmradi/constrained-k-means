from constrainedClustering.computeDistance import dist
import numpy as np

def Explore(data, k, oracle):
    neighborhoods = []
    isTraversed = []
    dslen = len(data)
    x = np.random.choice(dslen)
    neighborhoods.append([x])
    isTraversed.append(x)

    while len(neighborhoods) < k:
        max_dist = 0
        farthest = None

        for i in range(dslen):
            if i not in isTraversed:
                distance = dist(i, isTraversed, data)
                if distance > max_dist:
                    max_dist = distance
                    farthest = i


        new_neighborhood = True

        if farthest == None:
            break

        for neighborhood in neighborhoods:
            if oracle.query(farthest, neighborhood[0]):
                neighborhood.append(farthest)
                new_neighborhood = False
                break

        if new_neighborhood:
            neighborhoods.append([farthest])


        isTraversed.append(farthest)

    return neighborhoods


def Consolidate(neighborhoods, data, oracle, method='FFQS'):
    if method == "FFQS":
        n = len(data)
        neighborhoodsUnion = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoodsUnion.add(i)

        remaining = set()
        for i in range(n):
            if i not in neighborhoodsUnion:
                remaining.add(i)

        flag = True
        while oracle.can_query():
            if len(remaining) == 0 and flag:
                break

            flag = False

            i = np.random.choice(list(remaining))

            sorted_neighborhoods = sorted(neighborhoods, key=lambda neighborhood: dist(i, neighborhood, data))

            for neighborhood in sorted_neighborhoods:
                if oracle.query(i, neighborhood[0]):
                    neighborhood.append(i)
                    break

            neighborhoodsUnion.add(i)
            remaining.remove(i)

        return neighborhoods

    elif method == 'minmax':

        n = len(data)
        clusterSkeleton = set()

        for neighborhood in neighborhoods:
            for i in neighborhood:
                clusterSkeleton.add(i)

        remaining = set()
        for i in range(n):
            if i not in clusterSkeleton:
                remaining.add(i)

        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distances[i, j] = np.sqrt(((data[i] - data[j]) ** 2).sum())

        kernelWidth = np.percentile(distances, 20)

        while oracle.can_query():
            maxSimilarity = np.full(n, fill_value=float('+inf'))
            for data_i in remaining:
                maxSimilarity[data_i] = np.max([similarity(data[data_i], data[data_j], kernelWidth)for data_j in clusterSkeleton])

            mostUncertainIndex = maxSimilarity.argmin()

            sorted_neighborhoods = reversed(sorted(neighborhoods, key=lambda neighborhood: np.max(
                [similarity(data[mostUncertainIndex], data[n_i], kernelWidth) for n_i in neighborhood])))

            for neighborhood in sorted_neighborhoods:
                if oracle.query(mostUncertainIndex, neighborhood[0]):
                    neighborhood.append(mostUncertainIndex)
                    break

            clusterSkeleton.add(mostUncertainIndex)
            remaining.remove(mostUncertainIndex)

        return neighborhoods


def similarity(data, y, kernel_width):
    return np.exp(-((data - y) ** 2).sum() / (2 * (kernel_width ** 2)))



