# imports
from constrainedClustering.COP_Kmeans import COP_KMeans
from constrainedClustering.PC_Kmeans import PC_Kmeans
from constrainedClustering.generateConstraints import generateConstraintsUsingNeighborhoods,\
                                                generateRandomConstraints,\
                                                transitive_entailment_graph
from constrainedClustering.load_datasets import load_dataset
from constrainedClustering.ExploreConsolidate import Explore, Consolidate
from constrainedClustering.Oracle import Oracle

from sklearn import metrics
from sklearn.cluster import KMeans,AgglomerativeClustering
from matplotlib import pyplot as plt

# configs

DATASETS = ['iris', 'ecoli', 'glass', 'pima', 'sonar', 'wine', 'soybean', 'ionosphere', 'balance', 'breast']
DATASET_NAME = DATASETS[2]

MAX_QUERY_COUNT = 20
max_query_cont_list = [10, 20, 30, 40, 50, 60, 70]

MAX_CONSTRAINT = 15
max_constraint_list = [10, 20, 30, 40, 50, 60, 70 , 80 ,90 , 100]

GENERATE_ML_CL_METHOD = 'random'  # 'random' or 'active'


# read data
x, y, k = load_dataset(DATASET_NAME)



NMI_list_pckmeans = {}
NMI_list_copkmeans = {}
NMI_list_kmeans = {}
NMI_list_hierarcy = {}

repeat = 4


for number_constraint_or_query in max_constraint_list:
    sum_nmi_pckmeans = 0
    sum_nmi_copkmeans = 0
    sum_nmi_kmeans = 0
    sum_nmi_hierarcy = 0

    for i in range(repeat):
        print("***************************************************number_constraint :",number_constraint_or_query)
        print("***************************************************iteration :", i)
        # define oracle


        # explore consolidate|minmaxConsolidate phase [arg max_query]
        if GENERATE_ML_CL_METHOD == 'active':
            oracle = Oracle(y, number_constraint_or_query)
            neighborhoods = Explore(x,k,oracle)
            neighborhoods = Consolidate(neighborhoods,x,oracle,method='minmax')
            ml,cl = generateConstraintsUsingNeighborhoods(neighborhoods)
            ml_g, cl_g, n = transitive_entailment_graph(ml,cl,len(x))

        else : # GENERATE_ML_CL_METHOD == 'random':
            ml,cl = generateRandomConstraints(x,y,max_constraint=number_constraint_or_query)
            ml_g, cl_g, neighborhoods = transitive_entailment_graph(ml, cl, len(x))

        '''---------------- COP-Kmeans ---------------'''

        # COP_Kmeans test repeat n times
        cop_kmeans = COP_KMeans(k,ml_g,cl_g)
        cop_kmeans.fit(x)
        predict_labels_copkmeans = cop_kmeans.is_clustered

        # NMI cop_kmeans
        NMI_copkmeans = metrics.normalized_mutual_info_score(y,predict_labels_copkmeans)
        sum_nmi_copkmeans += NMI_copkmeans

        '''---------------- PC-Kmeans ---------------'''

        # PC_Kmeans  test repeat n times
        pc_kmeans = PC_Kmeans(k, ml_g, cl_g, neighborhoods,y)
        pc_kmeans.fit(x)
        predict_labels_pckmeans = pc_kmeans.is_clustered

        # NMI PC_Kmeans
        NMI_pckmeans = metrics.normalized_mutual_info_score(y, predict_labels_pckmeans)
        sum_nmi_pckmeans+=NMI_pckmeans

        '''---------------- simple-Kmeans ---------------'''
        # simple K-means
        k_means = KMeans(n_clusters=k).fit(x)
        predict_labels_kmeans = k_means.labels_

        # NMI simple K-means
        NMI_kmeans = metrics.normalized_mutual_info_score(y, predict_labels_kmeans)
        sum_nmi_kmeans += NMI_kmeans

        '''---------------- Hierarchical ----------------'''
        hierarchical_clus = AgglomerativeClustering(n_clusters=k)
        hierarchical_clus.fit(x)
        predict_labels_hierarchy = hierarchical_clus.labels_

        NMI_hierarchy = metrics.normalized_mutual_info_score(y, predict_labels_hierarchy)
        sum_nmi_hierarcy += NMI_hierarchy

    NMI_list_copkmeans[number_constraint_or_query] = sum_nmi_copkmeans/repeat
    NMI_list_pckmeans[number_constraint_or_query] = sum_nmi_pckmeans/repeat
    NMI_list_kmeans[number_constraint_or_query] = sum_nmi_kmeans/repeat
    NMI_list_hierarcy[number_constraint_or_query] = sum_nmi_hierarcy/repeat
# plot nmi result of all test

print('\n\n\n\n\n\n\n\n')
print(DATASET_NAME,'-- constraints:',GENERATE_ML_CL_METHOD)
print('pckmeans NMI:',NMI_list_pckmeans)
print('copkmeans NMI:',NMI_list_copkmeans)
print('simple kmeans NMI:',NMI_list_kmeans)
print('agglomorative NMI:',NMI_list_hierarcy)


labels = ['PC-Kmeans', 'COP-Kmeans', 'Simple K-means', 'Agglomorative']
lst = [NMI_list_pckmeans, NMI_list_copkmeans , NMI_list_kmeans , NMI_list_hierarcy]
shapes = ['-o', '-v', '-s','-x']
for item, shape in zip(lst, shapes):
    xx = []
    yy = []
    for key, value in item.items():
        xx.append(key)
        yy.append(value)
    plt.plot(xx,yy,shape)
    plt.legend(labels)

plt.title(DATASET_NAME.upper())
plt.ylabel('Normalized Mutual Information')
plt.xlabel('# of query')
plt.show()

