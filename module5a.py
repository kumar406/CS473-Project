from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

def cluster_erds(input, K):
    #turn erd into vector of stuff
    vecs = []
    dictionary = {}
    for erd in input:
        for features in erd:
            for word in features[1:]:
                dictionary[word] = 1

    keys = list(dictionary.keys())
    print(keys)
    for erd in input:       
        vec = [0 for x in range(7 + len(dictionary))]
        for feature in erd:
            if (feature[0] == 'entity'):
                vec[0] += 1
            elif (feature[0] == 'weak_entity'):
                vec[1] += 1
            elif (feature[0] == 'rel'):
                vec[2] += 1
            elif (feature[0] == 'ident_rel'):
                vec[3] += 1
            elif (feature[0] == 'rel_attr'):
                vec[4] += 1
            elif (feature[0] == 'many'):
                vec[5] += 1
            elif (feature[0] == 'one'):
                vec[6] += 1
            for x in feature[1:]:
                if (x in dictionary):
                    vec[keys.index(x) + 7] += 1
        vecs.append(vec / np.linalg.norm(vec))

    print(vecs)
    # if k = 0 determine k
    if(int(K) == 0):
        best_k = 2
        curr_highest_score = -1
        for val in range(2, len(vecs)):
            kmeans = KMeans(n_clusters=val, random_state = 0, init='k-means++').fit(vecs)
            s_score = silhouette_score(vecs, kmeans.labels_, metric='euclidean')
            if(s_score > curr_highest_score):
                curr_highest_score = s_score
                best_k = val
        K = best_k

    # kmeans++
    kmeans = KMeans(n_clusters=int(K), random_state = 0, init='k-means++').fit(vecs)
    kmeans_clusters = kmeans.labels_

    score = silhouette_score(vecs, kmeans_clusters, metric='euclidean')
    print(score)

    # get clusters for erd files
    clusters = []
    for number in range(int(K)):
        clusters.append(number + 1)

    # get each filename's cluster assignment 
    erds_cluster_assignment = {k: [] for k in imgs}

    index = 0
    for key,val in erds_cluster_assignment.items():
        erds_cluster_assignment[key] = kmeans_clusters[index] + 1
        index = index + 1

    # group filenames by cluster
    erds_clusters = {k: [] for k in clusters}

    for key,val in erds_cluster_assignment.items():
        erds_clusters[val].append(key)
        
    with open("base_line_clusters.txt", "w") as f:
        for key,values in erds_clusters.items():
            f.write(str(values) + "\n")