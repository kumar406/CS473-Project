import easyocr
import os
import math
import re
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import math
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def module5(STOPWORDS_FILE, PARAMETERS_FILE, image_words_output, m1_out):

    # google stopwords
    with open(STOPWORDS_FILE) as f:
        stopwords = [word.strip() for word in f.readlines()]

    # Initialize easyocr reader
    reader = easyocr.Reader(['en'])

    # parameters for module4 methods
    file_name = PARAMETERS_FILE
    dir_path = None
    K = None

    # read file for directory path and K
    f = open(file_name)
    dir_path = f.readline().strip()
    K = f.readline().strip()

    erds_filenames = image_words_output.keys()
    erds_contents = image_words_output.values()

    # <------------------MODULE 3---------------------------------->

    # seperate words with spaces
    erds_remove_spaces = []
    for array in erds_contents:
        tmp = []
        for word in array:
            words = word.split(' ')
            for split_word in words:
                tmp.append(split_word)
        erds_remove_spaces.append(tmp)

    erds_contents = erds_remove_spaces

    # seperate words with underscores
    erds_remove_underscore = []
    for array in erds_contents:
        tmp = []
        for word in array:
            words = word.split('_')
            for split_word in words:
                tmp.append(split_word)
        erds_remove_underscore.append(tmp)

    erds_contents = erds_remove_underscore

    # seperate words with hyphens
    erds_remove_hyphen = []
    for array in erds_contents:
        tmp = []
        for word in array:
            words = word.split('-')
            for split_word in words:
                tmp.append(split_word)
        erds_remove_hyphen.append(tmp)

    erds_contents = erds_remove_hyphen

    # seperate camel case
    erds_remove_camel_case = []
    for array in erds_contents:
        tmp = []
        for word in array:
            words = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()
            for split_word in words: 
                tmp.append(split_word)
        erds_remove_camel_case.append(tmp)

    erds_contents = erds_remove_camel_case

    # remove stopwords
    for erd in erds_contents:
        for word in erd:
            if word.lower() in stopwords:
                erd.remove(word)

    # stem words 
    ps = PorterStemmer()

    stemmed_erd_content = []

    for erd in erds_contents:
        stemmed_words = [ps.stem(word) for word in erd]
        stemmed_erd_content.append(stemmed_words)

    # <---------------------------------------------------->

    # get unique words words
    words = []
    for erd in stemmed_erd_content:
        for word in erd:
            words.append(word)
    unique_words = set(words)
    unique_words = {k: 0 for k in unique_words}

    # get document vector
    erds_vectors = []

    for erd in stemmed_erd_content:
        dictionary = corpora.Dictionary()
        dict_BoW_corpus = [dictionary.doc2bow(erd, allow_update=True)]
        dict_id_words = [[(dictionary[id], count) for id, count in line] for line in dict_BoW_corpus]
        dict_tf = dict(dict_id_words[0])
        
        # term frequency for each erd
        words_erd = unique_words.copy()
        for key,val in dict_tf.items():
            if(val > 0):
                words_erd[key] = math.log10(val)+1

        vector = []
        for key,val in words_erd.items():
            vector.append(val)
        
        erds_vectors.append(vector)

        words_erd = unique_words   

    # dictionary of file name with vectors 
    erds_vectors_with_filename = {k: [] for k in erds_filenames}

    index = 0 
    for key,val in erds_vectors_with_filename.items():
        erds_vectors_with_filename[key] = erds_vectors[index]
        index = index + 1

    # combine vectors with method 1 in module 5 
    for erd in m1_out:
        fname = os.path.basename(os.path.splitext(erd['img'])[0])
        if (fname in erds_vectors_with_filename):
            vec = [0 for x in range(7)]
            for entity in erd['labels']:
                vec[(entity['label']['id'] - 1)] += 1
            erds_vectors_with_filename[fname] += vec

    erds_vect = []

    for key, value in erds_vectors_with_filename.items():
        erds_vect.append(value)

    erds_vectors = erds_vect

    # if k = 0 determine k
    if(int(K) == 0):
        best_k = 2
        curr_highest_score = -1
        for val in range(2, len(erds_filenames)):
            kmeans = KMeans(n_clusters=val, random_state = 0, init='k-means++').fit(erds_vectors)
            s_score = silhouette_score(erds_vectors, kmeans.labels_, metric='euclidean')
            if(s_score > curr_highest_score):
                curr_highest_score = s_score
                best_k = val
        K = best_k

    # kmeans++
    kmeans = KMeans(n_clusters=int(K), random_state = 0, init='k-means++').fit(erds_vectors)
    kmeans_clusters = kmeans.labels_

    score = silhouette_score(erds_vectors, kmeans_clusters, metric='euclidean')
    print(score)

    # get clusters for erd files
    clusters = []
    for number in range(int(K)):
        clusters.append(number + 1)

    # get each filename's cluster assignment 
    erds_cluster_assignment = {k: [] for k in erds_filenames}

    index = 0
    for key,val in erds_cluster_assignment.items():
        erds_cluster_assignment[key] = kmeans_clusters[index] + 1
        index = index + 1

    # group filenames by cluster
    erds_clusters = {k: [] for k in clusters}

    for key,val in erds_cluster_assignment.items():
        erds_clusters[val].append(key)
        
    return erds_clusters




    




    
    
