import easyocr
import os
import sys
import math
import re
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import math
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def getERDs():

    # Initialize easyocr reader
    reader = easyocr.Reader(['en'])

    # parameters for module4 methods
    file_name = "parameters.txt"
    dir_path = None
    K = None

    # parse args
    if len(sys.argv) >= 2:
        file_name = sys.argv[1]

    # read file for directory path and K
    f = open(file_name)
    dir_path = f.readline().strip()
    K = f.readline().strip()

    # find all Images
    img_list = os.listdir(dir_path) 

    image_words = dict()
    for img in img_list:
        temp_word_list = reader.readtext(dir_path + "/" + img, detail = 0)
        image_words[os.path.splitext(img)[0]] = temp_word_list
    
    return(image_words, K)

# <---------------------------------------------------->

def module3(contents):
    
    erds_contents = contents

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

    # get stopwords list
    with open("stopwords.txt") as f:
        stopwords = [word.strip() for word in f.readlines()]

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

    erds_contents = stemmed_erd_content

    return erds_contents

# <---------------------------------------------------->

def kMeans():

    image_words, K = getERDs()
    
    erds_filenames = image_words.keys()
    erds_contents = module3(image_words.values())


    # get unique words words
    words = []
    for erd in erds_contents:
        for word in erd:
            words.append(word)
    unique_words = set(words)
    unique_words = {k: 0 for k in unique_words}

    # get document vector
    erds_vectors = []

    for erd in erds_contents:
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
        
    with open("base_line_clusters.txt", "w") as f:
        for key,values in erds_clusters.items():
            f.write(str(values) + "\n")


kMeans()