# coding: utf-8
from __future__ import print_function, unicode_literals
import numpy as np
import pandas as pd
import preprocess as pp
import cluster as cl


# Global Constants
PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                     'clustering/data/train-test-summaries'])
PATH_OUTPUT = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                       'clustering/output/'])
PATH_REPLACE = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                        'clustering/custom/replacements.json'])
PATH_SW = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                   'clustering/custom/combined-stop-words.txt'])

# Get the data
datadict = pp.DataLoader(PATH_DATA=PATH_DATA).load_unclassified_data()
# Preprocess, Tokenize, and Vectorize
pv = pp.PatentVectorizer(datadict['data'], PATH_REPLACE=PATH_REPLACE,
                         PATH_SW=PATH_SW)
vectordict = pv.patent_totfidf()
# Properties of the vectorized data
pv.vector_characteristics()

# max_features = []
# feature_names = vectordict['tfidf_instance'].get_feature_names()
# for oned in np.argsort(vectordict['tfidf_vectors'].toarray(), axis=1):
#     max_features.append([feature_names[j] for j in oned[-5:]])

# features = {'feature-names': feature_names,
#             'idf': vectordict['tfidf_instance'].idf_}
# df_features = pd.DataFrame(features)
# df_features = df_features.sort_index(by='idf')
# df_features.to_csv(''.join([PATH_OUTPUT, 'features.csv']))
# Cluster vectors
sils = []
for num_clusters in xrange(2, 10, 1):
    km, clusters = cl.cluster(vectordict['tfidf_vectors'],
                              num_clusters=num_clusters)

    print('\n***Principal Component Analysis***\n')
    pv.pca_metric(clusters, np.unique(clusters))

    # Compute metrics
    sil_scores = cl.cluster_metrics(km, clusters, vectordict['tfidf_vectors'])

    # Collect output into pandas dataframe and write to disk
    dictionary = {'filenames': datadict['filenames'], 'clusters': clusters,
                  'silhouettes': sil_scores,
                  'data': datadict['data']}
    df = pd.DataFrame(dictionary, columns=['filenames', 'clusters',
                                           'silhouettes', 'data'])
    df = df.sort_index(by='clusters')
    print(df.head())
    df.to_csv(''.join([PATH_OUTPUT, 'output.csv']),
              encoding='latin-1', index=False)

    # Plot metrics
    # cl.sil_plot(df)
    print('\nFor num_clusters =', num_clusters,
          '\tAverage Silhouette Score=', df['silhouettes'].mean())
    sils.append(df['silhouettes'].mean())
for index, num_clusters in enumerate(xrange(2, 10, 1)):
    print(num_clusters, '\t', sils[index])
