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
data, filenames = pp.file_tostring(PATH_DATA)
# Preprocess, Tokenize, and Vectorize
pv = pp.PatentVectorizer(data, PATH_REPLACE=PATH_REPLACE,
                         PATH_SW=PATH_SW)
vectordict = pv.patent_totfidf()
# Properties of the vectorized data
# pv.vector_characteristics(patentdict)
# print('\n***Principal Component Analysis***\n')
# pv.pca_metric(patentdict)

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
num_clusters = 4
km, labels = cl.cluster(vectordict['tfidf_vectors'], num_clusters=num_clusters)

# Compute metrics
sil_scores = cl.cluster_metrics(km, labels, vectordict['tfidf_vectors'])

# Collect output into pandas dataframe and write to disk
dictionary = {'filenames': filenames, 'clusters': labels,
              'silhouettes': sil_scores,
              'data': data}
df = pd.DataFrame(dictionary, columns=['filenames', 'clusters',
                                       'silhouettes', 'data'])
df = df.sort_index(by='clusters')
print df
print df.head()
df.to_csv(''.join([PATH_OUTPUT, 'output.csv']),
          encoding='latin-1', index=False)

# Plot metrics
cl.sil_plot(df)
