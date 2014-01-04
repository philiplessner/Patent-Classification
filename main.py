import warnings
import numpy as np
import pandas as pd
import preprocess as pp
import cluster as cl


# Issue between pandas 0.12 and sklearn. Should be fixed in 0.13
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="pandas", lineno=570)
# Global Constants
PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                     'clustering/data/summaries'])
PATH_OUTPUT = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                       'clustering/output/'])
PATH_REPLACE = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                        'clustering/custom/replacements.json'])
PATH_SW = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                   'clustering/custom/combined-stop-words.txt'])

# Preprocess and tokenize
data, filenames = pp.file_tostring(PATH_DATA)
list_tokens = pp.patent_tokenizer(data,
                                  PATH_REPLACE=PATH_REPLACE,
                                  PATH_SW=PATH_SW)
print list_tokens

# Vectorize
(transformed, patenttransformer,
 vectorized, patentvectorizer) = pp.tokens_tovectors(list_tokens)
x, y = pp.vector_metrics(transformed, plot=True)
max_features = []
feature_names = patentvectorizer.get_feature_names()
for oned in np.argsort(transformed.toarray(), axis=1):
    max_features.append([feature_names[j] for j in oned[-5:]])
# fcs = sorted([fctuple for fctuple in zip(feature_names,
                                     # np.asarray(vectorized.sum(axis=0)).ravel())],
                                      # key=lambda idx: idx[1])
    # print str(fcs[1]) + ' ', fcs[0]

features = {'feature-names': feature_names,
            'idf': patenttransformer.idf_}
df_features = pd.DataFrame(features)
df_features = df_features.sort_index(by='idf')
df_features.to_csv(''.join([PATH_OUTPUT, 'features.csv']))
# Cluster vectors
num_clusters = 4
km, labels = cl.cluster(transformed, num_clusters=num_clusters)

# Compute metrics
sil_scores = cl.cluster_metrics(km, labels, transformed)

# Collect output into pandas dataframe and write to disk
dictionary = {'filenames': filenames, 'clusters': labels,
              'silhouettes': sil_scores, 'maxfeatures': max_features,
              'data': data}
df = pd.DataFrame(dictionary, columns=['filenames', 'clusters',
                                       'silhouettes', 'maxfeatures', 'data'])
df = df.sort_index(by='clusters')
print df
print df.head()
df.to_csv(''.join([PATH_OUTPUT, 'output.csv']), index=False)

# Plot metrics
cl.sil_plot(df)
