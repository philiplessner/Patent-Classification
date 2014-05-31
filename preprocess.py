# coding: utf-8
from __future__ import print_function, unicode_literals
import os
import io
import json
from itertools import cycle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from utils import removeNonAscii, RegexpReplacer


def file_tostring(PATH_DATA):
    '''Read text files into a list of strings
        Parameter
            PATH_DATA: Path to data files
        Returns
            data: list of strings
            filenames: list of filenames read
    '''
    data = []
    filenames = []
    for root, folders, files in os.walk(PATH_DATA):
        for filename in files:
            filenames.append(filename)
            filePath = os.path.join(root, filename)
            with io.open(filePath, 'r', encoding='latin_1',
                         errors='replace') as f:
                raw = f.read()
                cleaned = removeNonAscii(raw)
                data.append(cleaned)
    return data, filenames


def patent_totfidf(data, PATH_REPLACE=None, PATH_SW=None):
    with io.open(PATH_REPLACE, 'r', encoding='utf-8') as f:
        d = json.load(f)
    with io.open(PATH_SW, 'r', encoding='utf-8') as f:
        combined_stops = set([line.strip() for line in f])

    def patent_analyzer(document):
        # Replacements in string
        replacement_patterns = [(key, value) for key, value in d[0].items()]
        clean = RegexpReplacer(patterns=replacement_patterns).replace(document)
        # Lower case string
        lower_case = clean.lower()
        # Tokenize string to list of tokens
        tokenized = RegexpTokenizer('[\w]+').tokenize(lower_case)
        # Remove stop words from list of tokens
        english_stops = set(stopwords.words('english'))
        stops = english_stops | combined_stops
        filtered = [token for token in tokenized if token not in stops]
        # Stem list of tokens
        english_stemmer = SnowballStemmer('english')
        return [english_stemmer.stem(token) for token in filtered]

    patentvectorizer = CountVectorizer(analyzer=patent_analyzer,
                                       max_df=0.95,
                                       decode_error='ignore')
    vectorized = patentvectorizer.fit_transform(data)
    patenttransformer = TfidfTransformer()
    transformed = patenttransformer.fit_transform(vectorized.toarray())
    vectordict = {'tfidf_vectors': transformed,
                  'tfidf_instance': patenttransformer,
                  'count_vectors': vectorized,
                  'countvector_instance': patentvectorizer,
                  'feature_names': patentvectorizer.get_feature_names()}
    return vectordict


def vector_characteristics(patentdict, vectordict):
    print('Vectorized Shape: ', vectordict['count_vectors'].shape)
    print('Transformed Shape: ', vectordict['tfidf_vectors'].shape)
    print('Number of Categories: ', len(patentdict['target_names']))
    print('\n****Target Names***\n', patentdict['target_names'])
    print('Number of Features:', len(vectordict['feature_names']))
    print('\n****Feature Names****\n', vectordict['feature_names'])


def pca_metric(d, vd):
    vector_pca = TruncatedSVD(
        n_components=2).fit_transform(vd['tfidf_vectors'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(10, 10))
    for i, c in zip(np.unique(d['target']), cycle(colors)):
        plt.scatter(vector_pca[d['target'] == i, 0],
                    vector_pca[d['target'] == i, 1],
                    c=c, label=d['target_names'][i], alpha=0.5)

    plt.legend(loc='best')
    plt.show()


def mds_metric(vectors, plot=True):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000,
                       eps=1e-6, random_state=seed, n_jobs=1)
    pos = mds.fit_transform(vectors)
    if plot:
        plt.figure()
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        plt.scatter(x, y)
        plt.show()
    return x, y
