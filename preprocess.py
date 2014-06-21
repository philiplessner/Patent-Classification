# coding: utf-8
from __future__ import print_function, unicode_literals
import os
import io
import json
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
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
                data.append(raw)
    return data, filenames


def loadClassifiedData(pathtraining):
    '''
    Loads classified data to be used for training or a model
    or to perform a training/testing split
    Parameter
        pathtraining: path to top level directory where
                      classified data is located
    Returns:
        dataDict: {'data': list of strings, one string per document
                   'target': array of  category labels correspoding to data
                   'target_names': names of the categories}
    '''
    dataDict = load_files(pathtraining,
                          description=None, categories=None,
                          load_content=True, shuffle=True,
                          encoding='latin-1', decode_error='strict',
                          random_state=0)
    return dataDict


class PatentVectorizer(object):

    def __init__(self, data, PATH_REPLACE=None, PATH_SW=None):
        with io.open(PATH_REPLACE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        with io.open(PATH_SW, 'r', encoding='utf-8') as f:
            combined_stops = set([line.strip() for line in f])
        english_stops = set(stopwords.words('english'))
        self.stops = english_stops | combined_stops
        self.english_stemmer = SnowballStemmer('english')
        self.data = data
        replacement_patterns = [(key, value)
                                for key, value in d[0].items()]
        self.replacer = RegexpReplacer(patterns=replacement_patterns)

    def _patent_analyzer(self, document):
        '''
        Custom analyzer that preprocesses a summarized patent document
        and tokenizes it.
        Parameter
            document: A single string representing a summarized patent document
        Returns
            a list of tokens from string
        '''
        # Strip non-ascii characters
        s_acii = removeNonAscii(document)
        # Replacements in string
        clean = self.replacer.replace(s_acii)
        # Lower case string
        lower_case = clean.lower()
        # Tokenize string to list of tokens
        tokenized = RegexpTokenizer('[\w]+').tokenize(lower_case)
        # Remove stop words from list of tokens
        filtered = [token for token in tokenized if token not in self.stops]
        # Stem list of tokens
        return [self.english_stemmer.stem(token) for token in filtered]

    def patent_totfidf(self):
        '''
        Convert summarized patent documents to tf-idf vectors
        Returns
            vectordict: {'tfidf_vectors': transformed,
                         'tfidf_instance': patenttransformer,
                         'count_vectors': vectorized,
                         'countvector_instance': patentvectorizer,
                         'feature_names': patentvectorizer.get_feature_names()}
        '''
        patentvectorizer = TfidfVectorizer(analyzer=self._patent_analyzer,
                                           max_df=0.95,
                                           decode_error='ignore')
        vectorized = patentvectorizer.fit_transform(self.data)
        self.vectordict = {'tfidf_vectors': vectorized,
                           'tfidf_instance': patentvectorizer,
                           'feature_names': patentvectorizer.get_feature_names()}
        return self.vectordict

    def vector_characteristics(self, labels=None):
        '''
        Print some vector characteristics
        '''
        print('Transformed Shape: ', self.vectordict['tfidf_vectors'].shape)
        print('Number of Features:', len(self.vectordict['feature_names']))
        print('\n****Feature Names****\n', self.vectordict['feature_names'])
        if labels:
            print('Number of Categories: ', len(labels))
            print('\n****Target Names***\n', labels)

    def pca_metric(self, cl, labels):
        vector_pca = TruncatedSVD(
            n_components=2).fit_transform(self.vectordict['tfidf_vectors'])
        colors = ['b', 'g', 'r', 'y', 'k', 'c', 'm']
        plt.figure(figsize=(10, 10))
        for i, c in zip(np.unique(cl), cycle(colors)):
            plt.scatter(vector_pca[cl == i, 0],
                        vector_pca[cl == i, 1],
                        c=c, label=labels[i], alpha=1.0)

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
