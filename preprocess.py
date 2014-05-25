# coding: utf-8
from __future__ import print_function, unicode_literals
import os
import codecs
import json
import re
from itertools import cycle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import manifold
from sklearn.decomposition import RandomizedPCA
import matplotlib.pyplot as plt
from utils import removeNonAscii, RegexpReplacer, pipe

# Global constant
REGEX = re.compile(r",\s*")


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
            with codecs.open(filePath, 'r', 'latin_1', errors='replace') as f:
                raw = f.read()
                cleaned = removeNonAscii(raw)
                data.append(cleaned)
    return data, filenames


def tokenize_strings(strings):
    '''Take a list of strings and convert each
       string into a list of string of tokens.
        Parameters
            strings: list of strings, e.g.,
            [str1, str2, str3... strN]
        Returns
           a generator for a list of unigram tokens separated by commas, e.g.,
           [[tokena1, tokenb1...], [tokena2, tokenb2...]...]
    '''
    tokenizer = RegexpTokenizer('[\w]+')
    return (tokenizer.tokenize(string) for string in strings)


def remove_stopwords(lists_tokens, filePath=None, verbose=False):
    '''Remove stopwords. The stopwords are a union of those from the nltk
       english stopwords files and any custom stopwords in an external file.
       Parameters
        list_tokens: a generator for strings of comma separated tokens, e.g.,
                     [[tokena1, tokenb1...], [tokena2, tokenb2...]...]
        filePath: path to file containing custom stopwords. The file
                  should have one stopword per line
        verbose: If True, prints out diagnostic information
        Returns
            filtered: new generator containing tokens with stopwords removed
    '''
    english_stops = set(stopwords.words('english'))
    if filePath:
        with codecs.open(filePath, 'r', 'utf-8') as f:
            combined_stops = set([line.strip() for line in f])
    stops = english_stops | combined_stops
    filtered = ((token for token in tokens
                 if token not in stops)
                for tokens in lists_tokens)
    if verbose:
        print(stops)
        total = 0
        for tokens in lists_tokens:
            total += len(tokens)
        print('Length of Original= ', total)
        total = 0
        for tokens in filtered:
            total += len(tokens)
        print('Length of Filtered= ', total)
    return filtered


def stem(lists_tokens):
    '''Take lists of tokens and stem them.
        Parameters
            list_tokens: lists containing strings of tokens, e.g.,
                         [[tokena1, tokenb1...], [tokena2, tokenb2...]...]
        Returns
            stemmed: lists containing strings of stemmed tokens, e.g.,
                     [[tokena1s, tokenb1s...], [tokena2s, tokenb2s...]...]
    '''
    english_stemmer = SnowballStemmer('english')
    stemmed = ((english_stemmer.stem(token)
                for token in tokens)
               for tokens in lists_tokens)
    return stemmed


def flatten(lists_tokens, sep=', '):
    '''Takes lists of tokens as strings and converts each list into
       one string with tokens separated by white space. Format needed by
       Scikit-learn for tokens as input to CountVectorizer.
        Parameters
            list_tokens: lists containing strings of tokens, e.g.,
                         [[tokena1, tokenb1...], [tokena2, tokenb2...]...]
        Returns
            tokens_w: a generator for strings of tokens
                      separated by sep, e.g.,
                      ['tokena1, tokena2,...', 'tokenb1, tokenb2,...',...]
    '''
    return (sep.join(l) for l in lists_tokens)


def patent_preprocessor(data, PATH_REPLACE=None, PATH_SW=None):
    '''Workflow for taking a lists of strings and tokenizing them
       into a form that Scikit-learn can use.
        Parameters
            data: list of strings, e.g.,
            ['string1`, 'string2', ....]
            PATH_REPLACE: path to json file with replacement patterns
            PATH_SW: path to txt file with stop words
        Returns
             a list containing tokenized strings, e.g.,
                      [ 'tokena1 tokena2...', 'tokenb1 tokenb2...' ]
    '''
    fname = PATH_REPLACE
    with open(fname, 'r') as f:
        d = json.load(f)
    replacement_patterns = [(key, value) for key, value in d[0].items()]
    replacer = RegexpReplacer(patterns=replacement_patterns)
    clean = lambda data: (replacer.replace(datum) for datum in data)
    ls_tolower = lambda ls: (element.lower() for element in ls)
    tokenized = pipe((clean, ), (ls_tolower, ), (tokenize_strings, ))(data)
    filtered = remove_stopwords(tokenized, filePath=PATH_SW)
    stemmed = stem(filtered)
    gen_tokens = flatten(stemmed, sep=', ')
    return [tokens for tokens in gen_tokens]


def custom_tokenizer(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]


def tokens_tovectors(list_tokens, verbose=True):
    '''Convert word tokens to vectors.
        Parameters
            list_tokens: strings of whitespace separated tokens (features)
            verbose: True to print additional diagnositc information
        Returns
            patentvectorizer: class of CountVectorizer with initialized
                              parameters
            vectorized: counts for each feature
            patenttransformer: class of TfidfTransformer with initialized
                         parameters
            transformed: tf-idf values for each feature

    '''

    patentvectorizer = CountVectorizer(tokenizer=custom_tokenizer,
                                       max_df=0.95,
                                       decode_error='ignore')
    vectorized = patentvectorizer.fit_transform(list_tokens)
    patenttransformer = TfidfTransformer()
    transformed = patenttransformer.fit_transform(vectorized.toarray())
    if verbose:
        num_samples, num_features = vectorized.shape
        print('#samples {0:d} #features {1:d}'.format(num_samples,
                                                      num_features))
        print(patentvectorizer.get_feature_names())
        print('****Vectorized****')
        print(vectorized)
        print(vectorized.toarray())
        print('****Transformed****')
        print(transformed)
        print(transformed.toarray())
    vectordict = {'tfidf_vectors': transformed,
                  'tfidf_instance': patenttransformer,
                  'count_vectors': vectorized,
                  'countvector_instance': patentvectorizer}
    return vectordict


def vector_metrics(vectors, plot=True):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000,
                       eps=1e-6, random_state=seed,
                       dissimilarity='euclidean', n_jobs=1)
    pos = mds.fit_transform(vectors)
    if plot:
        plt.figure()
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        plt.scatter(x, y)
        plt.show()
    return x, y


def vector_characteristics(patentdict, vectordict):
    print('Vectorized Shape: ', vectordict['count_vectors'].shape)
    print('Transformed Shape: ', vectordict['tfidf_vectors'].shape)
    print('Number of Categories: ', len(patentdict['target_names']))
    print('\n****Target Names***\n', patentdict['target_names'])
    print('Number of Features:', len(vectordict['feature_names']))
    print('\n****Feature Names****\n', vectordict['feature_names'])


def pca_metric(d, vd):
    vector_pca = RandomizedPCA(n_components=2).fit_transform(vd['count_vectors'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure()
    for i, c in zip(np.unique(d['target']), cycle(colors)):
        plt.scatter(vector_pca[d['target'] == i, 0],
                    vector_pca[d['target'] == i, 1],
                    c=c, label=d['target_names'][i], alpha=0.5)

    plt.legend(loc='best')
    plt.show()
