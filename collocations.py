from __future__ import print_function
import re
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import preprocess as pp
from utils import lower_case


def find_ngrams(data, PATH_SW=None, ntopbg=10, ntoptg=10):
    '''Find top occuring bigrams and trigrams in a corpus
        Parameters
            data: list of strings (each string in list is a document in corpus)
            PATH_SW: path to stop words file
            ntopbg: how many bigrams to return
            ntoptg: how many trigrams to return
        Returns
            topbg: list of tuples containing top bigrams
            toptg: list of tuples containing top trigrams

    '''
    long_string = ' '.join(data)
    tokenizer = RegexpTokenizer('[\w]+')
    words = tokenizer.tokenize(long_string)
    # english_stemmer = SnowballStemmer('english')
    # stemmed = [english_stemmer.stem(item)
               # for item in filter_stops]
    # print(stemmed)
    bef = BigramCollocationFinder.from_words(words)
    tcf = TrigramCollocationFinder.from_words(words)
    with open(PATH_SW, 'r') as f:
        stops = [re.sub(r'\s', '', line) for line in f]
    stopset = set(stops)
    filter_stops = lambda w: w in stopset
    bef.apply_word_filter(filter_stops)
    tcf.apply_word_filter(filter_stops)
    tcf.apply_freq_filter(3)
    topbg = bef.nbest(BigramAssocMeasures.likelihood_ratio, ntopbg)
    toptg = tcf.nbest(TrigramAssocMeasures.likelihood_ratio, ntoptg)
    return topbg, toptg


if __name__ == '__main__':
    PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/clustering/data/DWPI'])
    PATH_SW = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                       'clustering/custom/combined-stop-words.txt'])
    data, filenames = pp.file_tostring(PATH_DATA)
    lowercase = lower_case(data)
    topbg, toptg = find_ngrams(lowercase, PATH_SW=PATH_SW)
    print('****Bigrams****')
    print(topbg, '\n')
    print('****Trigrams****')
    print(toptg)
