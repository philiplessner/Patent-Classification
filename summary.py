# coding=utf-8
from __future__ import print_function, unicode_literals, division
import os
import io
import re
import operator
import string
from subprocess import call
import tempfile
from collections import OrderedDict
import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from utils import lower_case, RegexpReplacer


class MakeSummary(object):
    '''
    Methods for summarizing text
    '''
    def __init__(self, r):
        '''
        Parameter
            r: compression ratio for summary (float)
        '''
        self.r = r

    def _preprocess(self, document):
        '''
        Strip excess white space and newlines
        Parameter
            document: document string
        Returns
            a string with spaces stripped from beginning and end and
            newlines replace with spaces
        '''
        return ' '.join(document.strip().split('\n'))

    def _find_strsent(self, sentences, strs=None):
        '''
        Find if strings are present in a list of strings.
        Parameters
            sentences: a list of strings
            strs: a list of strings to find in sentences
        Returns
            nfound_strs: a list of the total number of matches in
                         each sentence
        '''
        exact_match = re.compile(r'\b%s\b' % '\\b|\\b'.join(strs))
        found_strs = [exact_match.findall(sentence)
                      for sentence in sentences]
        nfound_strs = [len(found) for found in found_strs]
        return nfound_strs

    def _sent_tok(self, string):
        '''
        Takes a string and tokenizes it into sentences.
        Parameters
            string: a string
        Returns
            a list of strings with each string a sentence
        '''
        replacement_patterns = [(r'U[.]S[.]', 'US '),
                                (r'Pat[.]', 'Patent '),
                                (r'Fig[.]', 'Fig '),
                                (r'FIG[.]', 'Fig')]
        replacer = RegexpReplacer(patterns=replacement_patterns)
        cleaned = replacer.replace(string)
        stok = re.compile('[.!?][\s]{1,2}(?=[A-Z])')
        return stok.split(cleaned)

    def _get_topsent(self, sentence_scores):
        '''
        Get the top ranked sentences and join them in original order
        Parameter
            sentence_scores:
            list of tuples [(score, sentence=-string, sentence-position)...]
        Returns
            string of top ranked sentence in original order
        '''
        sentence_ranked = sorted(sentence_scores, key=lambda y: y[0],
                                 reverse=True)
        sum_length = int(round((self.r / 100.0) * len(sentence_scores)))
        tops = sentence_ranked[0:sum_length]
        tops_ordered = sorted(tops, key=lambda y: y[2])
        tops_ordered.append((0, '', 0))
        return '. '.join([x[1] for x in tops_ordered])

    def textrank(self, document):
        '''
        Use TextRank algorithm to find most relvant sentence in document
        Parameters
            document: document string
        Returns
            string of top ranked sentences in original order
        '''
        sentences = self._sent_tok(self._preprocess(document))

        bow_matrix = CountVectorizer().fit_transform(sentences)
        normalized = TfidfTransformer().fit_transform(bow_matrix)

        similarity_graph = normalized * normalized.T

        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores = nx.pagerank(nx_graph)
        # List of tuples [(score, sentence-string, sentence-position)...]
        sentence_scores = [(scores[i], s, i)
                           for i, s in enumerate(sentences)]
        return self._get_topsent(sentence_scores)

    def patentsum(self, document):
        '''
        Rank sentences according to presence of clue words and keywords (Rake)
        Parameter
            document: document string
        Returns
            string of top ranked sentences in original order
        '''
        clue_words = ['advantage', 'avoid', 'cost', 'costly', 'decrease',
                      'difficult', 'effectivenss', 'efficiency',
                      'goal', 'important', 'improved',
                      'increase', 'issue', 'limit', 'needed',
                      'overhead', 'performance', 'problem',
                      'reduced', 'resolve', 'shorten', 'simplify',
                      'suffer', 'superior', 'weakness']
        sent_tokens = self._sent_tok(self._preprocess(document))
        lcsent_tokens = lower_case(sent_tokens)
        # Check for clues in sentences
        nfound_clues = np.array(self._find_strsent(lcsent_tokens,
                                                   strs=clue_words))
        # Generate a list of keywords (rake)
        rake = RakeKeywordExtractor()
        keyphrases = rake.extract(document, incl_scores=True)
        phrases_only = [phrase for phrase, score in keyphrases]
        # Check for keywords phrases in sentences
        nfound_keyphrases = np.array(self._find_strsent(lcsent_tokens,
                                     strs=phrases_only[0:7]))
        # Rank sentences
        subtotals = nfound_clues + nfound_keyphrases
        # List of tuples [(score, sentence-stirng, sentence-position)...]
        sentence_scores = zip(subtotals, sent_tokens,
                              range(0, len(sent_tokens) + 1))
        return self._get_topsent(sentence_scores)

    def ots(self, document):
        '''
        Use ots to summarize content
        Parameters
            content: full text
        Returns
            outs: summarized text
        '''
        temp_dir = tempfile.mkdtemp()
        temp1 = tempfile.NamedTemporaryFile(
            suffix=".txt", dir=temp_dir, delete=False)
        temp2 = tempfile.NamedTemporaryFile(
            suffix=".txt", dir=temp_dir, delete=False)

        with io.open(temp1.name, 'w', encoding='utf-8') as f:
            f.write(self._preprocess(document))
        with io.open(temp2.name, 'w', encoding='utf-8') as outfile:
            call(['ots', '-r', unicode(int(self.r)), temp1.name],
                 stdout=outfile)
        with io.open(temp2.name, 'r', encoding='utf-8') as f:
            outs = f.read()
        os.remove(temp1.name)
        os.remove(temp2.name)
        return outs


class RakeKeywordExtractor(object):

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words())
        self.top_fraction = 1  # consider top third candidate keywords by score

    def _generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            words = map(lambda x: "|" if x in self.stopwords else x,
                        nltk.word_tokenize(sentence.lower()))
            phrase = []
            for word in words:
                if word == "|" or isPunct(word):
                    if len(phrase) > 0:
                        phrase_list.append(phrase)
                        phrase = []
                else:
                    phrase.append(word)
        return phrase_list

    def _calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = len(filter(lambda x: not isNumeric(x), phrase)) - 1
            for word in phrase:
                word_freq.inc(word)
                word_degree.inc(word, degree)  # other words
        for word in word_freq.keys():
            word_degree[word] = word_degree[word] + word_freq[word]  # itself
        # word score = deg(w) / freq(w)
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_degree[word] / word_freq[word]
        return word_scores

    def _calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores[word]
            phrase_scores[" ".join(phrase)] = phrase_score
        return phrase_scores

    def extract(self, text, incl_scores=False):
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(
            phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.iteritems(),
                                      key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        if incl_scores:
            return sorted_phrase_scores[0:int(n_phrases / self.top_fraction)]
        else:
            return map(lambda x: x[0],
                       sorted_phrase_scores[0:int(n_phrases / self.top_fraction)])


def isPunct(word):
    return len(word) == 1 and word in string.punctuation


def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False


def open_strip(filepath):
    '''
    Strip newlines from end of line
    Parameter
        filepath: full path to file
    Returns
        list of strings stripped on newlines
    '''
    with io.open(filepath, 'r', encoding='utf-8') as f:
        return [re.sub(r'\n', '', line) for line in f]


def split_file(filepath, strings):
    ''' Split patent text into sections.
      Parameters
        filepath: file containing patent text
        strings: list of strings to match against for dictionary keys
      Returns
        d: ordered dictionary containing sections.
           Each key is a section name and
           each value is the text of that section
    '''
    d = OrderedDict()
    # Append the lookahead assertion so that a string in the
    # middle of a line will not be counted as a section name
    stringsn = [stringx + '(?![ .,])' for stringx in strings]
    stringn = '|'.join(stringsn)
    pat = re.compile(stringn, re.IGNORECASE)
    with io.open(filepath, 'r', encoding='latin_1') as f:
        for i, line in enumerate(f):
            if (i == 0) and (not pat.search(line)):
                tmp = 'DEFAULT'
                d[tmp] = ''
            elif pat.search(line.strip()):
                # line = re.sub(r'[\n\r]+', '', line)
                d[line.strip()] = ''
                tmp = line.strip()
            else:
                d[tmp] = ' '.join([d[tmp], line.strip()])
    return d


def filter_dict(d, filters):
    '''
    Makes a dictionary with only those keys to be used in generating summary
    Parameters
        d: dictionary containing keys that are section titles and value
           that are the content of that section
        filters: section titles to keep
    Returns
        od: ordered dictionary containing sections to be used in summary
    '''
    if len(d.keys()) == 1:
        od = d
        return od
    string = '|'.join(filters)
    pat = re.compile(string, re.IGNORECASE)
    od = OrderedDict(d)
    for key in od.keys():
        if (not pat.search(key)):
            del od[key]
    return od


def make_summary(d):
    '''
    Use a summarization technique to select the most important content
    Parameter
        d: dictionary containing sections to be summarized.
           Each section will be summarized seperately
    Returns
        summarized sections as a single string
    '''
    # Argument to MakeSummary is compression ratio
    sumo = MakeSummary(50.0)
    return '\n'.join([sumo.patentsum(d[key]) for key in d.keys()])


if __name__ == '__main__':
    # Initial Setup
    BASE_PATH = ''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/clustering/custom/'])
    PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/clustering/data/uspto-full-text'])
    PATH_OUTPUT = ''.join(['/Users/dpmlto1/Documents/Patent/',
                           'Thomson Innovation/clustering/data/',
                           'new-summaries/'])
    # Get set-up data
    section_titles = open_strip(''.join([BASE_PATH, 'section-titles.txt']))
    filtered_titles = open_strip(''.join([BASE_PATH, 'filtered-titles.txt']))

    # Split into sections
    for root, folders, files in os.walk(PATH_DATA):
        for filename in files:
            filePath = os.path.join(root, filename)
            sectdict = split_file(filePath, section_titles)
            filtered_dict = filter_dict(sectdict, filtered_titles)
            # Make the summary
            with io.open(''.join([PATH_OUTPUT,
                                  os.path.join(os.path.split(root)[1],
                                               filename)]),
                         'w', encoding='latin_1') as f:
                f.write(make_summary(filtered_dict))
