# coding=utf-8
from __future__ import print_function
import os
import codecs
import re
from collections import OrderedDict
import numpy as np
from rake_nltk import RakeKeywordExtractor
from utils import RegexpReplacer, lower_case


def open_strip(filepath):
    with codecs.open(filepath, 'r', 'utf-8') as f:
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
    string = '|'.join(strings)
    pat = re.compile(string, re.IGNORECASE)
    with codecs.open(filepath, 'r', 'latin_1') as f:
        for i, line in enumerate(f):
            if (i == 0) and (not pat.search(line)):
                tmp = 'DEFAULT'
                d[tmp] = ''
            elif pat.search(line):
                line = re.sub(r'[\n\r]+', '', line)
                d[line] = ''
                tmp = line
            else:
                d[tmp] += line
    return d


def sent_tok(string):
    '''
    Takes a string and tokenizes it into sentences.
    Parameters
        string: a string
    Returns
        a list of strings with each string a sentence
    '''
    # rdict = {'U[.]S[.]': 'US',
    #          'Pat[.]': 'Patent'}
    # robj = re.compile('|'.join(rdict.keys()))
    # result = robj.sub(lambda m: rdict[m.group(0)], string)
    replacement_patterns = [(r'U[.]S[.]', 'US '),
                            (r'Pat[.]', 'Patent '),
                            (r'Fig[.]', 'Fig '),
                            (r'FIG[.]', 'Fig')]
    replacer = RegexpReplacer(patterns=replacement_patterns)
    cleaned = replacer.replace(string)
    stok = re.compile('[.!?][\s]{1,2}(?=[A-Z])')
    return stok.split(cleaned)


def find_strsent(sentences, strs=None):
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


def filter_dict(d, filters):
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


def make_summary(d, clue_words):
    summary = []
    for key in d.keys():
        # Split section into sentences
        sent_tokens = sent_tok(d[key])
        lcsent_tokens = lower_case(sent_tokens)
        # Check for clues in sentences
        nfound_clues = np.array(find_strsent(lcsent_tokens, strs=clue_words))
        # Generate a list of keywords (rake)
        rake = RakeKeywordExtractor()
        keyphrases = rake.extract(d[key], incl_scores=True)
        phrases_only = [phrase for phrase, score in keyphrases]
        # Check for keywords phrases in sentences
        nfound_keyphrases = np.array(find_strsent(lcsent_tokens,
                                     strs=phrases_only[0:7]))
        # Rank sentences
        subtotals = nfound_clues + nfound_keyphrases
        sect_summary = '****' + str(key) + '****' + '\n'
        for i, (subtotal, sent_token) in enumerate(zip(subtotals,
                                                       sent_tokens)):
            if (subtotal > 1):
                sent_token = sent_token + ' '
                sect_summary += sent_token
        summary.append(sect_summary)
    return '\n'.join(summary)


def section_diagnostics(d, clue_words):
    print('****SECTIONS ****', '\n')
    # Process each section
    for key in d.keys():
        print(key)
        print(d[key], '\n')
        # Split section into sentences
        sent_tokens = sent_tok(d[key])
        lcsent_tokens = lower_case(sent_tokens)
        # Check for clues in sentences
        nfound_clues = np.array(find_strsent(lcsent_tokens, strs=clue_words))
        # Generate a list of keywords (rake)
        rake = RakeKeywordExtractor()
        keyphrases = rake.extract(d[key], incl_scores=True)
        print(keyphrases, '\n')
        phrases_only = [phrase for phrase, score in keyphrases]
        # Check for keywords phrases in sentences
        nfound_keyphrases = np.array(find_strsent(lcsent_tokens,
                                     strs=phrases_only[0:7]))
        # Rank sentences
        subtotals = nfound_clues + nfound_keyphrases
        for i, (subtotal, sent_token) in enumerate(zip(subtotals,
                                                       sent_tokens)):
            if (subtotal > 1):
                print(i, subtotal, sent_token, '\n')
    print('\n', '****KEYS****', '\n')
    for key in d.keys():
        print(key)


if __name__ == '__main__':
    # Initial Setup
    BASE_PATH = ''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/clustering/custom/'])
    PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/clustering/data/us-only'])
    PATH_OUTPUT = ''.join(['/Users/dpmlto1/Documents/Patent/',
                           'Thomson Innovation/clustering/data/summaries/'])
    # Get set-up data
    clues = open_strip(''.join([BASE_PATH, 'clue-words.txt']))
    section_titles = open_strip(''.join([BASE_PATH, 'section-titles.txt']))
    filtered_titles = open_strip(''.join([BASE_PATH, 'filtered-titles.txt']))

    # Split into sections
    for root, folders, files in os.walk(PATH_DATA):
        for filename in files:
            filePath = os.path.join(root, filename)
            sectdict = split_file(filePath, section_titles)
            filtered_dict = filter_dict(sectdict, filtered_titles)
            # Make the summary
            with codecs.open(''.join([PATH_OUTPUT, filename]),
                             'w', 'latin_1') as f:
                f.write(make_summary(filtered_dict, clues))
