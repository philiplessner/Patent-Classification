# coding=utf-8
from __future__ import print_function, unicode_literals
import os
import codecs
import re
from collections import OrderedDict
from sumtech import MakeSummary


def open_strip(filepath):
    '''
    Strip newlines from end of line
    Parameter
        filepath: full path to file
    Returns
        list of strings stripped on newlines
    '''
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
    # Append the lookahead assertion so that a string in the
    # middle of a line will not be counted as a section name
    stringsn = [stringx + '(?![ .,])' for stringx in strings]
    stringn = '|'.join(stringsn)
    pat = re.compile(stringn, re.IGNORECASE)
    with codecs.open(filepath, 'r', 'latin_1') as f:
        for i, line in enumerate(f):
            if (i == 0) and (not pat.search(line)):
                tmp = 'DEFAULT'
                d[tmp] = ''
            elif pat.search(line.strip()):
                # line = re.sub(r'[\n\r]+', '', line)
                d[line.strip()] = ''
                tmp = line.strip()
            else:
                d[tmp] += line.strip()
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
    summary = []
    # Argument to MakeSummary is compression ratio
    sumo = MakeSummary(50.0)
    for key in d.keys():
        # sect_summary = '\n'.join(['****', unicode(key), '****',
                                  # sumo.patentsum(d[key])])
        sect_summary = sumo.patentsum(d[key])
        summary.append(sect_summary)
    return '\n'.join(summary)


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
            with codecs.open(''.join([PATH_OUTPUT,
                                      os.path.join(os.path.split(root)[1],
                                                   filename)]),
                             'w', 'latin_1') as f:
                f.write(make_summary(filtered_dict))
                # for key in filtered_dict:
                    # f.write('\n****' + key + '****\n')
                    # f.write(filtered_dict[key])
