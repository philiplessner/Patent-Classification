# coding: utf-8
from __future__ import print_function, division, unicode_literals
import preprocess as pp
import classify as cl


def main():
    # Get unclassified data
    PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/clustering/data/',
                         'unclassified-summaries/'])
    data, filenames = pp.file_tostring(PATH_DATA)
    # Perform the classification
    vector_filepath = ''.join(['/Users/dpmlto1/Documents/Patent/'
                               'Thomson Innovation/clustering/',
                               'data/vectorvocab.pkl'])
    clffile = [''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                        'clustering/data/classifier.pkl'])]
    predicted_class = cl.patent_predict(data, vector_filepath, clffile)
    for pc, filename in zip(predicted_class, filenames):
        print(pc, '\t', filename)

if __name__ == '__main__':
        main()
