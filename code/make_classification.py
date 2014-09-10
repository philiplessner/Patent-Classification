# coding: utf-8
from __future__ import print_function, division, unicode_literals
import preprocess as pp
import classify as cl


def main():
    # Get unclassified data
    PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/clustering/data/',
                         'unclassified-summaries/'])
    datadict = pp.DataLoader(PATH_DATA=PATH_DATA).load_unclassified_data()
    # Perform the classification
    vector_filepath = ''.join(['/Users/dpmlto1/Documents/Patent/'
                               'Thomson Innovation/clustering/',
                               'data/vectorvocab.pkl'])
    categoryname_filepath = ''.join(['/Users/dpmlto1/Documents/Patent/'
                                     'Thomson Innovation/clustering/',
                                     'data/targetnames.pkl'])
    clffile = [''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                        'clustering/data/classifier.pkl'])]
    predicted = cl.patent_predict(datadict['data'], vector_filepath, clffile)
    category_names = cl.unpickle_withdill(categoryname_filepath)
    targetd = dict(zip(range(len(category_names) + 1), category_names))
    # Make the classfication report
    df_p = cl.category_probs(predicted['X'], predicted['y'], predicted['clf'],
                             category_names, targetd, datadict['filenames'])
    ordercols = (['category_name', 'category_num', 'files'] + category_names)
    df_p = df_p.reindex_axis(ordercols, axis=1, copy=False)
    df_p.to_csv(''.join(['/Users/dpmlto1/Documents/Patent/'
                         'Thomson Innovation/clustering/',
                         'data/classification_report.csv']))

    print(df_p)

if __name__ == '__main__':
        main()
