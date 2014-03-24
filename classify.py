from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cross_validation import ShuffleSplit, cross_val_score
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import preprocess as pp
from utils import removeNonAscii


def loadClassifiedData(pathtraining):
    '''
    Loads classified data to be used for training or a model
    or to perform a training/testing split
    Returns:
        X_data: A list containing strings read from the files
        y_target: A ndarray containing the classification
                categories which have been converted to numbers.
        Each string in X_data corresponds to a category in
        y_target
    '''
    dataDict = load_files(pathtraining,
                          description=None, categories=None,
                          load_content=True, shuffle=True,
                          encoding='latin-1', decode_error='strict',
                          random_state=0)
    return dataDict


def display_important_features(feature_names, target_names, weights, n_top=30):
    for i, target_name in enumerate(target_names):
        print("Class: " + target_name)
        print("")

        sorted_features_indices = weights[i].argsort()[::-1]

        most_important = sorted_features_indices[:n_top]
        print(", ".join("{0}: {1:.4f}".format(feature_names[j], weights[i, j])
                        for j in most_important))
        print("...")

        least_important = sorted_features_indices[-n_top:]
        print(", ".join("{0}: {1:.4f}".format(feature_names[j], weights[i, j])
                        for j in least_important))
        print("")

# Global Constants
PATH_REPLACE = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                        'clustering/custom/replacements.json'])
PATH_SW = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                   'clustering/custom/combined-stop-words.txt'])
PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                     'clustering/data/new-summaries/'])


# Get the training and testing data
patentdict = loadClassifiedData(PATH_DATA)
categories = patentdict['target']
data = patentdict['data']
# for datum, category in zip(data, categories):
#     print(str(category))
#     print(datum[0:100])

data_stripped = [removeNonAscii(item) for item in data]

# Preprocess and tokenize
list_tokens = pp.patent_tokenizer(data_stripped,
                                  PATH_REPLACE=PATH_REPLACE,
                                  PATH_SW=PATH_SW)
# print(list_tokens)

# Vectorize
(transformed, patenttransformer,
 vectorized, patentvectorizer) = pp.tokens_tovectors(list_tokens,
                                                     verbose=False)
print('Vectorized Shape: ', vectorized.shape)
print('Transformed Shape: ', transformed.shape)
feature_names = patentvectorizer.get_feature_names()
target_names = patentdict['target_names']
print('Number of Categories: ', len(target_names))
print('\n****Target Names***\n', target_names)
print('Number of Features:', len(feature_names))
print('\n****Feature Names****\n', feature_names)
pp.pca_metric(vectorized, patentdict)

# Train the classifier
#clf = MultinomialNB()
clf = PassiveAggressiveClassifier(C=1)
cv = ShuffleSplit(len(data_stripped), n_iter=10,
                  test_size=0.7, random_state=42)

# Get information on what the classifier learned
fitted = clf.fit(transformed, categories)
feature_weights = clf.coef_
print('****Feature Weights****\n', feature_weights)
display_important_features(feature_names, target_names, feature_weights)

# Calculate the classifier metrics
f1_scores = []
cms = []
for train, test in cv:
    X_train, y_train = transformed[train], categories[train]
    X_test, y_test = transformed[test], categories[test]
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    predicted = clf.predict(X_test)
    f1_scores.append(test_score)
    cms.append(confusion_matrix(y_test, predicted))
print('****F1 Test Scores****\n')
for f1_score in f1_scores:
    print('\t {0:0.3f}'.format(f1_score))
scores = cross_val_score(clf, transformed, categories, cv=cv)
# print('f1 scores: ')
# print(['{:.3f}'.format(val) for val in scores])
print('Average: {0:0.3f}\nStd Dev: {1:0.4f}\n'.format(scores.mean(),
                                                      scores.std()))
mat = np.matrix(sum(cms))
frac = np.round(mat / mat.sum(axis=1, dtype='float'), decimals=3)
print('****Confusion Matrix****\n')
df = pd.DataFrame(dict(zip(target_names, frac.transpose())),
                  columns=target_names)
df['names'] = target_names
df['# of documents'] = [categories[categories == i].shape[0]
                        for i in range(len(target_names))]
print(df)
# predicted = clf.predict(transformed)
# print('****Classification Report****\n')
# print(classification_report(categories, predicted,
                            # target_names=patentdict['target_names']))
