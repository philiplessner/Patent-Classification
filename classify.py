from __future__ import print_function
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report
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
                     'clustering/data/categories/'])


# Get the training and testing data
patentdict = loadClassifiedData(PATH_DATA)
categories = patentdict['target']
data = patentdict['data']
for datum, category in zip(data, categories):
    print (str(category))
    print(datum[0:100])

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
cv = ShuffleSplit(len(data_stripped), n_iter=5,
                  test_size=0.5, random_state=42)

# Get information on what the classifier learned
fitted = clf.fit(transformed, categories)
feature_weights = clf.coef_
print('****Feature Weights****\n', feature_weights)
display_important_features(feature_names, target_names, feature_weights)

# Calculate the classifier metrics
scores = cross_val_score(clf, transformed, categories, cv=cv)
print('f1 scores: ')
print(['{:.3f}'.format(val) for val in scores])
print('Accuracy: {0:0.3f} (+/- {1:0.4f})\n'.format(scores.mean(),
                                                   scores.std() / 2))
predicted = clf.predict(transformed)
print('****Classification Report****\n')
print(classification_report(categories, predicted,
                            target_names=patentdict['target_names']))
print('****Confusion Matrix****\n')
print(confusion_matrix(categories, predicted))
