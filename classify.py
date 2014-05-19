from __future__ import print_function, division, unicode_literals
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
# from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics import roc_curve, auc
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


def display_important_features(feature_names, target_names, clf, n_top=30):
    '''
    Display highest weighted and lowest weighted feastures for each category
    Parameters
        feature_names: tokens vectorized
        target_names: category names
        clf: classifier instance
        n_top: number of features to display
    '''
    fitted = clf.fit(transformed, categories)
    weights = clf.coef_
    print('****Feature Weights****\n', weights)
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


def classifier_metrics(transformed, categories, clf, cv):
    '''
    Calculate f1 scores and confusion matrix for several cv splits
    Parameters
        transformed: X vector (TF-IDF for each feature/category)
        categories: y vector
        clf: classifier instance
        cv: cross validation instance
    Returns
        cms: confusion matrix averaged over cv runs
        f1_scores: f1 score for each cv run
    '''
    labels = np.unique(categories)
    f1_scores = []
    cms = []
    tprs = defaultdict(list)
    fprs = defaultdict(list)
    roc_scores = defaultdict(list)
    for train, test in cv:
        X_train, y_train = transformed[train], categories[train]
        X_test, y_test = transformed[test], categories[test]
        clf.fit(X_train, y_train)
        # train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        predicted = clf.predict(X_test)
        f1_scores.append(test_score)
        cms.append(confusion_matrix(y_test, predicted))
        # for label in labels:
        #     y_label_test = np.asarray(y_test == label, dtype=int)
        #     proba = clf.predict_proba(X_test)
        #     proba_label = proba[:, label]
        #     fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
        #     roc_scores[label].append(auc(fpr, tpr))
        #     tprs[label].append(tpr)
        #     fprs[label].append(fpr)
    return cms, f1_scores


def classifier_scores(f1_scores):
    '''
    Prints f1 scores for cv runs and mean and std of  scores
    Parameter
        f1_scores: array of f1 scores for several cv runs
    '''
    print('****F1 Test Scores****\n')
    for f1_score in f1_scores:
        print('\t {0:0.3f}'.format(f1_score))
    scores = cross_val_score(clf, transformed, categories, cv=cv)
    # print('f1 scores: ')
    # print(['{:.3f}'.format(val) for val in scores])
    print('Average: {0:0.3f}\nStd Dev: {1:0.4f}\n'.format(scores.mean(),
                                                          scores.std()))


def output_confusionmatrix(cms, categories, target_names):
    '''
    Prints avg confusion matrix and also plots it
    Parameters
        cms: array of confusion matricies from cv runs
        categories: y vector
        target_names: category names
    '''
    # Calculate average confusion matrix and normalize to fractions
    mat = np.matrix(sum(cms))
    frac = np.round(mat / mat.sum(axis=1, dtype='float'), decimals=3)

    # Print normailized average confusion matrix
    print('****Confusion Matrix****\n')
    df = pd.DataFrame(dict(zip(target_names, frac.transpose())),
                      columns=target_names)
    df['names'] = target_names
    df['# of documents'] = [categories[categories == i].shape[0]
                            for i in range(len(target_names))]
    print(df)

    # Plot confusion matrix
    fig = plt.figure(figsize=(10., 10.))
    ax = fig.add_subplot(111)
    plt.matshow(frac, fignum=False, cmap='Blues', vmin=0., vmax=1.0)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticks(range(len(target_names)))
    ax.set_yticklabels(target_names)
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.colorbar()
    plt.show()


def plot_roc(auc_score, name, tpr, fpr, label=None):
    plt.clf()
    plt.figure(num=None, figsize=(5, 4))
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.fill_between(fpr, tpr, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (AUC = %0.2f) / %s' % (auc_score, label),
              verticalalignment="bottom")
    plt.legend(loc="lower right")
    plt.show()


def probs_class(patentdict, transformed, clf):
    '''
    Probabilities of classes for each data
    '''
    df_p = pd.DataFrame(clf.predict_proba(transformed),
                        columns=patentdict['target_names'])
    df_p['category_num'] = patentdict['target']
    targetl = patentdict['target'].tolist()
    targetd = dict(zip(range(len(patentdict['target_names']) + 1),
                       patentdict['target_names']))
    for k, v in targetd.iteritems():
        for i, e in enumerate(targetl):
            if k == e:
                targetl[i] = v
    df_p['category_name'] = targetl
    df_p['files'] = [os.path.splitext(os.path.basename(filename))[0]
                     for filename in patentdict['filenames']]
    return df_p


if __name__ == '__main__':
    # Global Constants
    PATH_REPLACE = ''.join(['/Users/dpmlto1/Documents/Patent/',
                            'Thomson Innovation/',
                            'clustering/custom/replacements.json'])
    PATH_SW = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                       'clustering/custom/combined-stop-words.txt'])
    PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                         'clustering/data/new-summaries/'])

    # Get the training and testing data
    patentdict = loadClassifiedData(PATH_DATA)
    categories = patentdict['target']
    data = patentdict['data']

    data_stripped = [removeNonAscii(item) for item in data]

    # Preprocess and tokenize
    list_tokens = pp.patent_tokenizer(data_stripped,
                                      PATH_REPLACE=PATH_REPLACE,
                                      PATH_SW=PATH_SW)

    # Vectorize
    (transformed, patenttransformer,
     vectorized, patentvectorizer) = pp.tokens_tovectors(list_tokens,
                                                         verbose=False)
    feature_names = patentvectorizer.get_feature_names()
    target_names = patentdict['target_names']
    pp.vector_characteristics(vectorized, transformed,
                              feature_names, target_names)
    pp.pca_metric(vectorized, patentdict)

    # Train the classifier
    # clf = MultinomialNB()
    Cp = [0.01, 0.1, 1.0]
    clf = SGDClassifier(loss='log', shuffle=True)

    # Get information on what the classifier learned
    display_important_features(feature_names, target_names, clf)

    # Calculate the classifier metrics
    cv = ShuffleSplit(len(data_stripped), n_iter=10,
                      test_size=0.7, random_state=42)
    cms, f1_scores = classifier_metrics(transformed, categories, clf, cv)
    classifier_scores(f1_scores)
    output_confusionmatrix(cms, categories, target_names)

    # Train Model on full dataset
    clf.fit(transformed, categories)
    # Get probablities of each class
    print('****Class Probabilities****')
    df_p = probs_class(patentdict, transformed, clf)
    print(df_p.head())
