# coding: utf-8
from __future__ import print_function, division, unicode_literals
from collections import defaultdict
import os
import dill
import numpy as np
from scipy.stats import sem
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.externals import joblib


def display_important_features(patentdict, vectordict, clf, n_top=30):
    '''
    Display highest weighted and lowest weighted feastures for each category
    Parameters
        feature_names: tokens vectorized
        target_names: category names
        clf: classifier instance
        n_top: number of features to display
    '''
    clf.fit(vectordict['tfidf_vectors'].toarray(), patentdict['target'])
    weights = clf.coef_
    print('\n***Classifier Features***\n')
    for i, target_name in enumerate(patentdict['target_names']):
        print("Class: " + target_name)
        print("")

        sorted_features_indices = weights[i].argsort()[::-1]

        most_important = sorted_features_indices[:n_top]
        print(", ".join("{0}: {1:.4f}".format(vectordict['feature_names'][j],
                                              weights[i, j])
                        for j in most_important))
        print("...")

        least_important = sorted_features_indices[-n_top:]
        print(", ".join("{0}: {1:.4f}".format(vectordict['feature_names'][j],
                                              weights[i, j])
                        for j in least_important))
        print("")


def classifier_metrics(patentdict, vectordict, clf, cv):
    '''
    Calculate f1 scores, confusion matrix, class Probabilities
    for several cv splits
    Parameters
        transformed: X vector (TF-IDF for each feature/category)
        patentdict
        clf: classifier instance
        cv: cross validation instance
    Returns
        cms: confusion matrix averaged over cv runs
        f1_scores: f1 score for each cv run
        df_p: pandas dataframe with probabities for each class
    '''
    categories = patentdict['target']
    f1_scores = []
    cms = []
    df_p = pd.DataFrame()
    targetd = dict(zip(range(len(patentdict['target_names']) + 1),
                       patentdict['target_names']))
    for i, (train, test) in enumerate(cv):
        X_train, y_train = vectordict[
            'tfidf_vectors'][train], categories[train]
        X_test, y_test = vectordict['tfidf_vectors'][test], categories[test]
        clf.fit(X_train.toarray(), y_train)
        # train_score = clf.score(X_train, y_train)
        predicted = clf.predict(X_test.toarray())
        f1_scores.append(f1_score(y_test, predicted, average='weighted'))
        cms.append(confusion_matrix(y_test, predicted))
        df_p = df_p.append(category_probs(X_test, y_test, clf,
                                          patentdict['target_names'], targetd,
                                          patentdict['filenames'][test]))

    ordercols = (['category_name', 'category_num', 'files'] +
                 patentdict['target_names'])
    df_p = df_p.reindex_axis(ordercols, axis=1, copy=False)
    return cms, f1_scores, df_p


def category_probs(X, y, clf, target_names, targetd, filenames):
    proba = clf.predict_proba(X.toarray())
    df_t = pd.DataFrame(proba, columns=target_names)
    df_t['category_num'] = y
    df_t['category_name'] = df_t['category_num'].map(targetd)
    df_t['files'] = [os.path.splitext(os.path.basename(filename))[0]
                     for filename in filenames]
    return df_t


def classifier_scores(f1_scores, patentdict, vectordict, clf, cv):
    '''
    Prints f1 scores for cv runs and mean and std of  scores
    Parameter
        f1_scores: array of f1 scores for several cv runs
    '''
    print('****F1 Test Scores****\n')
    for f1score in f1_scores:
        print('\t {0:0.3f}'.format(f1score))
    print('Average: {0:0.3f}\nStd Dev: {1:0.4f}\n'.format(np.mean(f1_scores),
                                                          np.std(f1_scores)))


def output_confusionmatrix(cms, patentdict):
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
    df = pd.DataFrame(dict(zip(patentdict['target_names'], frac.transpose())),
                      columns=patentdict['target_names'])
    df['names'] = patentdict['target_names']
    df['# of documents'] = [patentdict['target'][patentdict['target'] == i].shape[0]
                            for i in range(len(patentdict['target_names']))]
    print(df)

    # Plot confusion matrix
    fig = plt.figure(figsize=(10., 10.))
    ax = fig.add_subplot(111)
    plt.matshow(frac, fignum=False, cmap='Blues', vmin=0., vmax=1.0)
    ax.set_xticks(range(len(patentdict['target_names'])))
    ax.set_xticklabels(patentdict['target_names'])
    ax.set_yticks(range(len(patentdict['target_names'])))
    ax.set_yticklabels(patentdict['target_names'])
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.colorbar()
    plt.show()


def calculate_roc(patentdict, vectordict, clf):
    categories = patentdict['target']
    labels = np.unique(categories)
    tprs = defaultdict(list)
    fprs = defaultdict(list)
    roc_scores = defaultdict(list)
    X_train, X_test, y_train, y_test = train_test_split(
        vectordict['tfidf_vectors'],
        patentdict['target'],
        test_size=0.5)
    proba = clf.predict_proba(X_test.toarray())
    for label in labels:
        y_label_test = np.asarray(y_test == label, dtype=int)
        proba_label = proba[:, label]
        fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
        roc_scores[label].append(auc(fpr, tpr))
        tprs[label].append(tpr)
        fprs[label].append(fpr)
        scores_to_sort = roc_scores[label]
        median = np.argsort(
            scores_to_sort)[int(len(scores_to_sort) / 2)]
        plot_roc(roc_scores[label][median], tprs[label][median],
                 fprs[label][median],
                 label='{} vs rest'.format(
                     patentdict['target_names'][label]))


def plot_roc(auc_score, tpr, fpr, label=None):
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


def display_scores(params, scores, append_star=False):
    """Format the mean score +/- std error for params"""
    params = ", ".join("{0}={1}".format(k, v)
                       for k, v in params.items())
    line = "{0}:\t{1:.3f} (+/-{2:.3f})".format(
        params, np.mean(scores), sem(scores))
    if append_star:
        line += " *"
    return line


def display_grid_scores(grid_scores, top=None):
    """Helper function to format a report on a grid of scores"""

    grid_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    if top is not None:
        grid_scores = grid_scores[:top]

    # Compute a threshold for staring models with overlapping
    # stderr:
    _, best_mean, best_scores = grid_scores[0]
    threshold = best_mean - 2 * sem(best_scores)

    for params, mean_score, scores in grid_scores:
        append_star = mean_score + 2 * sem(scores) > threshold
        print(display_scores(params, scores, append_star=append_star))


def pickle_withdill(tobepickled, filepath):
    with open(filepath, 'w') as f:
        pickledfile = dill.dump(tobepickled, f)
    return pickledfile


def unpickle_withdill(filepath):
    with open(filepath, 'r') as f:
        unpickled = dill.load(f)
    return unpickled


def pickle_bestclassifier(gs_clf):
    filepath = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                        'clustering/data/classifier.pkl'])
    clffile = joblib.dump(gs_clf.best_estimator_, filepath, compress=9)
    return clffile


def patent_predict(data, vector_filepath, clffile):
    clf = joblib.load(clffile[0])
    tfidf_instance = unpickle_withdill(vector_filepath)
    print(clf)
    print(tfidf_instance)
    X = tfidf_instance.transform(data)
    predicted_class = clf.predict(X)
    predicted = {'clf': clf,
                 'X': X,
                 'y': predicted_class}
    return predicted
