# coding: utf-8
from __future__ import print_function, division, unicode_literals
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
import preprocess as pp
import classify as cl


def main():

    # Paths to files
    PATH_REPLACE = ''.join(['/Users/dpmlto1/Documents/Patent/',
                            'Thomson Innovation/',
                            'clustering/custom/replacements.json'])
    PATH_SW = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                       'clustering/custom/combined-stop-words.txt'])
    PATH_DATA = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                         'clustering/data/new-summaries/'])

    # Get the training and testing data
    patentdict = pp.loadClassifiedData(PATH_DATA)
    # Preprocess, Tokenize, and Vectorize
    pv = pp.PatentVectorizer(patentdict['data'], PATH_REPLACE=PATH_REPLACE,
                             PATH_SW=PATH_SW)
    vectordict = pv.patent_totfidf()
    vectorfile = cl.pickle_vectorvocab(vectordict['tfidf_instance'])
    # Properties of the vectorized data
    pv.vector_characteristics(patentdict)
    print('\n***Principal Component Analysis***\n')
    pv.pca_metric(patentdict)

    # Train the classifier
    # clf = SGDClassifier(loss='log', shuffle=True)
    # clf = RandomForestClassifier(n_estimators=100)
    # clf = MultinomialNB(alpha=0.001)
    # clf = LogisticRegression(C=500.0)
    # clf = SVC(C=1.0, kernel=str('linear'), gamma=1.0, probability=True)
    estimator_type = MultinomialNB
    # estimator_params = {'C': np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]),
                        # 'penalty': ['l1', 'l2']}
    estimator_params = {'alpha': [0.001, 0.005, 0.01, 0.1, 1]}
    gs_clf = GridSearchCV(estimator_type(), estimator_params,
                          scoring='f1', cv=5)
    gs_clf.fit(vectordict['tfidf_vectors'].toarray(), patentdict['target'])
    # gs_clf.fit(vectordict['tfidf_vectors'], patentdict['target'])

    # Pickle the best estimator
    clffile = cl.pickle_bestclassifier(gs_clf)

    print('\n***Grid Search Report***\n')
    print('***Estimator***\n', gs_clf, '\n')
    print('***Scores***\n')
    cl.display_grid_scores(gs_clf.grid_scores_, top=20)

    clf = estimator_type(**gs_clf.best_params_)
    # Get information on what the classifier learned
    try:
        cl.display_important_features(patentdict, vectordict, clf)
    except AttributeError:
        print('Class has no coef_ Property')
    except ValueError:
        print('Class has no coef_ Property')

    # Calculate the classifier metrics
    cv = ShuffleSplit(len(patentdict['data']), n_iter=10,
                      test_size=0.5, random_state=42)
    cms, f1_scores, df_p = cl.classifier_metrics(patentdict,
                                                 vectordict, clf, cv)
    cl.classifier_scores(f1_scores, patentdict, vectordict, clf, cv)
    cl.calculate_roc(patentdict, vectordict, clf)
    cl.output_confusionmatrix(cms, patentdict)
    print('****Class Probabilities****')
    print(df_p.head())

    cl.patent_predict(vectorfile, clffile)


if __name__ == '__main__':
    main()
