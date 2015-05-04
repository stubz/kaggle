import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing, feature_extraction
from scipy.stats import expon
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

if __name__ == "__main__":
    data_train = pd.read_csv('train.csv', index_col='id')
    data_test = pd.read_csv('test.csv',index_col = 'id')
    sample = pd.read_csv('sampleSubmission.csv')

    le = preprocessing.LabelEncoder()
    le.fit(["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])

    X=data_train.ix[:, ~data_train.columns.isin(['target'])]
    #y=data_train['target']
    y=le.transform(data_train['target'])
    # standardising the data 
    # X = StandardScaler().fit_transform(X)
    tfidf = feature_extraction.text.TfidfTransformer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    X_train = tfidf.fit_transform(X_train).toarray()
    X_test = tfidf.fit_transform(X_test).toarray()

    """
    X_train = X_train[:1000]
    X_test = X_test[:100]
    y_train = y_train[:1000]
    y_test = y_test[:100]
    """
    ##################################################################
    ### Modelling by Random Forest with parameter tuning
    ##################################################################
    # build a classifier
    clf = GradientBoostingClassifier()

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 30),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "n_estimators": sp_randint(1, 1000),}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)

    # predict on test set
    preds = random_search.predict_proba(X_test)

    # create submission file
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv('submission_gbm_tfidf.csv', index_label='id')

