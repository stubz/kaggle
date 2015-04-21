import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from scipy.stats import expon

data_train = pd.read_csv('train.csv', index_col='id')
data_test = pd.read_csv('test.csv',index_col = 'id')
sample = pd.read_csv('sampleSubmission.csv')

le = preprocessing.LabelEncoder()
le.fit(["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])

# check if there is any NA values
data_train.shape
data_train.dropna().shape
data_test.shape
data_test.dropna().shape

# descriptive statistics
train_smry = data_train.describe()
train_smry.iloc[:, :10]

X=data_train.ix[:, ~data_train.columns.isin(['target'])]
#y=data_train['target']
y=le.transform(data_train['target'])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# test 
#X_train = X_train[:1000]
#X_test = X_test[:100]
#y_train = y_train[:1000]
#y_test = y_test[:100]

tuned_parameters = {'kernel': ['rbf'], 'gamma': expon(scale=.1), 'C': expon(scale=100) }

clf = RandomizedSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1)
clf.fit(X_train, y_train)
print clf.best_estimator_
pred = clf.predict(X_test)
pred_label = le.inverse_transform(pred)

submission = DataFrame({'label':pred_label}, columns=['label'],index=np.arange(1,len(pred)+1))
submission['Class_1'] = np.zeros(len(pred))
submission['Class_1'][pred==1] = np.ones(len(pred==1))
submission['Class_2'] = np.zeros(len(pred))
submission['Class_2'][pred==2] = np.ones(len(pred==2))
submission['Class_3'] = np.zeros(len(pred))
submission['Class_3'][pred==3] = np.ones(len(pred==3))
submission['Class_4'] = np.zeros(len(pred))
submission['Class_4'][pred==4] = np.ones(len(pred==4))
submission['Class_5'] = np.zeros(len(pred))
submission['Class_5'][pred==5] = np.ones(len(pred==5))
submission['Class_6'] = np.zeros(len(pred))
submission['Class_6'][pred==6] = np.ones(len(pred==6))
submission['Class_7'] = np.zeros(len(pred))
submission['Class_7'][pred==7] = np.ones(len(pred==7))
submission['Class_8'] = np.zeros(len(pred))
submission['Class_8'][pred==8] = np.ones(len(pred==8))
submission['Class_9'] = np.zeros(len(pred))
submission['Class_9'][pred==9] = np.ones(len(pred==9))

submission = submission.drop('label', axis=1)
submission.to_csv('submission_svm.csv', index_label='id')


#

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
]

clf = SVC(gamma=2, C=1)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test.astype(str))
svm_pred = clf.predict(X_test)
# need to convert the results to str
# http://stackoverflow.com/questions/19820369/unable-to-solve-an-error-while-running-gridsearch
confusion_matrix(y_test.astype(str), svm_pred.astype(str))
# very poor fit. it mostly predicts as class 2

# http://qiita.com/sotetsuk/items/16ffd76978085bfd7628
## チューニングパラメータ
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['accuracy', 'precision', 'recall']

for score in scores:
    print '\n' + '='*50
    print score
    print '='*50

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score, n_jobs=-1)
    clf.fit(X_train, y_train)

    print "\n+ ベストパラメータ:\n"
    print clf.best_estimator_

    print"\n+ トレーニングデータでCVした時の平均スコア:\n"
    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)

    print "\n+ テストデータでの識別結果:\n"
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)



