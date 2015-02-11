"""Module to train classifiers"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale

from McNulty_feature_selection import retrieve_dataframe
from McNulty_feature_selection import dummify
from McNulty_feature_selection import train_test

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


best_columns = ['age', 'chol_mg_dl', 'st_max_heart_rt_ach', 'st_depression',
                'cp2', u'cp3', u'cp_asym', u'fbs1', u'thal3', u'thal7',
                'cleveland', u'hungarian', u'switzerland', u'ecg_norm',
                'ecg_ST-T_abn', u'angina_yes', u'up_slope', u'zero_vess',
                'one_vess', 'two_vess', 'diagnosis']

models_dict = {LogisticRegression: "LOGISTIC REGRESSION",
               KNeighborsClassifier: "K-NEAREST-NEIGHBORS",
               GaussianNB: 'NAIVE BAYES',
               SVC: 'SUPPORT VECTOR MACHINE',
               RandomForestClassifier: 'RANDOM FOREST'}


def train_knn(x, y, testx, testy):
    """train a KNN for different values of k and return the one with highest
    accuracy"""

    neighs = []
    for k in range(1, 11):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x, y)
        neighs.append(neigh)

    acc_scores = []
    for neigh in neighs:
        acc_score = accuracy_score(testy, neigh.predict(testx))
        acc_scores.append(acc_score)

    max_score = max(acc_scores)
    best_k = acc_scores.index(max_score)

    return neighs[best_k]


def class_report(x, y, model):
    """print accuracy and classification reports for the given model"""

    print models_dict[type(model)]
    print 'accuracy: %.2g' % accuracy_score(y, model.predict(x))
    print 'cross val score: %.2g' % np.mean(cross_val_score(
                                            model, x, y, cv=10))
    report = classification_report(np.array(y), model.predict(x))
    print report


def classifier_score(model, sample):
    """returns the probability of classifier model"""

    knn = (type(model) == KNeighborsClassifier)
    lm = (type(model) == LogisticRegression)
    gnb = (type(model) == GaussianNB)
    svc = (type(model) == SVC)
    rfc = (type(model) == RandomForestClassifier)
    if knn or lm or gnb or rfc:
        return model.predict_proba(sample)[:, 1]
    elif svc:
        return model.decision_function(sample)[:, 0]


def plot_roc(x, y, models):
    """Plot the ROC for the various classifiers;
    x, y are the test sample, models is a list of classifiers"""

    plt.figure()
    plt.ylim(ymax=1.1)
    plt.xlim(xmin=-0.1)
    for model in models:
        y_score = classifier_score(model, x.astype(float))
        fp, tp, thresh = roc_curve(y, y_score)
        area = roc_auc_score(y, y_score)
        label = models_dict[type(model)] + ' area: %.2g' % area
        plt.plot(fp, tp, label=label)

    plt.legend(loc=4)
    plt.ion()
    plt.show()


if __name__ == '__main__':

    raw_men = retrieve_dataframe()
    dummified_men = dummify(raw_men)
    men = dummified_men[best_columns]
    (train, test) = train_test(men)

    # X = men.loc[:, best_columns[:-1]]
    # Y = men['diagnosis']
    # skf = StratifiedKFold(Y, n_folds=10)

    X = train.loc[:, best_columns[:-1]]
    Y = train['diagnosis']
    x = test.loc[:, best_columns[:-1]]
    y = test['diagnosis']

    lm = LogisticRegression(random_state=1)
    knn = train_knn(X, Y, x, y)
    gnb = GaussianNB()
    svc = SVC()
    rfc = RandomForestClassifier(random_state=1)

    lm.fit(X, Y)
    gnb.fit(X, Y)
    svc.fit(X, Y)
    rfc.fit(X, Y)

    class_report(x, y, lm)
    # class_report(x, y, knn)
    # class_report(x, y, gnb)
    # class_report(x, y, svc)
    class_report(x, y, rfc)

    models = [lm, knn, gnb, svc, rfc]
    plot_roc(x, y, models)
