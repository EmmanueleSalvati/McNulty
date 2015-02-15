"""Train the two final logistic regressions"""

import pandas as pd
import numpy as np


from McNulty_feature_selection import train_test
from McNulty_feature_selection import retrieve_dataframe
from McNulty_feature_selection import dummify

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score


all_columns = ['age', 'chol_mg_dl', 'st_max_heart_rt_ach', 'st_depression',
               'fbs1', 'thal6', 'thal7',
               'cleveland', u'hungarian', u'switzerland', u'ecg_norm',
               'ecg_ST-T_abn', u'angina_yes', u'down_slope', u'zero_vess',
               'one_vess', 'two_vess', 'diagnosis']

non_ST_columns = ['age', 'chol_mg_dl', 'fbs1', 'thal6', 'thal7', 'cleveland',
                  'hungarian', 'switzerland', 'angina_yes', 'diagnosis']


if __name__ == '__main__':

    raw_men = retrieve_dataframe()
    dummified_men = dummify(raw_men)
    all_men = dummified_men[all_columns]
    non_ST_men = dummified_men[non_ST_columns]

    lm_all = LogisticRegression(random_state=1)
    lm_non_ST = LogisticRegression(random_state=1)

    X_all = all_men.loc[:, all_columns[:-1]]
    X_non_ST = non_ST_men.loc[:, non_ST_columns[:-1]]
    Y = all_men['diagnosis']

    lm_all.fit(X_all, Y)
    lm_non_ST.fit(X_non_ST, Y)

    non_ST_score = np.mean(cross_val_score(lm_non_ST, X_non_ST, Y, cv=10))
    all_score = np.mean(cross_val_score(lm_all, X_all, Y, cv=10))

    print "------- No Stress Test: %.5g" % non_ST_score
    print classification_report(np.array(Y), lm_non_ST.predict(X_non_ST))
    print "------- With Stress Test: %.5g" % all_score
    print classification_report(np.array(Y), lm_all.predict(X_all))
