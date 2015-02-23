"""Run the final Logistic Regression"""

import numpy as np

from McNulty_feature_selection import retrieve_dataframe
from McNulty_feature_selection import dummify

from statsmodels.formula.api import Logit

# BEST_COLUMNS = ['age', 'chol_mg_dl', 'st_max_heart_rt_ach', 'st_depression',
#                 'fbs1', 'thal6', 'thal7',
#                 'cleveland', u'hungarian', u'switzerland', u'ecg_norm',
#                 'ecg_ST-T_abn', u'angina_yes', u'down_slope', u'zero_vess',
#                 'one_vess', 'two_vess', 'diagnosis']

BEST_COLUMNS = ['age', 'chol_mg_dl', 'fbs1', 'thal6', 'thal7',
                'cleveland', 'hungarian', 'switzerland', 'angina_yes',
                'zero_vess', 'one_vess', 'two_vess', 'diagnosis']


def create_param_dict(feature, coeff):
    """Returns a dictionary cholesterol: (beta * cholesterol)"""

    param_dict = {}
    for elem in feature:
        param_dict[elem] = elem * coeff

    return param_dict


def bin_dataset(array_to_bin, n):
    """Given an array, bin it in n bins"""

    from scipy.stats import binned_statistic

    min = np.min(array_to_bin)
    max = np.max(array_to_bin)
    bin_means = binned_statistic(array_to_bin, array_to_bin, bins=n,
                                 range=(min, max))[0]

    return bin_means


if __name__ == '__main__':
    raw_men = retrieve_dataframe()
    dummified_men = dummify(raw_men)
    men = dummified_men[BEST_COLUMNS]

    X = men[BEST_COLUMNS[:-1]]
    Y = men['diagnosis']

    X['Ones'] = 1.0
    model = Logit(Y, X).fit()
    print model.summary()

    # beta_cholesterol = model.params['chol_mg_dl']
    # beta_st_depression = model.params['age']
    # chol_dict = create_param_dict(X['chol_mg_dl'], beta_cholesterol)
    # st_depr_dict = create_param_dict(X['age'], beta_st_depression)

    # chol = np.array(X['chol_mg_dl'])
    # chol_array = bin_dataset(chol, 10)

    # age = np.array(X['age'])
    # age_array = bin_dataset(age, 10)

    prediction = {}
    for age in range(28, 78):
        for chol in range(85, 604):
            prediction[age, chol] = model.predict([age, chol, 1.])[0]

    import csv
    with open('predictions_full_noST.csv', 'w') as csvfile:
        predict_writer = csv.writer(csvfile)
        for age in range(28, 78):
            for chol in range(85, 604):
                predict_writer.writerow([age, chol, model.predict([age, chol, 1.])[0]])

