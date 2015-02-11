"""Module to choose my features based on logistic regression;
I used statsmodels first, sklearn after"""

import pandas as pd
from McNulty_read_sql import import_df
from McNulty_helper import clean_question_marks
from McNulty_helper import zero_to_NaN
from McNulty_helper import change_types
from McNulty_helper import NaN_to_modes
from McNulty_helper import reduce_diagnosis
from McNulty_helper import hospital_to_number
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import Logit
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


if __name__ == '__main__':

    sns.set()

    # This project requires a data set stored on the cloud
    raw_df = import_df('all_hospitals')
    raw_df = clean_question_marks(raw_df)
    zero_to_NaN(raw_df, 'chol_mg_dl')
    zero_to_NaN(raw_df, 'rest_bp')
    change_types(raw_df)
    df = NaN_to_modes(raw_df)
    df = reduce_diagnosis(df)
    men = df.loc[df['sex'] == 1.0, ]
    men['hospital'] = men['hospital'].apply(hospital_to_number)

    (train_arr, test_arr) = train_test_split(men)
    train = pd.DataFrame(train_arr, copy=True)
    test = pd.DataFrame(test_arr, copy=True)
    train.columns = men.columns
    test.columns = men.columns
    change_types(train)
    change_types(test)
    del train_arr
    del test_arr

    # Add categorical variables to the training dataframe
    cp = pd.get_dummies(train['chest_pain_type'])
    fbs = pd.get_dummies(train['fast_blood_sugar'])
    thal = pd.get_dummies(train['thal_defect'])
    hosp = pd.get_dummies(train['hospital'])
    ecg = pd.get_dummies(train['rest_ecg'])
    angina = pd.get_dummies(train['st_exercise_angina'])
    slope = pd.get_dummies(train['st_exercise_slope'])
    vessels = pd.get_dummies(train['colored_vessels'])
    cp.columns = ['cp1', 'cp2', 'cp3', 'cp_asym']
    fbs.columns = ['fbs0', 'fbs1']
    thal.columns = ['thal3', 'thal6', 'thal7']
    hosp.columns = ['cleveland', 'hungarian', 'switzerland', 'va']
    ecg.columns = ['ecg_norm', 'ecg_ST-T_abn', 'ecg_left_hyper']
    angina.columns = ['angina_yes', 'angina_no']
    slope.columns = ['up_slope', 'no_slope', 'down_slope']
    vessels.columns = ['zero_vess', 'one_vess', 'two_vess', 'three_vess']

    cp_test = pd.get_dummies(test['chest_pain_type'])
    fbs_test = pd.get_dummies(test['fast_blood_sugar'])
    thal_test = pd.get_dummies(test['thal_defect'])
    hosp_test = pd.get_dummies(test['hospital'])
    ecg_test = pd.get_dummies(test['rest_ecg'])
    angina_test = pd.get_dummies(test['st_exercise_angina'])
    slope_test = pd.get_dummies(test['st_exercise_slope'])
    vessels_test = pd.get_dummies(test['colored_vessels'])
    cp_test.columns = ['cp1', 'cp2', 'cp3', 'cp_asym']
    fbs_test.columns = ['fbs0', 'fbs1']
    thal_test.columns = ['thal3', 'thal6', 'thal7']
    hosp_test.columns = ['cleveland', 'hungarian', 'switzerland', 'va']
    ecg_test.columns = ['ecg_norm', 'ecg_ST-T_abn', 'ecg_left_hyper']
    angina_test.columns = ['angina_yes', 'angina_no']
    slope_test.columns = ['up_slope', 'no_slope', 'down_slope']
    vessels_test.columns = ['zero_vess', 'one_vess', 'two_vess', 'three_vess']

    train = pd.concat([train, cp, fbs, thal, hosp, ecg,
                       angina, slope, vessels], axis=1)
    train['age'] = scale(train['age'])
    test['age'] = scale(test['age'])

    test = pd.concat([test, cp_test, fbs_test, thal_test, hosp_test,
                      ecg_test, angina_test, slope_test, vessels_test],
                     axis=1)

    # Run Log Regression on non-Stress-test features
    print "\nRun Logistic Regression withOUT stress-test"
    X = train[['age', 'cp2', 'cp_asym', 'thal3', 'hungarian',
               'switzerland', 'va']]
    X['Ones'] = 1.0
    Y = train['diagnosis']
    model = Logit(Y, X).fit()
    print model.summary()

    X2 = train[['age', 'cp1', 'cp2', 'cp3', 'thal3', 'hungarian',
                'switzerland', 'cleveland', 'angina_yes', 'no_slope',
                'down_slope', 'one_vess', 'two_vess', 'three_vess']]
    X2['Ones'] = 1.0

    corr_matrix = X2.corr()
    # plt.figure()
    # sm.graphics.plot_corr(corr_matrix, xnames=list(X2.columns.values))
    plt.ion()
    plt.show()

    print "\nRun Logistic Regression WITH stress-test"
    model2 = Logit(Y, X2).fit()
    print model2.summary()

    # Use sklearn to select the best features
    train_with_dummies = train.drop(['chest_pain_type', 'fast_blood_sugar',
                                     'thal_defect', 'hospital', 'rest_ecg',
                                     'st_exercise_angina', 'st_exercise_slope',
                                     'colored_vessels',
                                     'cp1', 'angina_no', 'va', 'down_slope',
                                     'patient_id', 'fbs0', 'three_vess'],
                                    axis=1)
    k_best = SelectKBest(f_regression, k=20).fit(train_with_dummies, Y)
    chosen_ones = k_best.get_support()
    X2_new = train_with_dummies.columns[chosen_ones]
    for col in X2_new:
        print col

    # This is the list of 20 features that sklearn selects:
    # Index([u'age', u'chol_mg_dl', u'st_max_heart_rt_ach', u'st_depression',
    #        u'cp2', u'cp3', u'cp_asym', u'fbs1', u'thal3', u'thal7',
    #        u'cleveland', u'hungarian', u'switzerland', u'ecg_norm',
    #        u'ecg_ST-T_abn', u'angina_yes', u'up_slope', u'zero_vess',
    #        u'one_vess', u'two_vess'], dtype='object')
