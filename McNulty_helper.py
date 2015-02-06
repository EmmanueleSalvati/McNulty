"""Helper functions for McNulty"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns

titles = {'chest_pain_type': 'chest pain type'}


def clean_question_marks(dataframe):
    """Takes a dataframe, replaces '?' with np.nan, returns the dataframe"""

    clean_df = dataframe.replace('?', np.nan)
    return clean_df


def zero_to_NaN(dataframe, column):
    """Replace zeros with NaN's in specified columns"""

    dataframe[column].replace('0', np.nan, inplace=True)


def change_types(dataframe):
    """Takes a dataframe and changes the strings to floats or ints

    age INT,
    sex INT,
    chest_pain_type INT,
    rest_bp DOUBLE,
    chol_mg_dl DOUBLE,
    fast_blood_sugar INT,
    rest_ecg INT,
    st_max_heart_rt_ach DOUBLE,
    st_exercise_angina INT,
    st_depression DOUBLE,
    st_exercise_slope INT,
    colored_vessels DOUBLE,
    thal_defect INT,
    diagnosis INT,
    hospital VARCHAR(255))
    patient_id INT;"""

    for col in dataframe.columns:
        if not col == 'hospital':
            tmp_col = dataframe[col].astype(float)
            dataframe[col] = tmp_col

    return dataframe


def NaN_to_modes(df):
    """Input: a dataframe; output: the same dataframe, with NaN's replaced by
    the modes of the corresponding columns"""

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(df[df.columns[:14]])
    imp_df = imp.transform(df[df.columns[:14]])
    new_df = pd.DataFrame(imp_df[:, :], columns=df.columns[:14])
    new_df['hospital'] = df['hospital']
    new_df['patient_id'] = df['patient_id']

    return new_df


def select_features(X, Y):
    """Input: X = df['age', 'sex', '...']
              Y = df['diagnosis']
    Output: an array of list of features, Y; they are imputed of NaN's
    with the modes over the columns, instead."""

    impX = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    impY = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    impX.fit(X)
    impY.fit(Y)
    Xprime = impX.transform(X)
    Yprime = impY.transform(Y)

    return (Xprime, pd.Series(Yprime[0]))


def make_facet_plots(df, variable, seaborn_object):
    """Make the facet plots"""

    sns.set()
    g = seaborn_object.FacetGrid(df, col='hospital',
                                 sharey=False, size=4, aspect=1.)
    g.map(sns.barplot, str(variable))
    # g.set_xlabels(titles[str(variable)])


def train_classifiers(x, y):
    """x and y are the training sets, train the various classifiers;
    returns a tuple of all the classifiers trained:
    (LinearRegression,)"""

    linmodel = LogisticRegression()
    linmodel.fit(x, y)

    return linmodel


def get_range(column):
    """For a given column (a pd Series), returns a tuple min, max"""

    return min(column), max(column)

