"""Helper functions for McNulty"""

import numpy as np


def clean_question_marks(dataframe):
    """Takes a dataframe, replaces '?' with np.nan, returns the dataframe"""

    clean_df = dataframe.replace('?', np.nan)
    return clean_df


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
