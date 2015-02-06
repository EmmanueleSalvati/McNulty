"""Module to make plots"""

import McNulty_helper
from McNulty_read_sql import import_df
import seaborn as sns


if __name__ == '__main__':
    raw_df = import_df('all_hospitals')
    raw_df = McNulty_helper.clean_question_marks(raw_df)
    McNulty_helper.zero_to_NaN(raw_df, 'chol_mg_dl')
    McNulty_helper.zero_to_NaN(raw_df, 'rest_bp')
    McNulty_helper.change_types(raw_df)
    df = McNulty_helper.NaN_to_modes(raw_df)
    men = df.loc[df['sex'] == 1.0, ]

    McNulty_helper.make_facet_plots(men, 'chest_pain_type', sns)
