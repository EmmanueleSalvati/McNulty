#### To import the dataframe into memory and clean it
Fire up ipython and do:

    import McNulty_helper
    from McNulty_read_sql import import_df
    raw_df = import_df('all_hospitals')
    raw_df = McNulty_helper.clean_question_marks(raw_df)
    McNulty_helper.zero_to_NaN(raw_df, 'chol_mg_dl')
    McNulty_helper.zero_to_NaN(raw_df, 'rest_bp')
    McNulty_helper.change_types(raw_df)
    df = McNulty_helper.NaN_to_modes(raw_df)
    df = McNulty_helper.reduce_diagnosis(df)
    men = df.loc[df['sex']==1.0,]
**Note: I got rid of the women from the dataframe because there are too few of them. And I also reduced the diagnosis from 1->4 to 1**

#### Make 1D histograms, or bar charts
    %matplotlib inline
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

###### blood pressure
    g = sns.FacetGrid(men, col="hospital", sharey=False, size=4, aspect=1.)
    g.map(plt.hist, 'rest_bp', histtype='bar', align='mid')
    g.set_xlabels('blood pressure at rest')
the variables to plot are:
* chol\_mg\_dl
* age
* chest_pain_type
* fast_blood_sugar
* thal_defect

###### 2D plots
    g = sns.PairGrid(men, vars=['rest_bp', 'chol_mg_dl', 'age'])
    g.map_offdiag(plt.scatter)

#### To train the classifiers

    X = df[['age', 'sex', 'rest_bp', 'st_max_heart_rt_ach']]
    Y = df['diagnosis']
    xprime, yprime = McNulty_helper.select_features(X, Y)
    linear_model = McNulty_helper.train_classifiers(xprime, yprime)

