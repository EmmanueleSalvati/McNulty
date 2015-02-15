# McNulty

Repository of my *McNulty* project for the Metis datascience bootcamp. Here I keep instructions on how to run the various modules.

Dataset: Heart Disease Dataset from UCI Machine Learning Repository, found at [this link](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

### To run the code
Some EDA plots: Make_Plots.ipynb, Make_Stress_Plots.ipynb
Logistic regression results: Log_Regression.ipynb

Feature selection:

    python McNulty_feature_selection.py

Find best classifier:

    python McNulty_find_best_classifiers.py

### Helper Modules
* McNulty_config.py: [ignored on github] to connect to the MySQL database on the cloud
* McNulty_read_sql.py: to read the dataframe from the cloud
* McNulty_helper.py: functions to clean and manipulate the dataframe.

### Notes
1. Got rid of the women: too few
2. cholesterol has NaN values in Switzerland
3. ECG at rest has an unusual distribution for Cleveland
4. Num vessels has all zeros in veterans and Hungary: discard
5. ECG not useful: discard
6. Discard chest pain: almost all patients are asymptomatic chest pain, which means that they probably get redirected to other departments in the hospitals.
7. Logistic regression is the best classifier, according to the ROC curves I get from McNulty_find_best_classifiers.py