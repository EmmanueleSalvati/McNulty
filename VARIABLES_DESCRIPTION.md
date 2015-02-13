# McNulty

### Variables in the database
#### CONTINUOS:
* [0] age
* [3] rest_bp (mm Hg on admission to hospital)
* [4] chol_mg_dl
* [7] st_max_heart_rt_ach
* [9] st_depression (induced by exercise relative to rest)
* [11] colored_vessels (0 - 3; colored by flouroscopy)

#### CATEGORICAL:
* [1] sex (1 = male, 0 = female)
* [2] chest_pain_type (angina: 1 = typical, 2 = atypical, 3 = non-anginal, 4 = asymptomatic)
* [5] fast_blood_sugar (> 120 mg/dl: 1 = True, 0 = False)
* [6] rest_ecg (0 = normal, 1 = ST-T wave abnormality, 2 = probable or definite left ventrical hypertrophy)
* [8] st_exercise_angina (1 = yes, 0 = no)
* [10] st_exercise_slope (1 = unsloping, 2 = flat, 3 = downsloping)
* [12] thal_defect (3 = normal, 6 = fixed defect; 7 reversible defect)
* [13] diagnosis (0 = < 50% diameter narrowing, 1 or higher = > 50% diameter narrowing)
* [14] hospital
* [15] patient_id (PRIMARY KEY)

### Variables description
* age
* sex: 1 = male; 0 = female
* (cp, chest_pain_type)
    * Value 1: typical angina
    * Value 2: atypical angina
    * Value 3: non-anginal pain
    * Value 4: asymptomatic
* (trestbps, rest_bp)  resting blood pressure
* (chol, chol_mg_dl)
* (fbs, fast_blood_sugar) (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
* (restecg, rest_ecg)
    * Value 0: normal
    * Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    * Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
* (thalach, st_max_heart_rt_ach) maximum heart rate achieved
* (exang, st_exercise_angina) exercise induced angina (1 = yes; 0 = no)
* (oldpeak, st_depression) ST depression induced by exercise relative to rest
* (slope, st_exercise_slope) the slope of the peak exercise ST segment
    * Value 1: upsloping
    * Value 2: flat
    * Value 3: downsloping
* (ca, colored_vessels) number of major vessels (0-3) colored by flourosopy
* (thal, thal_defect) 3 = normal; 6 = fixed defect; 7 = reversable defect
* (num, diagnosis)       (the predicted attribute)
    * Value 0: < 50% diameter narrowing
    * Value 1: > 50% diameter narrowing
* (hospital)
    * Value 0: cleveland
    * Value 1: hungarian
    * Value 2: switzerland
    * Value 4: va