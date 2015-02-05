## To create TABLES:

- table_names:

* hungarian_tidy
* switzerland_tidy
* cleveland_tidy
* va_tidy
* all_hospitals

#### create command for each:
    CREATE TABLE all_hospitals(
        age VARCHAR(255),
        sex VARCHAR(255),
        chest_pain_type VARCHAR(255),
        rest_bp VARCHAR(255),
        chol_mg_dl VARCHAR(255),
        fast_blood_sugar VARCHAR(255),
        rest_ecg VARCHAR(255),
        st_max_heart_rt_ach VARCHAR(255),
        st_exercise_angina VARCHAR(255),
        st_depression VARCHAR(255),
        st_exercise_slope VARCHAR(255),
        colored_vessels VARCHAR(255),
        thal_defect VARCHAR(255),
        diagnosis VARCHAR(255),
        hospital VARCHAR(255));

#### to LOAD for each table use apt csv:

* hungarian_tidy.csv
* switzerland_tidy.csv
* cleveland_tidy.csv
* va_tidy.csv
* all_hospitals.csv


LOAD DATA LOCAL INFILE "hungarian_tidy.csv"
   INTO TABLE hungarian_tidy FIELDS TERMINATED BY "," IGNORE 1 LINES;''

#### to ADD tables to one table ( example):

'' INSERT IGNORE
   INTO all_hospitals
SELECT *
   FROM va_tidy;''

#### to ADD PRIMARY KEY:

''ALTER TABLE all_hospitals
ADD COLUMN patient_id INT PRIMARY KEY AUTO_INCREMENT;''