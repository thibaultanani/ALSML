import os
import pickle
import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


def read(filename_, separator=','):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename_))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


def launch(train_, model_, target_):
    X, y = train_.drop(target_, axis=1).values, train_[target_].values
    X_train, y_train = X, y
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model_.fit(X_train_scaled, y_train)
    return model_, scaler


def launch_regre(train_, model_, target_):
    X, y = train_.drop(target_, axis=1).values, train_[target_].values
    X_train, y_train = X, y
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    wrapper = MultiOutputRegressor(model_)
    wrapper.fit(X_train_scaled, y_train)
    return wrapper, scaler


def launch_cox(train_, time_, event_, target_):
    X = train_.drop(target_, axis=1)
    X_train = X
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_train_scaled['Time'], X_train_scaled['Event'] = time_, event_
    cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.0)
    cph.fit(X_train_scaled, duration_col='Time', event_col='Event')
    cph.print_summary()
    cph.plot()
    return cph, scaler

def find(row):
    if np.isnan(row['Death Date']) or row['Death Date'] > 13:
        return 13
    else:
        return row['Death Date']


def event_occurred(row):
    if np.isnan(row['Death Date']) or row['Death Date'] > 13:
        return False
    else:
        return True


if __name__ == '__main__':
    train = read(filename_="new_als_train")
    test = read(filename_="new_als_test")
    dataset = pd.concat([train, test], ignore_index=True, sort=False)
    folder = "ml_models"

    print("===New Patient Information===") # ID: 209
    fvc, decline_rate, pulse = 2.75, 0.212765957446809, 76
    gender, age, weight, height = True, 65, 74, 168
    q1, q2, q3, q5, q6, q7 = 4, 4, 4, 3, 2, 3
    kings_total, kings_niv, dbp = 2, 0, 110
    symptom, onset, mitos_movement = 47, True, 0
    patient = np.array([gender, age, weight, height, onset, q1, q2, q3, q5, q6, q7, symptom, fvc, pulse, dbp,
                        mitos_movement, kings_niv, kings_total, decline_rate])
    features = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                'Q5 Cutting', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Symptom Duration',
                'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'mitos movement', 'kings niv',
                'kings total', 'decline rate']
    patient_dict = dict(zip(features, patient))
    print(patient_dict)
    print("\n")

    print("===1-year Survival Classification Model===")
    model = LogisticRegression(random_state=42, class_weight='balanced', solver='lbfgs', max_iter=10000)
    target = "Survived"
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T3', 'ALSFRS T6',
             'ALSFRS T9', 'ALSFRS T12']
    time, event = dataset.apply(lambda row: find(row), axis=1), dataset.apply(lambda row: event_occurred(row), axis=1)
    dataset_survival = dataset.drop(drops, axis=1)
    dataset_survival = dataset_survival[features + [target]]
    model1, scaler1 = launch(dataset_survival, model, target)
    model2, scaler2 = launch_cox(dataset_survival, time, event, target)
    with open(os.path.join(folder, 'als_model.pkl'), 'wb') as file:
        pickle.dump(model1, file)
    with open(os.path.join(folder, 'als_model_scaler.pkl'), 'wb') as file:
        pickle.dump(scaler1, file)
    with open(os.path.join(folder, 'als_model_cox.pkl'), 'wb') as file:
        pickle.dump(model2, file)
    scaled_patient = scaler1.transform(patient.reshape(1, -1))
    print(scaled_patient)
    print("Test 1-year Survival (Logistic):", model1._predict_proba_lr(scaled_patient))
    log_hazard = model2.predict_log_partial_hazard(scaled_patient)
    print("Log hazard ratio:", log_hazard.values[0])
    hazard = model2.predict_partial_hazard(scaled_patient)
    print("Hazard ratio:", hazard.values[0])
    survival = model2.predict_survival_function(scaled_patient)
    survival.plot(title="Survival curves for the patient")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.show()
    print("\n")

    print("===ALSFRS Regression Model===")
    model = LGBMRegressor(verbosity=-1, random_state=42, n_estimators=50)
    target = ["ALSFRS T3", "ALSFRS T6", "ALSFRS T9", "ALSFRS T12"]
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'Survived']
    dataset_alsfrs = dataset.drop(drops, axis=1)
    dataset_alsfrs = dataset_alsfrs[features + [*target]]
    model2, scaler2 = launch_regre(dataset_alsfrs, model, target)
    joblib.dump(model2, os.path.join(folder, 'als_model_reg.joblib'), compress=3)
    print("Test Disease progression:", model2.predict(scaled_patient))
    print("\n\n")
