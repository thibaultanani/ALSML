import os
import pickle
import gzip
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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


if __name__ == '__main__':
    train = read(filename_="new_als_train")
    test = read(filename_="new_als_test")
    dataset = pd.concat([train, test], ignore_index=True, sort=False)
    folder = "ml_models"

    print("===New Patient Information===") # ID: 209
    fvc, decline_rate, pulse = 2.75, 0.212765957446809, 76
    gender, age, weight, height = True, 65, 74, 168
    q1, q2, q3, q5, q6, q7 = 4, 4, 4, 3, 2, 3
    kings_total, kings_niv, dbp = 2, 2, 110
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
    dataset_survival = dataset.drop(drops, axis=1)
    dataset_survival = dataset_survival[features + [target]]
    model1, scaler1 = launch(dataset_survival, model, target)
    with open(os.path.join(folder, 'als_model.pkl'), 'wb') as file:
        pickle.dump(model1, file)
    with open(os.path.join(folder, 'als_model_scaler.pkl'), 'wb') as file:
        pickle.dump(scaler1, file)
    scaled_patient = scaler1.transform(patient.reshape(1, -1))
    print(scaled_patient)
    print("Test 1-year Survival:", model1._predict_proba_lr(scaled_patient))
    print("\n")

    print("===ALSFRS Regression Model===")
    model = RandomForestRegressor(random_state=42)
    target = ["ALSFRS T3", "ALSFRS T6", "ALSFRS T9", "ALSFRS T12"]
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'Survived']
    dataset_alsfrs = dataset.drop(drops, axis=1)
    dataset_alsfrs = dataset_alsfrs[features + [*target]]
    model2, scaler2 = launch_regre(dataset_alsfrs, model, target)
    with gzip.open(os.path.join(folder, 'als_model_reg.pkl.gz'), 'wb') as file:
        pickle.dump(model2, file)
    print("Test Disease progression:", model2.predict(scaled_patient))
    print("\n\n")

