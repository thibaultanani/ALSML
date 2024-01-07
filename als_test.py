import math
import os

import numpy as np
import pandas as pd
import shap

from lightgbm import LGBMRegressor
from sklearn.metrics import confusion_matrix, recall_score, r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeClassifier


def read(filename, separator=','):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


def concat_rows(group, drop_lst):
    result = group.copy()

    drop_lst = [x for x in drop_lst if x in result.columns]
    result = result.drop(columns=drop_lst)

    lst = [" T_3", " T_6", " T_9"]
    suffixes = [lst[i] for i in range(len(group) - 1)]
    res = result.iloc[0].copy()
    for i, suffix in enumerate(suffixes):
        row = result.iloc[i + 1].add_suffix(suffix)
        res = res.combine_first(row)

    return res


def flatten(l):
    return [item for sublist in l for item in sublist]


def dist_for_percentage(reals, preds, percentage):
    if len(reals) != len(preds):
        raise ValueError("Les listes doivent avoir la même longueur")

    n, dist = len(reals), 0

    while True:
        results = [abs(reals[i] - preds[i]) <= dist for i in range(n)]
        true_count = sum(results)
        true_percentage = true_count / n

        if true_percentage >= percentage:
            return dist

        dist += 0.01


def create_regre(filename, drops_, target_, features_, params):
    data = read(filename=filename)
    data = data.rename(columns={'Sex': 'Gender'})
    static = ["Subject ID", "Source", "ID", "ExID", "Symptom Duration", "Onset Spinal", "Onset Bulbar", "Age", "Gender",
              "Height", "Death Date", "Survived", "Survival", "ALSFRS T3", "ALSFRS T6", "ALSFRS T9", "ALSFRS T12",
              "Period"]
    if len(features_) != 0:
        data = data[features_ + drops_ + [*target_]]
    static = [x for x in static if x in data.columns]
    t3 = (data.loc[(data['Period'] == 1) | (data['Period'] == 2)]).groupby('ID').apply(concat_rows, drop_lst=static)
    t6 = (data.loc[(data['Period'] == 1) |
                   (data['Period'] == 2) | (data['Period'] == 3)]).groupby('ID').apply(concat_rows, drop_lst=static)
    t9 = (data.loc[(data['Period'] == 1) | (data['Period'] == 2) |
                   (data['Period'] == 3) | (data['Period'] == 4)]).groupby('ID').apply(concat_rows, drop_lst=static)
    datasets, periods = [t3, t6, t9], ["", "T_3", "T_9", "T_12"]
    for i in range(len(datasets)):
        datasets[i] = pd.merge(datasets[i], data.loc[data['Period'] == 1][static], on="ID", how='left')
        datasets[i] = datasets[i].loc[datasets[i]['Survived'] == 1]
    data = data.loc[(data['Period'] == 1) & (data['Survived'] == 1)]
    t0 = data.copy()
    datasets.insert(0, t0)
    period = ["T3", "T6", "T9", "T12"]
    m = 0
    model_ = None
    real_df, pred_df = [], []
    for i in range(len(datasets)):
        dev, val = datasets[i].loc[datasets[i]['Source'] == "proact"], \
                   datasets[i].loc[datasets[i]['Source'] == "exonhit"]
        dev, val = dev.drop(drops_, axis=1), val.drop(drops_, axis=1)
        dev = dev.astype(np.float64)
        val = val.astype(np.float64)
        X_train, X_test = dev.drop(target_, axis=1), val.drop(target_, axis=1)
        y_train, y_test = dev[target_].values, val[target_].values
        model_i = LGBMRegressor(**params, random_state=42)
        wrapper = MultiOutputRegressor(model_i)
        wrapper.fit(X_train, y_train)
        if i == 0:
            model_ = wrapper
        y_pred = wrapper.predict(X_test)
        n_feats = X_train.shape[1]
        for j in range(m, 5):
            if j != 4:
                reals, preds = [k[j] for k in y_test], [k[j] for k in y_pred]
                time = period[j]
            else:
                reals, preds = flatten([k[m:] for k in y_test]), flatten([k[m:] for k in y_pred])
                time = "Total"
            rmse = math.sqrt(mean_squared_error(y_true=reals, y_pred=preds))
            rsquared_total = 1 - (1 - r2_score(reals, preds)) * (len(reals) - 1) / (len(reals) - n_feats - 1)
            pcc = np.corrcoef(reals, preds)[0, 1]
            dist_75 = dist_for_percentage(reals=reals, preds=preds, percentage=0.75)
            dist_80 = dist_for_percentage(reals=reals, preds=preds, percentage=0.80)
            dist_85 = dist_for_percentage(reals=reals, preds=preds, percentage=0.85)
            dist_90 = dist_for_percentage(reals=reals, preds=preds, percentage=0.90)
            print(time + " RMSE    " + ":", "{:.3f}".format(rmse), " R²      " + ":", "{:.4f}".format(rsquared_total),
                  " PCC      " + ":", "{:.3f}".format(pcc),
                  " 75%:", "{:.2f}".format(dist_75), " 80%:", "{:.2f}".format(dist_80),
                  " 85%:", "{:.2f}".format(dist_85), " 90%:", "{:.2f}".format(dist_90))
        print("\n")
        m = m + 1
        real_df.append(y_test)
        pred_df.append(y_pred)

    return model_


def launch(train_, test_, model_, target_, name):
    X, y = train_.drop(target_, axis=1).values, train_[target_].values
    X_train, y_train = X, y
    X_test, y_test = test_.drop(target_, axis=1).values, test_[target_].values
    model_.fit(X_train, y_train)
    y_pred = model_.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    score = recall_score(y_true=y_test, y_pred=y_pred, average="macro")
    print(f'\n{name}:\nscore: {score}\n{matrix}')
    explainer = shap.Explainer(model, train_.drop(target_, axis=1))
    shap_values = explainer(test_.drop(target_, axis=1))
    shap.summary_plot(shap_values, test_.drop(target_, axis=1), plot_type="bar")
    shap.summary_plot(shap_values, test_.drop(target_, axis=1))
    return model_


def patient_info(patient_):
    if patient_[0] == 0.0:
        gender_ = "Woman"
    else:
        gender_ = "Man"
    print(f"Gender: {gender_}, Age: {patient_[1]}, Weight: {patient_[2]}, Height: {patient_[3]}, q2: {patient_[4]}, "
          f"q5: {patient_[5]}, q7: {patient_[6]}, ALSFRS: {patient_[7]}, Symptom Duration: {patient_[8]}, "
          f"Pulse: {patient_[9]}, Systolic Blood Pressure: {patient_[10]}")


if __name__ == '__main__':
    train = read(filename="als_train")
    test = read(filename="als_test")
    train, test = train.loc[train['Period'] == 1], test.loc[test['Period'] == 1]
    train = train.rename(columns={'Sex': 'Gender'})
    test = test.rename(columns={'Sex': 'Gender'})

    print("===1-year Survival Classification Model===")
    model = RidgeClassifier(class_weight="balanced")
    target = "Survived"
    metric = "recall"
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T12']
    features = ['Gender', 'Age', 'Weight', 'Height', 'Q2 Salivation', 'Q5 Cutting', 'Q7 Turning in Bed', 'ALSFRS',
                'Symptom Duration', 'Pulse', 'Systolic Blood Pressure']
    train, test = train.drop(drops, axis=1), test.drop(drops, axis=1)
    launch(train, test, model, target, "Without Selection")
    train, test = train[features + [target]], test[features + [target]]
    model1 = launch(train, test, model, target, "Differential Evolution")
    print("\n\n")

    print("===ALSFRS Regression Model===\n")
    p = {'learning_rate': 0.026518209103740586, 'boosting_type': 'goss', 'n_estimators': 300, 'metric': 'rmse',
         'colsample_bytree': 0.5156322687606358, 'num_leaves': 175, 'subsample': 0.8155352104051389,
         'max_depth': 4, 'min_child_samples': 53, 'verbose': -1}
    model2 = create_regre(filename="dataset",
                          drops_=['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survived'],
                          target_=["ALSFRS T3", "ALSFRS T6", "ALSFRS T9", "ALSFRS T12"], features_=features, params=p)
    print("\n\n")

    gender, age, weight, height = 0, 55, 75, 175
    q2, q5, q7, alsfrs = 4, 4, 4, 30
    symptom, pulse, sbp = 22.53, 77.18, 131.71
    patient = np.array([gender, age, weight, height, q2, q5, q7, alsfrs, symptom, pulse, sbp])

    print("===Testing With A New Individual===\n")
    patient_info(patient)
    print("Test 1-year Survival:", model1._predict_proba_lr(patient.reshape(1, -1)))
    print("Test Disease progression:", model2.predict(patient.reshape(1, -1)))
