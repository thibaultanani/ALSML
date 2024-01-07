import os
import pickle
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import umap

from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler


def read(filename_, separator=','):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename_))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


def patient_info(patient_):
    if patient_[0] == 0.0:
        gender_ = "Woman"
    else:
        gender_ = "Man"
    print(f"Gender: {gender_}, Age: {patient_[1]}, Weight: {patient_[2]}, Height: {patient_[3]}, q2: {patient_[4]}, "
          f"q5: {patient_[5]}, q7: {patient_[6]}, ALSFRS: {patient_[7]}, Symptom Duration: {patient_[8]}, "
          f"Pulse: {patient_[9]}, Systolic Blood Pressure: {patient_[10]}, Forced Vital Capacity {patient_[11]}")


def launch(train_, model_, target_):
    X, y = train_.drop(target_, axis=1).values, train_[target_].values
    X_train, y_train = X, y
    model_.fit(X_train, y_train)
    return model_


def launch_regre(train_, model_, target_):
    X, y = train_.drop(target_, axis=1), train_[target_].values
    X_train, y_train = X, y
    wrapper = MultiOutputRegressor(model_)
    wrapper.fit(X_train, y_train)
    return wrapper


def part(n, x):
    lst = [0]
    res = 0
    for i in range(x):
        res = res + (n / x)
        lst.append(round(res, 2))
    return lst


def init_matrix_nan(m, n):
    mat = np.empty((m, n))
    mat[:] = np.nan
    return mat


def launch_umap(train_, target_, division_):
    X, y = train_.drop(target_, axis=1), train_[target_].values
    dev_X = X
    dev_X = scaler.fit_transform(dev_X)
    dev_X = pd.DataFrame(dev_X, columns=X.columns)
    trans = umap.UMAP(n_neighbors=15, metric="euclidean", n_components=2, random_state=42, n_jobs=1)
    trans.fit(dev_X)
    projected = trans.transform(dev_X)
    scaler_ = MinMaxScaler()
    projected = scaler_.fit_transform(projected)
    colors = ['dodgerblue', 'salmon']
    plt.scatter(projected[:, 0], projected[:, 1], c=y, s=5, cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(target)
    plt.show()
    plt.close()

    x_axis, y_axis = part(1, division_), part(1, division_)
    survival_matrix = init_matrix_nan(division_, division_)
    x_values, y_values, classes = projected[:, 0], projected[:, 1], y
    unique, counts = np.unique(classes, return_counts=True)
    res = counts[1] / counts[0]
    for i in range(len(x_axis) - 1):
        for j in range(len(y_axis) - 1):
            tmp_indices = [
                idx for idx in range(len(x_values)) if (
                        x_axis[i] <= x_values[idx] <= x_axis[i + 1] and
                        y_axis[j] <= y_values[idx] <= y_axis[j + 1]
                )
            ]
            if len(tmp_indices) < 5:
                survival_rate = np.nan
            else:
                weights = [1 if classes[idx] == 1 else res for idx in tmp_indices]
                survival_rate = np.average([classes[idx] for idx in tmp_indices], weights=weights) * 100
                # survival_rate = np.average([classes[idx] for idx in tmp_indices]) * 100
            survival_matrix[i][j] = survival_rate
    plt.pcolormesh(survival_matrix.transpose(), cmap='RdYlGn', edgecolors='k', linewidth=0.5)
    plt.colorbar(aspect=50, fraction=0.05)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xticks(np.linspace(0, len(survival_matrix[0]), 5))
    ax.set_yticks(np.linspace(0, len(survival_matrix[0]), 5))
    plt.tight_layout()
    plt.show()
    plt.close()

    return projected[:, 0], projected[:, 1], y, scaler, scaler_, trans, survival_matrix.T


if __name__ == '__main__':
    filename = "dataset"
    folder = "ml_models"
    dataset = read(filename_=filename)
    dataset = dataset.rename(columns={'Sex': 'Gender'})
    dataset = dataset.loc[dataset['Period'] == 1]

    print("===New Patient Information===")
    gender, age, weight, height = 0, 22, 75, 175
    q2, q5, q7, alsfrs = 4, 4, 4, 30
    symptom, pulse, sbp = 22.53, 77.18, 131.71
    fvc = 1.5
    patient = np.array([gender, age, weight, height, q2, q5, q7, alsfrs, symptom, pulse, sbp, fvc])
    patient_info(patient)
    print("\n")

    print("===1-year Survival Classification Model===")
    model = RidgeClassifier(class_weight="balanced")
    target = "Survived"
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T12']
    features = ['Gender', 'Age', 'Weight', 'Height', 'Q2 Salivation', 'Q5 Cutting', 'Q7 Turning in Bed', 'ALSFRS',
                'Symptom Duration', 'Pulse', 'Systolic Blood Pressure']
    dataset_survival = dataset.drop(drops, axis=1)
    dataset_survival = dataset_survival[features + [target]]
    model1 = launch(dataset_survival, model, target)
    with open(os.path.join(folder, 'als_model.pkl'), 'wb') as file:
        pickle.dump(model1, file)
    print("Test 1-year Survival:", model1._predict_proba_lr(patient[:-1].reshape(1, -1)))
    print("\n")

    print("===ALSFRS Regression Model===")
    p = {'learning_rate': 0.026518209103740586, 'boosting_type': 'goss', 'n_estimators': 300, 'metric': 'rmse',
         'colsample_bytree': 0.5156322687606358, 'num_leaves': 175, 'subsample': 0.8155352104051389,
         'max_depth': 4, 'min_child_samples': 53, 'verbose': -1}
    model = LGBMRegressor(**p, random_state=42)
    target = ["ALSFRS T3", "ALSFRS T6", "ALSFRS T9", "ALSFRS T12"]
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survived']
    dataset_alsfrs = dataset.drop(drops, axis=1)
    dataset_alsfrs = dataset_alsfrs[features + [*target]]
    model2 = launch_regre(dataset_alsfrs, model, target)
    with open(os.path.join(folder, 'als_model_reg.pkl'), 'wb') as file:
        pickle.dump(model2, file)
    print("Test Disease progression:", model2.predict(patient[:-1].reshape(1, -1)))
    print("\n\n")

    print("===UMAP Model===")
    scaler = MinMaxScaler()
    features = ['Forced Vital Capacity', 'Symptom Duration', 'ALSFRS', 'Height', 'Age', 'Weight', 'Pulse']
    target = "Survived"
    dataset_umap = dataset[features + [target]]
    division = 15
    model3 = launch_umap(dataset_umap, target, division_=division)
    with open(os.path.join(folder, 'als_model_umap.pkl'), 'wb') as file:
        pickle.dump(model3, file)
    fig = px.imshow(model3[6], color_continuous_scale='RdYlGn', origin='lower')
    x_grid = np.arange(-0.5, model3[6].shape[1], 1)
    y_grid = np.arange(-0.5, model3[6].shape[0], 1)
    fig.update_layout(
        plot_bgcolor='white'
    )
    for x_ in x_grid:
        fig.add_shape(
            type='line',
            x0=x_,
            x1=x_,
            y0=-0.5,
            y1=model3[6].shape[0] - 0.5,
            line=dict(color='black', width=1)
        )
    for y_ in y_grid:
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=model3[6].shape[1] - 0.5,
            y0=y_,
            y1=y_,
            line=dict(color='black', width=1)
        )
    val = pd.DataFrame(np.array([[fvc, symptom, alsfrs, height, age, weight, pulse]]),
                       columns=['Forced Vital Capacity', 'Symptom Duration', 'ALSFRS', 'Height', 'Age', 'Weight',
                                'Pulse'])
    val_X = model3[3].transform(val)
    projected_new = model3[5].transform(val_X)
    projected_new = model3[4].transform(projected_new)
    print("Test 1-year Survival (UMAP):", projected_new)
    range_min, range_max = 0, division
    projected_new = [[value[0] * (range_max - range_min) + range_min, value[1] * (range_max - range_min) + range_min]
                     for value in projected_new]
    for coord in projected_new:
        fig.add_shape(
            type='line',
            x0=coord[0] - 0.5,
            x1=coord[0] + 0.5,
            y0=coord[1] - 0.5,
            y1=coord[1] + 0.5,
            line=dict(color='black', width=5)
        )
        fig.add_shape(
            type='line',
            x0=coord[0] - 0.5,
            x1=coord[0] + 0.5,
            y0=coord[1] + 0.5,
            y1=coord[1] - 0.5,
            line=dict(color='black', width=5)
        )
    fig.show()

