import math

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from als_test import read


def fitness(train, subset, targets, models, standardisation, k):
    for target in targets:
        train_ = train[subset + [target]].copy()
        # train_ = train_.loc[train_[target] > 0]
        # print(train_.shape)
        X_train, y_train = train_.drop(columns=[target]), train_[target]
        max_score, max_model, kfold_scores = 10000, None, []
        all_y_val_overall = []
        all_y_pred_overall = []
        for model in models:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            scores = []
            all_y_val = []
            all_y_pred = []
            for train_index, val_index in kf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
                if standardisation:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_val = scaler.transform(X_val)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                score = math.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(score)
                all_y_val.extend(y_val)
                all_y_pred.extend(y_pred)
            avg_score = np.mean(scores)
            if max_score > avg_score:
                max_score = avg_score
                max_model = model.__class__.__name__
                kfold_scores = scores
                all_y_val_overall = all_y_val
                all_y_pred_overall = all_y_pred
        print(max_model, max_score, min(kfold_scores), max(kfold_scores), len(subset))
        rmse = math.sqrt(mean_squared_error(all_y_val_overall, all_y_pred_overall))
        rsquared_adj = (1 - (1 - r2_score(all_y_val_overall, all_y_pred_overall)) *
                        (len(all_y_val_overall) - 1) / (len(all_y_val_overall) - X_train.shape[1] - 1))
        rsquared = r2_score(all_y_val_overall, all_y_pred_overall)
        pcc = np.corrcoef(all_y_val_overall, all_y_pred_overall)[0, 1]
        print(f"{target} RMSE: {rmse}, rsquared: {rsquared}, rsquared_adj: {rsquared_adj}, PCC: {pcc}")


if __name__ == '__main__':
    train_df = read(filename="new_als_train")
    train_df = train_df.loc[train_df['Survived'] == True]
    target_feature = ['ALSFRS T3', 'ALSFRS T6', 'ALSFRS T9', 'ALSFRS T12']
    fold_number = 10
    std = True
    removal = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'Survived']
    scikit_models = [LinearRegression(), Ridge(random_state=42),
                     KNeighborsRegressor(weights='distance', algorithm='kd_tree', n_neighbors=int(math.sqrt(train_df.shape[0] / 10))),
                     DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42, n_estimators=50),
                     LGBMRegressor(verbosity=-1, random_state=42, n_estimators=50)]
    features = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                'Q5 Cutting', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Symptom Duration',
                'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'mitos movement', 'kings niv',
                'kings total', 'decline rate']
    print("alsfrs DE selection:")
    fitness(train=train_df, subset=features, targets=target_feature, models=scikit_models,
            standardisation=std, k=fold_number)
