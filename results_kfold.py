import math
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from als_test import read


def retrain_and_test(train, test, pulse, target, model, standardisation, model_name):
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    X_pulse = pulse.drop(columns=[target])
    y_pulse = pulse[target]
    if standardisation:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_pulse = scaler.transform(X_pulse)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm_test.ravel()
    score_test = balanced_accuracy_score(y_test, y_pred_test)
    print(f'\nEvaluation on test data for the {model_name} model (retrained): {score_test}')
    print(f'TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}')
    print(f'Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}, Specificity: {tn/(tn+fp)}')
    y_pred_pulse = model.predict(X_pulse)
    cm_pulse = confusion_matrix(y_pulse, y_pred_pulse)
    tn, fp, fn, tp = cm_pulse.ravel()
    score_pulse = balanced_accuracy_score(y_pulse, y_pred_pulse)
    print(f'\nEvaluation on pulse data for the {model_name} model (retrained): {score_pulse}')
    print(f'TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}')
    print(f'Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}, Specificity: {tn/(tn+fp)}')


def fitness(train, subset, target, models, metric, standardisation, k, test, pulse):
    train, test, pulse = train[subset + [target]], test[subset + [target]], pulse[subset + [target]]
    X_train, y_train = train.drop(columns=[target]), train[target]
    max_score, max_model, best_model_name, kfold_scores = -1, None, None, []
    all_y_val_overall = []
    all_y_pred_overall = []
    for model in models:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = []
        all_y_val = []
        all_y_pred = []
        for train_index, val_index in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            if standardisation:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = metric(y_val, y_pred)
            scores.append(score)
            all_y_val.extend(y_val)
            all_y_pred.extend(y_pred)
        avg_score = np.mean(scores)
        if max_score < avg_score:
            max_score = avg_score
            max_model = model
            best_model_name = model.__class__.__name__
            kfold_scores = scores
            all_y_val_overall = all_y_val
            all_y_pred_overall = all_y_pred
    cm = confusion_matrix(all_y_val_overall, all_y_pred_overall)
    tn, fp, fn, tp = cm.ravel()
    print(f"Model: {best_model_name}, Mean: {max_score}, Min: {min(kfold_scores)}, Max: {max(kfold_scores)},"
          f" Features: {len(train.columns) - 1}")
    print(f"TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}")
    print(f'Precision: {tp / (tp + fp)}, Recall: {tp / (tp + fn)}, Specificity: {tn / (tn + fp)}')
    retrain_and_test(train, test, pulse, target, max_model, standardisation, best_model_name)



if __name__ == '__main__':
    train_df, test_df, pulse_df = (read(filename="new_als_train"), read(filename="new_als_test"),
                                   read(filename="new_als_test_pulse"))
    pulse_df['Pulse'] = train_df['Pulse'].mean()
    pulse_df['Diastolic Blood Pressure'] = train_df['Diastolic Blood Pressure'].mean()
    pulse_df['Systolic Blood Pressure'] = train_df['Systolic Blood Pressure'].mean()
    target_feature = "Survived"
    removal = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T3', 'ALSFRS T6',
               'ALSFRS T9', 'ALSFRS T12']
    train_df = train_df.drop(columns=removal)
    test_df = test_df.drop(columns=removal)
    fold_number = 10
    scoring_metric = balanced_accuracy_score
    time_limit = 3600
    std = True
    scikit_models = [LogisticRegression(random_state=42, solver="lbfgs", class_weight="balanced", max_iter=10000),
                     KNeighborsClassifier(weights='distance', algorithm='kd_tree',
                                          n_neighbors=int(math.sqrt(train_df.shape[0] / fold_number))),
                     DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                     RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=50),
                     LGBMClassifier(class_weight="balanced", verbosity=-1, random_state=42, n_estimators=50),
                     SVC(class_weight="balanced", random_state=42, max_iter=5000, tol=0.01)]

    print("no selection")
    features_all = train_df.drop(columns=[target_feature]).columns.tolist()
    fitness(train=train_df, subset=features_all, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number, test=test_df, pulse=pulse_df)

    print("\n\nno derived features")
    features_no_derived = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                        'Q4 Handwriting', 'Q5 Cutting', 'Q5 Indic', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed',
                        'Q8 Walking', 'Q9 Climbing Stairs', 'Q10 Respiratory', 'ALSFRS', 'Symptom Duration',
                        'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'Systolic Blood Pressure']
    fitness(train=train_df, subset=features_no_derived, target=target_feature, models=scikit_models,
            metric=scoring_metric, standardisation=std, k=fold_number, test=test_df, pulse=pulse_df)

    print("\n\nanova")
    features_anova = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                      'Q4 Handwriting', 'Q5 Cutting', 'Q5 Indic', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed',
                      'Q8 Walking', 'Q9 Climbing Stairs', 'Q10 Respiratory', 'ALSFRS', 'Symptom Duration',
                      'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'bmi', 'bulbar score',
                      'upper limbs score', 'trunk score', 'lower limbs score', 'mitos movement', 'mitos swallowing',
                      'mitos communicating', 'mitos breathing', 'mitos total', 'kings bulbar', 'kings arm', 'kings leg',
                      'kings total', 'ft9 bulbar', 'ft9 fine motor', 'ft9 gross motor', 'ft9 respiratory', 'ft9 total',
                      'decline rate']
    fitness(train=train_df, subset=features_anova, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number, test=test_df, pulse=pulse_df)

    print("\n\ndifferential evolution")
    features_de = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                   'Q5 Cutting', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Symptom Duration',
                   'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'mitos movement', 'kings niv',
                   'kings total', 'decline rate']
    fitness(train=train_df, subset=features_de, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number, test=test_df, pulse=pulse_df)
