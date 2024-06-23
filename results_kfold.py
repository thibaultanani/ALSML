import numpy as np
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from als_test import read


def fitness(train, subset, target, models, metric, standardisation, k):
    train = train[subset + [target]]
    X_train, y_train = train.drop(columns=[target]), train[target]
    max_score, max_model, kfold_scores = -1, None, []
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
            max_model = model.__class__.__name__
            kfold_scores = scores
            all_y_val_overall = all_y_val
            all_y_pred_overall = all_y_pred
    cm = confusion_matrix(all_y_val_overall, all_y_pred_overall)
    tn, fp, fn, tp = cm.ravel()
    print(f"Model: {max_model}, Mean: {max_score}, Min: {min(kfold_scores)}, Max: {max(kfold_scores)},"
          f" Features: {len(train.columns) - 1}")
    print(f"True Positives: {tp}, False Negatives: {fn}, True Negatives: {tn}, False Positives: {fp}")


if __name__ == '__main__':
    train_df, test_df = read(filename="new_als_train"), read(filename="new_als_test")
    target_feature = "Survived"
    removal = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T3', 'ALSFRS T6',
               'ALSFRS T9', 'ALSFRS T12']
    train_df = train_df.drop(columns=removal)
    fold_number = 10
    scoring_metric = balanced_accuracy_score
    time_limit = 3600
    std = True
    scikit_models = [LogisticRegression(random_state=42, solver="lbfgs", class_weight="balanced", max_iter=10000),
                     RidgeClassifier(random_state=42, class_weight="balanced"),
                     KNeighborsClassifier(n_neighbors=10, weights='distance'),
                     DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                     RandomForestClassifier(random_state=42, class_weight='balanced'),
                     LGBMClassifier(class_weight="balanced", verbosity=-1, random_state=42),
                     LinearDiscriminantAnalysis(), GaussianNB()]

    print("no selection:")
    features_all = train_df.drop(columns=[target_feature]).columns.tolist()
    fitness(train=train_df, subset=features_all, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number)

    print("\ncorrelation:")
    features_corr = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                     'Q4 Handwriting', 'Q5 Cutting', 'Q5 Indic', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed',
                     'Q8 Walking', 'Q9 Climbing Stairs', 'Q10 Respiratory', 'ALSFRS', 'Symptom Duration',
                     'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'Systolic Blood Pressure', 'bmi',
                     'bulbar score', 'upper limbs score', 'trunk score', 'lower limbs score', 'mitos movement',
                     'mitos swallowing', 'mitos communicating', 'mitos breathing', 'mitos total', 'kings bulbar',
                     'kings arm', 'kings leg', 'kings niv', 'kings total', 'ft9 bulbar', 'ft9 fine motor',
                     'ft9 gross motor', 'ft9 respiratory', 'ft9 total', 'decline rate']
    fitness(train=train_df, subset=features_corr, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number)

    print("\nanova:")
    features_anova = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                      'Q4 Handwriting', 'Q5 Cutting', 'Q5 Indic', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed',
                      'Q8 Walking', 'Q9 Climbing Stairs', 'Q10 Respiratory', 'ALSFRS', 'Symptom Duration',
                      'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'bmi', 'bulbar score',
                      'upper limbs score', 'trunk score', 'lower limbs score', 'mitos movement', 'mitos swallowing',
                      'mitos communicating', 'mitos breathing', 'mitos total', 'kings bulbar', 'kings arm',
                      'kings leg', 'kings total', 'ft9 bulbar', 'ft9 fine motor', 'ft9 gross motor', 'ft9 respiratory',
                      'ft9 total', 'decline rate']
    fitness(train=train_df, subset=features_anova, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number)

    print("\nmutual info:")
    features_mi = ['Gender', 'Age', 'Weight', 'Height', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing', 'Q5 Cutting',
                   'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Q8 Walking', 'Q9 Climbing Stairs',
                   'Q10 Respiratory', 'ALSFRS', 'Symptom Duration', 'Forced Vital Capacity', 'Pulse', 'bmi',
                   'bulbar score', 'upper limbs score', 'trunk score', 'lower limbs score', 'mitos movement',
                   'mitos communicating', 'mitos total', 'kings leg', 'kings total', 'ft9 bulbar', 'ft9 gross motor',
                   'ft9 respiratory', 'ft9 total', 'decline rate']
    fitness(train=train_df, subset=features_mi, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number)

    print("\nmrmr")
    features_mrmr = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                     'Q4 Handwriting', 'Q5 Cutting', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Q8 Walking',
                     'Q9 Climbing Stairs', 'Q10 Respiratory', 'ALSFRS', 'Symptom Duration', 'Forced Vital Capacity',
                     'Pulse', 'Diastolic Blood Pressure', 'Systolic Blood Pressure', 'bmi', 'bulbar score',
                     'upper limbs score', 'trunk score', 'lower limbs score', 'mitos movement', 'mitos swallowing',
                     'mitos communicating', 'mitos breathing', 'mitos total', 'kings bulbar', 'kings arm',
                     'kings leg', 'kings total', 'ft9 bulbar', 'ft9 fine motor', 'ft9 gross motor', 'ft9 respiratory',
                     'ft9 total', 'decline rate']
    fitness(train=train_df, subset=features_mrmr, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number)

    print("\nreliefF")
    features_relief = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                       'Q4 Handwriting', 'Q5 Cutting', 'Q5 Indic', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed',
                       'Q8 Walking', 'Q9 Climbing Stairs', 'Q10 Respiratory', 'ALSFRS', 'Symptom Duration',
                       'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'Systolic Blood Pressure', 'bmi',
                       'bulbar score', 'upper limbs score', 'trunk score', 'lower limbs score', 'mitos movement',
                       'mitos swallowing', 'mitos communicating', 'mitos breathing', 'mitos total', 'kings bulbar',
                       'kings arm', 'kings leg', 'kings niv', 'kings total', 'ft9 bulbar', 'ft9 fine motor',
                       'ft9 gross motor', 'ft9 respiratory', 'ft9 total', 'decline rate']
    fitness(train=train_df, subset=features_relief, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number)

    print("\ndifferential evolution")
    features_de = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                   'Q5 Cutting', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Symptom Duration',
                   'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'mitos movement', 'kings niv',
                   'kings total', 'decline rate']
    fitness(train=train_df, subset=features_de, target=target_feature, models=scikit_models, metric=scoring_metric,
            standardisation=std, k=fold_number)


