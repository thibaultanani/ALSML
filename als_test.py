import math
import os
import numpy as np
import pandas as pd
import shap

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, recall_score, r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler


def read(filename, separator=','):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


def write(filename, data):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data.to_excel(path + '.xlsx', index=False)
    except FileNotFoundError:
        data.to_csv(path + '.csv', index=False)
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


def find(row):
    if np.isnan(row['Death Date']) or row['Death Date'] > 12.1:
        return 12.1
    else:
        return row['Death Date']


def event_occurred(row):
    if np.isnan(row['Death Date']) or row['Death Date'] > 12.1:
        return False
    else:
        return True


def launch_regre(train_, test_, models_, target_, type=1):
    X_train = train_.drop(target_, axis=1)
    X_test = test_.drop(target_, axis=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    ms, real_df, pred_df, rmse_df = [], [], [], []
    for i in range(4):
        m = models_[i].fit(X_train_scaled, train_[target_[i]])
        y_pred = m.predict(X_test_scaled)
        y_test = test_[target_[i]]
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        rsquared_adj = (1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_train.shape[1] - 1))
        rsquared = r2_score(y_test, y_pred)
        pcc = np.corrcoef(y_test, y_pred)[0, 1]
        print(f"{target_[i]} RMSE: {rmse:.4f}, rsquared: {rsquared:.4f}, rsquared_adj: {rsquared_adj:.4f},"
              f" PCC: {pcc:.4f}")
        ms.append(m)
        real_df.append(y_test), pred_df.append(y_pred), rmse_df.append(rmse)

    model_titles = ["T3", "T6", "T9", "T12"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        ax.set_facecolor('lightgray')
        ax.grid(color='white', zorder=0)
        ax.scatter(real_df[i], pred_df[i], label='Patients', zorder=2)
        if type == 1:
            ax.plot([-1, 41], [-1, 41], 'r--', label='Identity', zorder=3)
            ax.set_xlim([-1, 41])
            ax.set_ylim([-1, 41])
        else:
            ax.plot([-1, 49], [-1, 49], 'r--', label='Identity', zorder=3)
            ax.set_xlim([-1, 49])
            ax.set_ylim([-1, 49])
        ax.set_title(model_titles[i], fontsize=15)
        ax.set_xlabel('Reality', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.annotate(f'RMSE: {rmse_df[i]:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    # Ajouter une légende commune
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    return ms


def launch(train_, test_, model_, target_, name):
    # Séparation des caractéristiques et de la cible
    X_train = train_.drop(target_, axis=1)
    y_train = train_[target_]
    X_test = test_.drop(target_, axis=1)
    y_test = test_[target_]
    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Convertir les données normalisées en DataFrame pour conserver les noms des colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    # Entraînement du modèle
    model_.fit(X_train_scaled, y_train)
    # Prédiction sur les données de test
    y_pred_test = model_.predict(X_test_scaled)
    matrix_test = confusion_matrix(y_test, y_pred_test)
    score_test = recall_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    # Prédiction sur les données d'entraînement
    y_pred_train = model_.predict(X_train_scaled)
    matrix_train = confusion_matrix(y_train, y_pred_train)
    score_train = recall_score(y_true=y_train, y_pred=y_pred_train, average="macro")
    # Affichage des résultats
    print(f'\n{name}:\n')
    print(f'Score Train: {score_train}\nMatrice de confusion Train:\n{matrix_train}\n')
    print(f'Score Test: {score_test}\nMatrice de confusion Test:\n{matrix_test}\n')
    # Calcul des valeurs SHAP sur les données normalisées
    explainer = shap.Explainer(model_, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    # Vérifiez les valeurs SHAP
    shap_values_values = np.array([val.values for val in shap_values])
    if not np.all(np.isfinite(shap_values_values)):
        raise ValueError("Les valeurs SHAP contiennent des NaN ou des inf")
    # Affichage des valeurs SHAP avec les noms des variables
    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", max_display=50)
    shap.summary_plot(shap_values, X_test_scaled, max_display=50)
    X_test_scaled[target_] = y_test.values
    return model_, X_test_scaled


def predict_confidence(model_, test_, target_):
    X_test, y_test = test_.drop(target_, axis=1).values, test_[target_].values
    probas = model_.predict_proba(X_test)[:, 1]
    zones = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    results = []
    for (low, high) in zones:
        mask = (probas >= low) & (probas < high)
        if np.sum(mask) > 0:
            y_pred_zone = (probas[mask] >= 0.5).astype(int)
            accuracy = accuracy_score(y_test[mask], y_pred_zone)
            results.append((low, high, accuracy, np.sum(mask)))
    df_results = pd.DataFrame(results, columns=['Zone min', 'Zone max', 'Exactitude', 'Nombre d\'échantillons'])
    print(df_results)
    return probas


def model_stats(predictions_, test_, target_):
    X_test = test_.drop(target_, axis=1)
    probas = predictions_
    X_test['probas'] = probas
    zones = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    zone_labels = ['0:20', '20:40', '40:60', '60:80', '80:100']
    results = []
    for feature in X_test.columns[:-1]:
        feature_stats = {'Feature': feature}
        for (low, high), label in zip(zones, zone_labels):
            mask = (X_test['probas'] >= low) & (X_test['probas'] < high)
            feature_stats[f'{label} Avg'] = X_test.loc[mask, feature].mean().round(2)
            feature_stats[f'{label} Std'] = X_test.loc[mask, feature].std().round(2)
            feature_stats[f'{label} (nb patients)'] = mask.sum()
        results.append(feature_stats)
    df_results = pd.DataFrame(results)
    print(df_results)
    write(data=df_results, filename='feature_stats')


def kaplan_meier(time_, event_, predictions_):
    df = pd.DataFrame()
    df['probas'] = predictions_
    print(predictions_)
    df['groups'] = pd.cut(df['probas'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                          labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    df['Time'] = time_
    df['Event'] = event_
    print(df)
    print(len(df.loc[df['Event'] == True]), len(df.loc[df['Event'] == False]))
    kmf = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('lightgray')
    plt.grid(color='white', zorder=0)
    for name, grouped_df in df.groupby('groups'):
        kmf.fit(grouped_df['Time'], event_observed=grouped_df['Event'], label=name)
        kmf.plot_survival_function()
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.legend(title='Probability Cluster')
    plt.tight_layout()
    plt.grid()
    plt.show()
    plt.close()
    c_index = concordance_index(time_, predictions_, event_)
    print(f'C-index: {c_index:.2f}')
    results = multivariate_logrank_test(df['Time'], df['groups'], df['Event'])
    print(f'Global Log-Rank test p-value: {results.p_value:.4f}')


if __name__ == '__main__':
    train = read(filename="new_als_train")
    test = read(filename="new_als_test")
    print("===1-year Survival Classification Model===")
    model_class = LogisticRegression(random_state=42, class_weight='balanced', solver='lbfgs', max_iter=10000)
    target_class, target_alsfrs = "Survived", ['ALSFRS T3', 'ALSFRS T6', 'ALSFRS T9', 'ALSFRS T12']
    metric = "recall"
    drops_class = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T3', 'ALSFRS T6',
                   'ALSFRS T9', 'ALSFRS T12']
    drops_regre = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'Survived']
    features = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                'Q5 Cutting', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Symptom Duration',
                'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'mitos movement', 'kings niv',
                'kings total', 'decline rate']
    # Survival analysis
    time, event = test.apply(lambda row: find(row), axis=1), test.apply(lambda row: event_occurred(row), axis=1)
    train_class, test_class = train.drop(drops_class, axis=1), test.drop(drops_class, axis=1)
    launch(train_class, test_class, model_class, target_class, "Without Selection")
    train_class, test_class = train_class[features + [target_class]], test_class[features + [target_class]]
    model1, test_scaled = launch(train_class, test_class, model_class, target_class, "Differential Evolution")
    predictions = predict_confidence(model1, test_scaled, target_class)
    model_stats(predictions, test_class, target_class)
    kaplan_meier(time, event, predictions)
    print(model1.coef_)
    print(model1.intercept_)
    print("\n\n")

    # Disease progression
    train_regre, test_regre = (train.loc[train['Survived'] == True].drop(drops_regre, axis=1),
                               test.loc[test['Survived'] == True].drop(drops_regre, axis=1))
    models_regre = [RandomForestRegressor(random_state=42), RandomForestRegressor(random_state=42),
                    RandomForestRegressor(random_state=42), RandomForestRegressor(random_state=42)]
    model2 = launch_regre(train_regre, test_regre, models_regre, target_alsfrs)

    # ALSFRS-R survival
    train = read(filename="new_als_R_train")
    test = read(filename="new_als_R_test")
    model_class = RidgeClassifier(random_state=42, class_weight='balanced')
    target_class, target_alsfrs = "Survived", ['ALSFRSR T3', 'ALSFRSR T6', 'ALSFRSR T9', 'ALSFRSR T12']
    drops_class = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRSR T3', 'ALSFRSR T6',
                   'ALSFRSR T9', 'ALSFRSR T12']
    drops_regre = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'Survived']
    features = ['Gender', 'Age', 'Height', 'Q1 Speech', 'Q2 Salivation', 'Q5 Indic', 'Q7 Turning in Bed',
                'Q8 Walking', 'Q10 Respiratory', 'ALSFRS', 'Symptom Duration', 'Forced Vital Capacity', 'Pulse',
                'Diastolic Blood Pressure', 'ALSFRSR', 'R 1 Dyspnea', 'R 2 Orthopnea',
                'R 3 Respiratory Insufficiency', 'bmi', 'upper limbs score', 'lower limbs score', 'mitos movement',
                'mitos communicating', 'mitos total', 'kings leg', 'kings niv', 'kings total', 'ft9 bulbar',
                'ft9 total']
    time, event = test.apply(lambda row: find(row), axis=1), test.apply(lambda row: event_occurred(row), axis=1)
    train_class, test_class = train.drop(drops_class, axis=1), test.drop(drops_class, axis=1)
    launch(train_class, test_class, model_class, target_class, "Without Selection")
    train_class, test_class = train_class[features + [target_class]], test_class[features + [target_class]]
    model1, test_scaled = launch(train_class, test_class, model_class, target_class, "Differential Evolution")
    train_regre, test_regre = (train.loc[train['Survived'] == True].drop(drops_regre, axis=1),
                               test.loc[test['Survived'] == True].drop(drops_regre, axis=1))
    models_regre = [Ridge(random_state=42), Ridge(random_state=42),
                    Ridge(random_state=42), Ridge(random_state=42)]
    launch_regre(train_regre, test_regre, models_regre, target_alsfrs, 2)