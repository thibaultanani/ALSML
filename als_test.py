import math
import os
import numpy as np
import pandas as pd
import shap

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, r2_score, mean_squared_error, accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


output_folder = "figures"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


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
    if np.isnan(row['Death Date']) or row['Death Date'] > 13:
        return 13
    else:
        return row['Death Date']


def event_occurred(row):
    if np.isnan(row['Death Date']) or row['Death Date'] > 13:
        return False
    else:
        return True


def launch_regre(train_, test_, pulse_, models_, target_, type=1):
    ms = []
    real_df_test, pred_df_test = [], []
    real_df_pulse, pred_df_pulse = [], []
    rmse_df = []

    for i in range(4):
        train_ = train_.loc[train_[target_[i]] > 0]
        print(train_.shape, train.shape)
        test_ = test_.loc[test_[target_[i]] > 0]
        pulse_ = pulse_.loc[pulse_[target_[i]] > 0]
        X_train = train_.drop(target_, axis=1)
        X_test = test_.drop(target_, axis=1)
        X_pulse = pulse_.drop(target_, axis=1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_pulse_scaled = scaler.transform(X_pulse)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        X_pulse_scaled = pd.DataFrame(X_pulse_scaled, columns=X_pulse.columns)

        m = models_[i].fit(X_train_scaled, train_[target_[i]])
        y_pred_test = m.predict(X_test_scaled)
        y_test = test_[target_[i]]
        rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))
        rsquared_adj = (
            1 - (1 - r2_score(y_test, y_pred_test)) * (len(y_test) - 1) / (len(y_test) - X_train.shape[1] - 1)
        )
        rsquared = r2_score(y_test, y_pred_test)
        pcc = np.corrcoef(y_test, y_pred_test)[0, 1]
        print(
            f"(Test) {target_[i]} RMSE: {rmse:.4f}, rsquared: {rsquared:.4f}, rsquared_adj: {rsquared_adj:.4f}, PCC: {pcc:.4f}"
        )

        ms.append(m)
        real_df_test.append(y_test)
        pred_df_test.append(y_pred_test)
        rmse_df.append(rmse)

        y_pred_pulse = m.predict(X_pulse_scaled)
        y_pulse = pulse_[target_[i]]
        real_df_pulse.append(y_pulse)
        pred_df_pulse.append(y_pred_pulse)

        rmse = math.sqrt(mean_squared_error(y_pulse, y_pred_pulse))
        rsquared_adj = (
            1 - (1 - r2_score(y_pulse, y_pred_pulse)) * (len(y_pulse) - 1) / (len(y_pulse) - X_train.shape[1] - 1)
        )
        rsquared = r2_score(y_pulse, y_pred_pulse)
        pcc = np.corrcoef(y_pulse, y_pred_pulse)[0, 1]
        print(f"(Pulse) {target_[i]} RMSE: {rmse:.4f}, rsquared: {rsquared:.4f}, rsquared_adj: {rsquared_adj:.4f}, PCC: {pcc:.4f}")

    model_titles = ["T3", "T6", "T9", "T12"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        ax.set_facecolor('lightgray')
        ax.grid(color='white', zorder=0)
        ax.scatter(real_df_test[i], pred_df_test[i], label='Test', color='tab:blue', zorder=2)
        ax.scatter(real_df_pulse[i], pred_df_pulse[i], label='Pulse', color='tab:orange', zorder=2)

        if type == 1:
            ax.plot([-1, 41], [-1, 41], 'r--', zorder=3)
            ax.set_xlim([-1, 41])
            ax.set_ylim([-1, 41])
        else:
            ax.plot([-1, 49], [-1, 49], 'r--', zorder=3)
            ax.set_xlim([-1, 49])
            ax.set_ylim([-1, 49])

        ax.set_title(model_titles[i], fontsize=15)
        ax.set_xlabel('Reality', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)

    from matplotlib.lines import Line2D
    custom_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='Test'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', markersize=10, label='Pulse')
    ]
    fig.legend(handles=custom_handles, loc='lower center', ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path_scatter = os.path.join(output_folder, "scatter_models.jpg")
    plt.savefig(path_scatter, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()
    return ms


def launch(train_, test_, pulse_, model_, target_, name):
    X_train = train_.drop(target_, axis=1)
    y_train = train_[target_]
    X_test = test_.drop(target_, axis=1)
    y_test = test_[target_]
    X_pulse = pulse_.drop(target_, axis=1)
    y_pulse = pulse_[target_]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_pulse_scaled = scaler.transform(X_pulse)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_pulse_scaled = pd.DataFrame(X_pulse_scaled, columns=X_pulse.columns)

    model_.fit(X_train_scaled, y_train)

    y_pred_test = model_.predict(X_test_scaled)
    y_proba_test = model_.predict_proba(X_test_scaled)[:, 1]
    matrix_test = confusion_matrix(y_test, y_pred_test)
    score_test = recall_score(y_true=y_test, y_pred=y_pred_test, average="macro")

    y_pred_pulse = model_.predict(X_pulse_scaled)
    y_proba_pulse = model_.predict_proba(X_pulse_scaled)[:, 1]
    matrix_pulse = confusion_matrix(y_pulse, y_pred_pulse)
    score_pulse = recall_score(y_true=y_pulse, y_pred=y_pred_pulse, average="macro")

    y_pred_train = model_.predict(X_train_scaled)
    matrix_train = confusion_matrix(y_train, y_pred_train)
    score_train = recall_score(y_true=y_train, y_pred=y_pred_train, average="macro")

    print(f'\n{name}:\n')
    print(f'Score Train: {score_train}\nMatrice de confusion Train:\n{matrix_train}\n')
    print(f'Score Test: {score_test}\nMatrice de confusion Test:\n{matrix_test}\n')
    print(f'Score Pulse: {score_pulse}\nMatrice de confusion Pulse:\n{matrix_pulse}\n')

    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    auc_score = roc_auc_score(y_test, y_proba_test)
    fprp, tprp, _ = roc_curve(y_pulse, y_proba_pulse)
    auc_scorep = roc_auc_score(y_pulse, y_proba_pulse)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('lightgray')
    plt.grid(color='white', zorder=0)
    plt.plot(fpr, tpr, label=f'ROC Curve Test (AUC = {auc_score:.2f})', color='tab:blue')
    plt.plot(fprp, tprp, label=f'ROC Curve Pulse (AUC = {auc_scorep:.2f})', color='tab:orange')

    plt.plot([0, 1], [0, 1], linestyle='--', color='tab:red', label='Random Classifier')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.tight_layout()

    nom_roc = f"roc_curve_{name.replace(' ', '_')}.jpg"
    chemin_roc = os.path.join(output_folder, nom_roc)
    plt.savefig(chemin_roc, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()


    merged_ = pd.concat([X_test_scaled, X_pulse_scaled], ignore_index=True)
    explainer = shap.Explainer(model_, X_train_scaled)
    shap_values = explainer(merged_)

    shap_values_values = np.array([val.values for val in shap_values])
    if not np.all(np.isfinite(shap_values_values)):
        raise ValueError("Les valeurs SHAP contiennent des NaN ou des inf")

    shap.summary_plot(shap_values, merged_, plot_type="bar", max_display=50, show=False)
    nom_shap_bar = f"shap_bar_{name.replace(' ', '_')}.jpg"
    chemin_shap_bar = os.path.join(output_folder, nom_shap_bar)
    plt.savefig(chemin_shap_bar, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values, merged_, max_display=50, show=False)
    nom_shap_dot = f"shap_dot_{name.replace(' ', '_')}.jpg"
    chemin_shap_dot = os.path.join(output_folder, nom_shap_dot)
    plt.savefig(chemin_shap_dot, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()

    X_test_scaled[target_], X_pulse_scaled[target_] = y_test.values, y_pulse.values
    return model_, X_test_scaled, X_pulse_scaled


def predict_confidence(model_, test_, pulse_, target_):
    X_test, y_test = test_.drop(target_, axis=1).values, test_[target_].values
    X_pulse, y_pulse = pulse_.drop(target_, axis=1).values, pulse_[target_].values

    probas_test = model_.predict_proba(X_test)[:, 1]
    probas_pulse = model_.predict_proba(X_pulse)[:, 1]

    y_merged = np.concatenate([y_test, y_pulse])
    probas_merged = np.concatenate([probas_test, probas_pulse])

    zones = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    results_test = []
    results_pulse = []
    results_merged = []

    for (low, high) in zones:
        mask_test = (probas_test >= low) & (probas_test < high)
        if np.sum(mask_test) > 0:
            y_pred_zone_test = (probas_test[mask_test] >= 0.5).astype(int)
            accuracy_test = accuracy_score(y_test[mask_test], y_pred_zone_test)
            count_test = np.sum(mask_test)
        else:
            accuracy_test = None
            count_test = 0
        results_test.append((low, high, accuracy_test, count_test))

        mask_pulse = (probas_pulse >= low) & (probas_pulse < high)
        if np.sum(mask_pulse) > 0:
            y_pred_zone_pulse = (probas_pulse[mask_pulse] >= 0.5).astype(int)
            accuracy_pulse = accuracy_score(y_pulse[mask_pulse], y_pred_zone_pulse)
            count_pulse = np.sum(mask_pulse)
        else:
            accuracy_pulse = None
            count_pulse = 0
        results_pulse.append((low, high, accuracy_pulse, count_pulse))

        mask_merged = (probas_merged >= low) & (probas_merged < high)
        if np.sum(mask_merged) > 0:
            y_pred_zone_merged = (probas_merged[mask_merged] >= 0.5).astype(int)
            accuracy_merged = accuracy_score(y_merged[mask_merged], y_pred_zone_merged)
            count_merged = np.sum(mask_merged)
        else:
            accuracy_merged = None
            count_merged = 0
        results_merged.append((low, high, accuracy_merged, count_merged))

    df_test = pd.DataFrame(results_test, columns=['Zone min', 'Zone max', 'Exactitude Test', 'Nombre Test'])
    df_pulse = pd.DataFrame(results_pulse, columns=['Zone min', 'Zone max', 'Exactitude Pulse', 'Nombre Pulse'])
    df_merged = pd.DataFrame(results_merged, columns=['Zone min', 'Zone max', 'Exactitude Ensemble', 'Nombre Ensemble'])

    print(df_test)
    print(df_pulse)
    print(df_merged)
    return probas_test, probas_pulse, probas_merged


def model_stats(predictions_, test_, pulse_, target_):
    test_ = pd.concat([test_, pulse_], ignore_index=True)
    X_test = test_.drop(target_, axis=1)
    probas = predictions_
    X_test['probas'] = probas
    zones = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    zone_labels = ['[0,20)', '[20,40)', '[40,60)', '[60,80)', '[80,100]']
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
                          labels=['[0,20)', '[20,40)', '[40,60)', '[60,80)', '[80,100]'])
    df['Time'] = time_
    df['Event'] = event_
    print(df)
    print(len(df.loc[df['Event'] == True]), len(df.loc[df['Event'] == False]))
    kmf = KaplanMeierFitter()
    group_colors = {
        '[80,100]': 'tab:blue',
        '[60,80)': 'tab:green',
        '[40,60)': 'tab:olive',
        '[20,40)': 'tab:orange',
        '[0,20)': 'tab:red'
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('lightgray')
    plt.grid(color='white', zorder=0)
    for name, grouped_df in df.groupby('groups'):
        kmf.fit(grouped_df['Time'], event_observed=grouped_df['Event'], label=name)
        kmf.plot_survival_function(ax=ax, color=group_colors[name])
    plt.xlabel('Time (month)')
    plt.ylabel('Survival Probability')
    plt.legend(title='Survival Clusters')
    plt.tight_layout()
    plt.grid()

    chemin_km = os.path.join(output_folder, "kaplan_meier.jpg")
    plt.savefig(chemin_km, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()

    c_index = concordance_index(time_, predictions_, event_)
    print(f'C-index: {c_index:.4f}')
    results = multivariate_logrank_test(df['Time'], df['groups'], df['Event'])
    print(f'Global Log-Rank test p-value: {results.p_value:.4f}')


def cox_model(train_, test_, pulse_, target_):
    X_train = train_.drop([target_, 'Time', 'Event'], axis=1)
    X_test = test_.drop([target_, 'Time', 'Event'], axis=1)
    X_pulse = pulse_.drop([target_, 'Time', 'Event'], axis=1)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    X_pulse_scaled = pd.DataFrame(scaler.transform(X_pulse), columns=X_pulse.columns, index=X_pulse.index)
    X_train_scaled['Time'] = train_['Time']
    X_train_scaled['Event'] = train_['Event']
    X_test_scaled['Time'] = test_['Time']
    X_test_scaled['Event'] = test_['Event']
    X_pulse_scaled['Time'] = pulse_['Time']
    X_pulse_scaled['Event'] = pulse_['Event']
    cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.0)
    cph.fit(X_train_scaled, duration_col='Time', event_col='Event')
    cph.print_summary()

    ax = cph.plot()
    plt.tight_layout()

    dossier_cox = "cox_out"
    if not os.path.exists(dossier_cox):
        os.makedirs(dossier_cox)
    chemin_cox = os.path.join(dossier_cox, "summary.jpg")
    plt.savefig(chemin_cox, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()

    c_index_test = cph.score(X_test_scaled, scoring_method="concordance_index")
    print(f"C-index test: {c_index_test:.4f}")
    c_index_pulse = cph.score(X_pulse_scaled, scoring_method="concordance_index")
    print(f"C-index pulse: {c_index_pulse:.4f}")


def impute_pulse_from_train(train_df, pulse_df, target_col, max_iter, sample_posterior, vars_to_impute):
    vars_for_mice = [c for c in train_df.columns if c != target_col]
    pulse_vals = pulse_df[vars_for_mice].copy()
    imputer = IterativeImputer(random_state=42, max_iter=max_iter, sample_posterior=sample_posterior)
    imputer.fit(train_df[vars_for_mice])
    imputed_array = imputer.transform(pulse_vals)
    pulse_imputed = pulse_df.copy()
    for j, col in enumerate(vars_for_mice):
        if col in vars_to_impute:
            pulse_imputed[col] = imputed_array[:, j]
    return pulse_imputed


if __name__ == '__main__':
    train = read(filename="new_als_train")
    test = read(filename="new_als_test")
    pulse = read(filename="new_als_test_pulse")
    print("===1-year Survival Classification Model===")
    model_class = LogisticRegression(random_state=42, class_weight='balanced', solver='lbfgs', max_iter=10000)
    target_class, target_alsfrs = "Survived", ['ALSFRS T3', 'ALSFRS T6', 'ALSFRS T9', 'ALSFRS T12']
    metric = "recall"
    drops_class = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T3', 'ALSFRS T6',
                   'ALSFRS T9', 'ALSFRS T12']
    drops_class_pulse = ['ID', 'Death Date', 'ALSFRS T3', 'ALSFRS T6', 'ALSFRS T9', 'ALSFRS T12']
    drops_regre = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'Survived']
    drops_regre_pulse = ['ID', 'Death Date', 'Survived']
    features = ['Gender', 'Age', 'Weight', 'Height', 'Onset', 'Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
                'Q5 Cutting', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Symptom Duration',
                'Forced Vital Capacity', 'Pulse', 'Diastolic Blood Pressure', 'mitos movement', 'kings niv',
                'kings total', 'decline rate']

    train_class, test_class, pulse_class = (
        train.drop(drops_class, axis=1),
        test.drop(drops_class, axis=1),
        pulse.drop(drops_class_pulse, axis=1)
    )

    vars_to_impute = ['Pulse', 'Diastolic Blood Pressure', 'Systolic Blood Pressure']
    pulse_class[vars_to_impute] = np.nan
    pulse_class = impute_pulse_from_train(train_class, pulse_class, target_class, 20, False, vars_to_impute)
    col_order = list(train_class.columns)
    pulse_class = pulse_class.reindex(columns=col_order)

    launch(train_class, test_class, pulse_class, model_class, target_class, "Without Selection")
    train_class, test_class, pulse_class = (
        train_class[features + [target_class]],
        test_class[features + [target_class]],
        pulse_class[features + [target_class]]
    )
    model1, test_scaled, pulse_scaled = launch(
        train_class, test_class, pulse_class, model_class, target_class, "Differential Evolution"
    )
    predictions = predict_confidence(model1, test_scaled, pulse_scaled, target_class)[2]
    model_stats(predictions, test_class, pulse_class, target_class)
    merged = pd.concat([test, pulse], ignore_index=True)
    time, event = (
        merged.apply(lambda row: find(row), axis=1),
        merged.apply(lambda row: event_occurred(row), axis=1)
    )
    kaplan_meier(time, event, predictions)
    print(model1.coef_)
    print(model1.intercept_)
    print("\n\n")
    time_train, event_train = (
        train.apply(lambda row: find(row), axis=1),
        train.apply(lambda row: event_occurred(row), axis=1)
    )
    time_test, event_test = (
        test.apply(lambda row: find(row), axis=1),
        test.apply(lambda row: event_occurred(row), axis=1)
    )
    time_pulse, event_pulse = (
        pulse.apply(lambda row: find(row), axis=1),
        pulse.apply(lambda row: event_occurred(row), axis=1)
    )
    train_class['Time'], test_class['Time'], pulse_class['Time'] = time_train, time_test, time_pulse
    train_class['Event'], test_class['Event'], pulse_class['Event'] = event_train, event_test, event_pulse
    cox_model(train_class, test_class, pulse_class, target_class)

    train_regre, test_regre, pulse_regre = (
        train.drop(drops_regre, axis=1),
        test.drop(drops_regre, axis=1),
        pulse.drop(drops_regre_pulse, axis=1)
    )
    pulse_regre[vars_to_impute] = np.nan
    pulse_regre = impute_pulse_from_train(train_regre, pulse_regre, target_class, 20, False, vars_to_impute)
    pulse_regre = pulse_regre[train_regre.columns]
    models_regre = [
        LGBMRegressor(verbosity=-1, random_state=42, n_estimators=50),
        LGBMRegressor(verbosity=-1, random_state=42, n_estimators=50),
        LGBMRegressor(verbosity=-1, random_state=42, n_estimators=50),
        LGBMRegressor(verbosity=-1, random_state=42, n_estimators=50)
    ]
    model2 = launch_regre(train_regre, test_regre, pulse_regre, models_regre, target_alsfrs)
