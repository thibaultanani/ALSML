import math
import os
import cv2
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import matplotlib.colors as mcolors

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


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


def std(lst):
    return np.std(lst)


def percent_col(data, col):
    print(data[col].value_counts())
    print(data[col].value_counts(normalize=True))


def projection():
    dev = read(filename="als_train")
    val = read(filename="als_test")
    data = pd.concat([dev, val], ignore_index=True, sort=False)

    data['ALSFRS T3'] = data.groupby('ID')['ALSFRS'].transform(lambda x: x.iloc[1])
    data['ALSFRS T6'] = data.groupby('ID')['ALSFRS'].transform(lambda x: x.iloc[2])
    data['ALSFRS T9'] = data.groupby('ID')['ALSFRS'].transform(lambda x: x.iloc[3])
    data.loc[data['Death Date'] <= 3, 'ALSFRS T3'] = 0
    data.loc[data['Death Date'] <= 6, 'ALSFRS T6'] = 0
    data.loc[data['Death Date'] <= 9, 'ALSFRS T9'] = 0

    dev = data.loc[data['Source'] == "proact"]
    val = data.loc[data['Source'] == "exonhit"]
    dev, val = dev.loc[dev['Period'] == 1], val.loc[val['Period'] == 1]
    data = data.loc[data['Period'] == 1]

    target_survival, target_t3, target_t6, target_t9, target_t12 = \
        "Survived", "ALSFRS T3", "ALSFRS T6", "ALSFRS T9", "ALSFRS T12"
    print(dev.shape, dev['Subject ID'].nunique())
    print(val.shape, val['Subject ID'].nunique())
    percent_col(dev, target_survival)
    percent_col(val, target_survival)

    scaler = MinMaxScaler()
    cols = ['Forced Vital Capacity', 'Symptom Duration', 'ALSFRS', 'Height', 'Age', 'Weight', 'Pulse']
    dev_X = dev[cols]
    dev_X = scaler.fit_transform(dev_X)
    dev_X = pd.DataFrame(dev_X, columns=cols)

    trans = umap.UMAP(n_neighbors=15, metric="euclidean", n_components=2, random_state=42)
    trans.fit(dev_X)
    projected = trans.transform(dev_X)
    scaler_ = MinMaxScaler()
    projected = scaler_.fit_transform(projected)

    # Projeter les données
    colors = ['dodgerblue', 'salmon']
    for column in [*dev_X.columns, target_survival, target_t3, target_t6, target_t9, target_t12]:
        if column in [target_survival, target_t3, target_t6, target_t9, target_t12]:
            X = dev
        else:
            X = dev_X
        if X[column].nunique() > 2:
            sc = plt.scatter(projected[:, 0], projected[:, 1], c=dev[column], s=5, cmap='jet')
            # sc.set_clim(vmin=0, vmax=np.max(X[column]))
            # sc.set_array(X[column])
            plt.colorbar()
        else:
            plt.scatter(projected[:, 0], projected[:, 1], c=X[column], s=5,
                        cmap=matplotlib.colors.ListedColormap(colors))
        plt.title(column)
        plt.show()
        plt.close()

    val_X = val[cols]
    val_X = scaler.transform(val_X)
    projected_new = trans.transform(val_X)
    projected_new = scaler_.transform(projected_new)

    plt.scatter(projected_new[:, 0], projected_new[:, 1], c=val[target_survival], s=5,
                cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(target_survival)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.show()
    plt.close()

    for x in [target_t3, target_t6, target_t9, target_t12]:
        plt.scatter(projected_new[:, 0], projected_new[:, 1], c=val[x], s=5, cmap='jet')
        plt.title(x)
        plt.colorbar()
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.show()
        plt.close()

    print([*["ID", "Subject ID", "Source", "X", "Y", "Survived", "ALSFRS T3", "ALSFRS T6", "ALSFRS T9", "ALSFRS T12"],
           *cols])

    df = pd.DataFrame()
    df["ID"] = data["ID"].values
    df["Subject ID"] = data["Subject ID"].values
    df["Source"] = data["Source"].values
    df["X"] = [*projected[:, 0], *projected_new[:, 0]]
    df["Y"] = [*projected[:, 1], *projected_new[:, 1]]
    df["Survived"] = data["Survived"].values
    df["ALSFRS T3"] = data["ALSFRS T3"].values
    df["ALSFRS T6"] = data["ALSFRS T6"].values
    df["ALSFRS T9"] = data["ALSFRS T9"].values
    df["ALSFRS T12"] = data["ALSFRS T12"].values
    df["Symptom Duration"] = data["Symptom Duration"].values
    df["ALSFRS"] = data["ALSFRS"].values
    df["Forced Vital Capacity"] = data["Forced Vital Capacity"].values
    df["Age"] = data["Age"].values
    df["Pulse"] = data["Pulse"].values
    df["Height"] = data["Height"].values
    df["Weight"] = data["Weight"].values
    df["Onset"] = data["Onset Spinal"].values
    df["Gender"] = data["Sex"].values
    df["Diastolic Blood Pressure"] = data["Diastolic Blood Pressure"].values
    df["Systolic Blood Pressure"] = data["Systolic Blood Pressure"].values
    df[['Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing',
        'Q4 Handwriting', 'Q5 Cutting', 'Q5 Indic', 'Q6 Dressing and Hygiene', 'Q7 Turning in Bed',
        'Q8 Walking', 'Q9 Climbing Stairs', 'Q10 Respiratory', ]] = data[
        ['Q1 Speech', 'Q2 Salivation', 'Q3 Swallowing', 'Q4 Handwriting', 'Q5 Cutting', 'Q5 Indic',
         'Q6 Dressing and Hygiene', 'Q7 Turning in Bed', 'Q8 Walking', 'Q9 Climbing Stairs',
         'Q10 Respiratory', ]].values

    write(filename="umap_data", data=df)


def generate_plot(x, y, title, labels, color, type_, type2_):
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_title(title, size=20)
    if type_ == 1:
        ax.scatter(x, y, c=color, s=5, label=labels)
        ax.legend(loc="upper right", fontsize=15)
    elif type_ == 2:
        cdict = {0: 'dodgerblue', 1: 'salmon'}
        for g in np.unique(color):
            i = np.where(color == g)
            ax.scatter(x[i], y[i], c=cdict[g], s=5, label=labels[g])
        ax.legend(loc="upper right", fontsize=15)
    else:
        if title == "Weight" or title == "D. Weight":
            color = [c if c < 130 else 130 for c in color]
        if title == "Symptom Duration" or title == "F. Symptom Duration":
            color = [c if c < 70 else 70 for c in color]
        s = ax.scatter(x, y, c=color, s=5, cmap='jet')
        cbar = plt.colorbar(s, aspect=50, fraction=0.05)
        cbar.ax.tick_params(labelsize=15)
        # fig.colorbar(s, orientation='vertical')
    if type2_ == 1:
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    elif type2_ == 2:
        plt.tick_params(top=False, bottom=False, right=False, labelbottom=False)
    elif type2_ == 3:
        plt.tick_params(top=False, left=False, right=False, labelleft=False)
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    path = "umap_stats/" + title.replace("\n", "") + ".png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()
    return path


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


def calculate_variance(data):
    mean = np.mean(data)
    variance = np.mean([(x - mean) ** 2 for x in data])
    return variance


def calculate_count(data, col):
    value_counts = data[col].value_counts()
    return str(value_counts[0]) + "/" + str(value_counts[1])


def classification(X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver='liblinear', C=10.0)
    # model = KNeighborsClassifier(weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred, average="macro")
    rec = recall_score(y_true=y_test, y_pred=y_pred, average="macro")
    f1s = f1_score(y_true=y_test, y_pred=y_pred, average="macro")

    print()
    print("Accuracy: ", acc)
    print("Precision:", pre)
    print("Recall:   ", rec)
    print("F1-Score: ", f1s)
    print()
    print(matrix)


def generate_plot_rate(matrix, title, cmap, val, div, idx):
    fig, ax = plt.subplots()
    ids, surv = val["ID"].values, val["Survived"].values
    df = val[["X", "Y"]]
    scaler = MinMaxScaler(feature_range=(0, div))
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_scaled["ID"], df_scaled["Survived"] = ids, surv
    df_scaled = df_scaled.loc[df_scaled['ID'].isin(idx)]
    if "ALSFRS" in title and "Variance" not in title:
        plt.pcolormesh(matrix.transpose(), cmap=cmap, edgecolors='k', linewidth=0.5, vmin=0, vmax=40)
    elif "Weight" in title:
        plt.pcolormesh(matrix.transpose(), cmap=cmap, edgecolors='k', linewidth=0.5, vmax=135)
    elif "Symptom Duration" in title:
        plt.pcolormesh(matrix.transpose(), cmap=cmap, edgecolors='k', linewidth=0.5, vmax=50)
    else:
        plt.pcolormesh(matrix.transpose(), cmap=cmap, edgecolors='k', linewidth=0.5)
    cbar = plt.colorbar(aspect=50, fraction=0.05)
    cbar.ax.tick_params(labelsize=15)
    # plt.scatter(df_scaled["X"], df_scaled["Y"], c=df_scaled["Survived"], s=5)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xticks(np.linspace(0, len(matrix[0]), 5))
    ax.set_yticks(np.linspace(0, len(matrix[0]), 5))
    plt.title(title, size=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    path = "umap_stats/" + title + ".png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()
    return path


def round_to_nearest(x, a, b):
    if x - a < b - x:
        return a
    else:
        return b


def confusion_matrix_scores(matrix):
    tn, fp, fn, tp = matrix.ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp)
    balanced_accuracy = (tpr + tnr) / 2
    f1_score = 2 * (precision * tpr) / (precision + tpr)

    return tpr, tnr, precision, balanced_accuracy, f1_score


def square(division):
    data = read("umap_data")
    dev = data.loc[data['Source'] == "proact"]
    val = data.loc[data['Source'] == "exonhit"]
    x_axis, y_axis = part(1, division), part(1, division)
    survival_matrix, alsfrs_t3_matrix, alsfrs_t6_matrix, alsfrs_t9_matrix, alsfrs_t12_matrix = \
        init_matrix_nan(division, division), init_matrix_nan(division, division), init_matrix_nan(division, division), \
        init_matrix_nan(division, division), init_matrix_nan(division, division)
    survival_var_matrix, alsfrs_t3_var_matrix, alsfrs_t6_var_matrix, alsfrs_t9_var_matrix, alsfrs_t12_var_matrix = \
        init_matrix_nan(division, division), init_matrix_nan(division, division), init_matrix_nan(division, division), \
        init_matrix_nan(division, division), init_matrix_nan(division, division)
    survival_spe_matrix, alsfrs_t3_spe_matrix, alsfrs_t6_spe_matrix, alsfrs_t9_spe_matrix, alsfrs_t12_spe_matrix = \
        init_matrix_nan(division, division), init_matrix_nan(division, division), init_matrix_nan(division, division), \
        init_matrix_nan(division, division), init_matrix_nan(division, division)
    # confusion matrix [[TN, FP], [FN, TP]]
    confusion_matrix, y_true, y_pred, y_true_t3_alsfrs, y_true_t6_alsfrs, y_true_t9_alsfrs, y_true_t12_alsfrs, \
    y_pred_t3_alsfrs, y_pred_t6_alsfrs, y_pred_t9_alsfrs, y_pred_t12_alsfrs, idxs = \
        [[0, 0], [0, 0]], [], [], [], [], [], [], [], [], [], [], []
    ids_20, ids_40, ids_60, ids_80, ids_100 = [], [], [], [], []
    idxs_70, idxs_80, idxs_90 = [], [], []
    matrix_sur_70, matrix_als_t3_70, matrix_als_t6_70, matrix_als_t9_70, matrix_als_t12_70 = \
        init_matrix_nan(division, division), init_matrix_nan(division, division), init_matrix_nan(division, division), \
        init_matrix_nan(division, division), init_matrix_nan(division, division)
    matrix_sur_80, matrix_als_t3_80, matrix_als_t6_80, matrix_als_t9_80, matrix_als_t12_80 = \
        init_matrix_nan(division, division), init_matrix_nan(division, division), init_matrix_nan(division, division), \
        init_matrix_nan(division, division), init_matrix_nan(division, division)
    matrix_sur_90, matrix_als_t3_90, matrix_als_t6_90, matrix_als_t9_90, matrix_als_t12_90 = \
        init_matrix_nan(division, division), init_matrix_nan(division, division), init_matrix_nan(division, division), \
        init_matrix_nan(division, division), init_matrix_nan(division, division)
    res = dev["Survived"].value_counts()
    threshold = res[1] / res[0]
    error_t3, error_t6, error_t9, error_t12 = [], [], [], []
    for i in range(len(x_axis) - 1):
        for j in range(len(y_axis) - 1):
            tmp_dev = dev[(x_axis[i] <= dev['X']) & (dev['X'] <= x_axis[i + 1]) &
                          (y_axis[j] <= dev['Y']) & (dev['Y'] <= y_axis[j + 1])]
            tmp_val = val[(x_axis[i] <= val['X']) & (val['X'] <= x_axis[i + 1]) &
                          (y_axis[j] <= val['Y']) & (val['Y'] <= y_axis[j + 1])]
            if len(tmp_dev) < 5:
                survival_rate, alsfrs_t3_mean, alsfrs_t6_mean, alsfrs_t9_mean, alsfrs_t12_mean, \
                survival_var, alsfrs_t3_var, alsfrs_t6_var, alsfrs_t9_var, alsfrs_t12_var = \
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                weights = [1 if x == 1 else threshold for x in tmp_dev['Survived']]
                survival_rate = np.average(tmp_dev['Survived'], weights=weights)
                alsfrs_t3_mean = round(tmp_dev['ALSFRS T3'].sum() / len(tmp_dev), 2)
                alsfrs_t6_mean = round(tmp_dev['ALSFRS T6'].sum() / len(tmp_dev), 2)
                alsfrs_t9_mean = round(tmp_dev['ALSFRS T9'].sum() / len(tmp_dev), 2)
                alsfrs_t12_mean = round(tmp_dev['ALSFRS T12'].sum() / len(tmp_dev), 2)
                survival_var = np.var((tmp_dev['Survived']).values)
                alsfrs_t3_var = np.var((tmp_dev['ALSFRS T3']).values)
                alsfrs_t6_var = np.var((tmp_dev['ALSFRS T6']).values)
                alsfrs_t9_var = np.var((tmp_dev['ALSFRS T9']).values)
                alsfrs_t12_var = np.var((tmp_dev['ALSFRS T12']).values)
            survival_matrix[i][j] = survival_rate
            alsfrs_t3_matrix[i][j] = alsfrs_t3_mean
            alsfrs_t6_matrix[i][j] = alsfrs_t6_mean
            alsfrs_t9_matrix[i][j] = alsfrs_t9_mean
            alsfrs_t12_matrix[i][j] = alsfrs_t12_mean
            survival_var_matrix[i][j] = survival_var
            alsfrs_t3_var_matrix[i][j] = alsfrs_t3_var
            alsfrs_t6_var_matrix[i][j] = alsfrs_t6_var
            alsfrs_t9_var_matrix[i][j] = alsfrs_t9_var
            alsfrs_t12_var_matrix[i][j] = alsfrs_t12_var
            if survival_rate <= 0.2:
                ids_20.extend(tmp_dev["ID"].values)
            elif 0.2 < survival_rate <= 0.4:
                ids_40.extend(tmp_dev["ID"].values)
            elif 0.4 < survival_rate <= 0.6:
                ids_60.extend(tmp_dev["ID"].values)
            elif 0.6 < survival_rate <= 0.8:
                ids_80.extend(tmp_dev["ID"].values)
            elif survival_rate > 0.8:
                ids_100.extend(tmp_dev["ID"].values)
            # if survival_rate <= 0.1 or survival_rate >= 0.9:
            if not math.isnan(survival_rate):
                survival_spe_matrix[i][j] = survival_rate
                alsfrs_t3_spe_matrix[i][j] = alsfrs_t3_mean
                alsfrs_t6_spe_matrix[i][j] = alsfrs_t6_mean
                alsfrs_t9_spe_matrix[i][j] = alsfrs_t9_mean
                alsfrs_t12_spe_matrix[i][j] = alsfrs_t12_mean
                rate = round_to_nearest(survival_rate, 0, 1)
                for index, row in tmp_val.iterrows():
                    if rate == 1 and row["Survived"] == 1:
                        y_true.append(1)
                        y_pred.append(1)
                        confusion_matrix[1][1] = confusion_matrix[1][1] + 1
                    elif rate == 0 and row["Survived"] == 0:
                        y_true.append(0)
                        y_pred.append(0)
                        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                    elif rate == 1 and row["Survived"] == 0:
                        y_true.append(0)
                        y_pred.append(1)
                        confusion_matrix[0][1] = confusion_matrix[0][1] + 1
                    else:
                        y_true.append(1)
                        y_pred.append(0)
                        confusion_matrix[1][0] = confusion_matrix[1][0] + 1
                    idxs.append(row["ID"])
                    y_pred_t3_alsfrs.append(alsfrs_t3_mean)
                    y_true_t3_alsfrs.append(row["ALSFRS T3"])
                    error_t3.append(alsfrs_t3_mean - row["ALSFRS T3"])
                    y_pred_t6_alsfrs.append(alsfrs_t6_mean)
                    y_true_t6_alsfrs.append(row["ALSFRS T6"])
                    error_t6.append(alsfrs_t3_mean - row["ALSFRS T6"])
                    y_pred_t9_alsfrs.append(alsfrs_t9_mean)
                    y_true_t9_alsfrs.append(row["ALSFRS T9"])
                    error_t9.append(alsfrs_t3_mean - row["ALSFRS T9"])
                    y_pred_t12_alsfrs.append(alsfrs_t12_mean)
                    y_true_t12_alsfrs.append(row["ALSFRS T12"])
                    error_t12.append(alsfrs_t3_mean - row["ALSFRS T12"])

    for i in range(len(x_axis) - 1):
        for j in range(len(y_axis) - 1):
            tmp_dev = dev[(x_axis[i] <= dev['X']) & (dev['X'] <= x_axis[i + 1]) &
                          (y_axis[j] <= dev['Y']) & (dev['Y'] <= y_axis[j + 1])]
            tmp_val = val[(x_axis[i] <= val['X']) & (val['X'] <= x_axis[i + 1]) &
                          (y_axis[j] <= val['Y']) & (val['Y'] <= y_axis[j + 1])]
            if len(tmp_dev) < 5:
                survival_rate, alsfrs_t3_mean, alsfrs_t6_mean, alsfrs_t9_mean, alsfrs_t12_mean, \
                survival_var, alsfrs_t3_var, alsfrs_t6_var, alsfrs_t9_var, alsfrs_t12_var = \
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                weights = [1 if x == 1 else threshold for x in tmp_dev['Survived']]
                survival_rate = np.average(tmp_dev['Survived'], weights=weights)
                alsfrs_t3_mean = round(tmp_dev['ALSFRS T3'].sum() / len(tmp_dev), 2)
                alsfrs_t6_mean = round(tmp_dev['ALSFRS T6'].sum() / len(tmp_dev), 2)
                alsfrs_t9_mean = round(tmp_dev['ALSFRS T9'].sum() / len(tmp_dev), 2)
                alsfrs_t12_mean = round(tmp_dev['ALSFRS T12'].sum() / len(tmp_dev), 2)
            for index, row in tmp_val.iterrows():
                if survival_matrix[i][j] <= 0.3 or survival_matrix[i][j] >= 0.7:
                    matrix_sur_70[i][j], matrix_als_t3_70[i][j], matrix_als_t6_70[i][j], \
                    matrix_als_t9_70[i][j], matrix_als_t12_70[i][j] = \
                        survival_rate, alsfrs_t3_mean, alsfrs_t6_mean, alsfrs_t9_mean, alsfrs_t12_mean
                    idxs_70.append(row["ID"])
                if survival_matrix[i][j] <= 0.2 or survival_matrix[i][j] >= 0.8:
                    matrix_sur_80[i][j], matrix_als_t3_80[i][j], matrix_als_t6_80[i][j], \
                    matrix_als_t9_80[i][j], matrix_als_t12_80[i][j] = \
                        survival_rate, alsfrs_t3_mean, alsfrs_t6_mean, alsfrs_t9_mean, alsfrs_t12_mean
                    idxs_80.append(row["ID"])
                if survival_matrix[i][j] <= 0.1 or survival_matrix[i][j] >= 0.9:
                    matrix_sur_90[i][j], matrix_als_t3_90[i][j], matrix_als_t6_90[i][j], \
                    matrix_als_t9_90[i][j], matrix_als_t12_90[i][j] = \
                        survival_rate, alsfrs_t3_mean, alsfrs_t6_mean, alsfrs_t9_mean, alsfrs_t12_mean
                    idxs_90.append(row["ID"])
    print("\n")
    print(classification_report(y_true, y_pred))

    rmse_t3 = math.sqrt(mean_squared_error(y_true=y_true_t3_alsfrs, y_pred=y_pred_t3_alsfrs))
    rmse_t6 = math.sqrt(mean_squared_error(y_true=y_true_t6_alsfrs, y_pred=y_pred_t6_alsfrs))
    rmse_t9 = math.sqrt(mean_squared_error(y_true=y_true_t9_alsfrs, y_pred=y_pred_t9_alsfrs))
    rmse_t12 = math.sqrt(mean_squared_error(y_true=y_true_t12_alsfrs, y_pred=y_pred_t12_alsfrs))
    rsquared = 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - 7 - 1)

    tpr, tnr, precision, balanced_accuracy, f1__score = confusion_matrix_scores(np.array(confusion_matrix))
    print("Confusion Matrix:\n", confusion_matrix)
    print("Sensitivity: {:.4f}".format(tpr))
    print("Specificity: {:.4f}".format(tnr))
    print("Precision: {:.4f}".format(precision))
    print("B. Accuracy: {:.4f}".format(balanced_accuracy))
    print("F1 score: {:.4f}".format(f1__score))
    print("RMSE T3: {:.4f}".format(rmse_t3))
    print("RMSE T6: {:.4f}".format(rmse_t6))
    print("RMSE T9: {:.4f}".format(rmse_t9))
    print("RMSE T12: {:.4f}".format(rmse_t12))
    print("R²: {:.4f}".format(rsquared))

    dev_ = dev.copy()
    val_ = val[["ID", "X", "Y", "Survived"]].copy()
    val = val.loc[val['ID'].isin(idxs)]

    div = division
    fig1 = generate_plot_rate(matrix=survival_matrix, title="1-year Survival", cmap='RdYlGn', val=val_, div=div,
                              idx=idxs)
    fig1_1 = generate_plot_rate(matrix=survival_matrix, title="A. 1-year Survival", cmap='RdYlGn', val=val_, div=div,
                                idx=idxs)
    fig2_1 = generate_plot_rate(matrix=alsfrs_t3_matrix, title="3-month ALSFRS", cmap='RdYlGn', val=val_, div=div,
                                idx=idxs)
    fig2_2 = generate_plot_rate(matrix=alsfrs_t6_matrix, title="6-month ALSFRS", cmap='RdYlGn', val=val_, div=div,
                                idx=idxs)
    fig2_3 = generate_plot_rate(matrix=alsfrs_t9_matrix, title="9-month ALSFRS", cmap='RdYlGn', val=val_, div=div,
                                idx=idxs)
    fig2_4 = generate_plot_rate(matrix=alsfrs_t12_matrix, title="1-year ALSFRS", cmap='RdYlGn', val=val_, div=div,
                                idx=idxs)
    fig3 = generate_plot_rate(matrix=survival_var_matrix, title="1-year Survival (Variance)", cmap='coolwarm',
                              val=val_, div=div, idx=idxs)
    fig4_1 = generate_plot_rate(matrix=alsfrs_t3_var_matrix, title="3-month ALSFRS (Variance)", cmap='coolwarm',
                                val=val_, div=div,
                                idx=idxs)
    fig4_2 = generate_plot_rate(matrix=alsfrs_t6_var_matrix, title="6-month ALSFRS (Variance)", cmap='coolwarm',
                                val=val_, div=div,
                                idx=idxs)
    fig4_3 = generate_plot_rate(matrix=alsfrs_t9_var_matrix, title="9-month ALSFRS (Variance)", cmap='coolwarm',
                                val=val_, div=div,
                                idx=idxs)
    fig4_4 = generate_plot_rate(matrix=alsfrs_t12_var_matrix, title="1-year ALSFRS (Variance)", cmap='coolwarm',
                                val=val_, div=div,
                                idx=idxs)
    fig5 = generate_plot_rate(matrix=matrix_sur_70, title="1-year Survival Rate (70%)", cmap='RdYlGn',
                              val=val_, div=div, idx=idxs_70)
    fig6_1 = generate_plot_rate(matrix=matrix_als_t3_70, title="3-month ALSFRS (70%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_70)
    fig6_2 = generate_plot_rate(matrix=matrix_als_t6_70, title="6-month ALSFRS (70%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_70)
    fig6_3 = generate_plot_rate(matrix=matrix_als_t9_70, title="9-month ALSFRS (70%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_70)
    fig6_4 = generate_plot_rate(matrix=matrix_als_t12_70, title="1-year ALSFRS (70%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_70)
    fig7 = generate_plot_rate(matrix=matrix_sur_80, title="1-year Survival Rate (80%)", cmap='RdYlGn',
                              val=val_, div=div, idx=idxs_80)
    fig8_1 = generate_plot_rate(matrix=matrix_als_t3_80, title="3-month ALSFRS (80%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_80)
    fig8_2 = generate_plot_rate(matrix=matrix_als_t6_80, title="6-month ALSFRS (80%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_80)
    fig8_3 = generate_plot_rate(matrix=matrix_als_t9_80, title="9-month ALSFRS (80%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_80)
    fig8_4 = generate_plot_rate(matrix=matrix_als_t12_80, title="1-year ALSFRS (80%)", cmap='RdYlGn', val=val_,
                                div=div, idx=idxs_80)
    fig9 = generate_plot_rate(matrix=matrix_sur_90, title="1-year Survival Rate (90%)", cmap='RdYlGn',
                              val=val_, div=div, idx=idxs_90)
    fig10_1 = generate_plot_rate(matrix=matrix_als_t3_90, title="3-month ALSFRS (90%)", cmap='RdYlGn', val=val_,
                                 div=div, idx=idxs_90)
    fig10_2 = generate_plot_rate(matrix=matrix_als_t6_90, title="6-month ALSFRS (90%)", cmap='RdYlGn', val=val_,
                                 div=div, idx=idxs_90)
    fig10_3 = generate_plot_rate(matrix=matrix_als_t9_90, title="9-month ALSFRS (90%)", cmap='RdYlGn', val=val_,
                                 div=div, idx=idxs_90)
    fig10_4 = generate_plot_rate(matrix=matrix_als_t12_90, title="1-year ALSFRS (90%)", cmap='RdYlGn', val=val_,
                                 div=div, idx=idxs_90)

    # figures = [fig1, fig2, fig3, fig4]
    figures = [fig2_1, fig2_2, fig2_3, fig2_4]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        img = cv2.imread(figures[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    path = "umap_stats/umap_strat.png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()

    figures = [fig3, fig4_1, fig4_2, fig4_3, fig4_4]
    fig, axs = plt.subplots(1, 5, figsize=(15, 15))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        img = cv2.imread(figures[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    path = "umap_stats/umap_variance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()

    figures = [fig5, fig6_1, fig6_2, fig6_3, fig6_4, fig7, fig8_1, fig8_2, fig8_3, fig8_4, fig9,
               fig10_1, fig10_2, fig10_3, fig10_4]
    fig, axs = plt.subplots(3, 5, figsize=(12, 7))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        img = cv2.imread(figures[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    path = "umap_stats/umap_proba.png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()

    dev_ = dev_[
        ['ID', 'Gender', 'Onset', 'Age', 'Height', 'Weight', 'ALSFRS', 'Symptom Duration', 'Forced Vital Capacity',
         'Pulse', "Diastolic Blood Pressure", "Systolic Blood Pressure"]]
    df_ids_20 = dev_.loc[dev_['ID'].isin(ids_20)]
    df_ids_40 = dev_.loc[dev_['ID'].isin(ids_40)]
    df_ids_60 = dev_.loc[dev_['ID'].isin(ids_60)]
    df_ids_80 = dev_.loc[dev_['ID'].isin(ids_80)]
    df_ids_100 = dev_.loc[dev_['ID'].isin(ids_100)]

    stats_20, stats_40, stats_60, stats_80, stats_100 = pd.Series(dtype='float64'), pd.Series(dtype='float64'), \
                                                        pd.Series(dtype='float64'), pd.Series(
        dtype='float64'), pd.Series(dtype='float64')
    for col in [x for x in dev_.columns if x != "ID"]:
        if dev_[col].nunique() <= 2:
            stats_20[col] = calculate_count(df_ids_20, col)
            stats_40[col] = calculate_count(df_ids_40, col)
            stats_60[col] = calculate_count(df_ids_60, col)
            stats_80[col] = calculate_count(df_ids_80, col)
            stats_100[col] = calculate_count(df_ids_100, col)
        else:
            stats_20[col] = "{}:{}".format(round(df_ids_20[col].mean(), 2), round(std(df_ids_20[col].values), 2))
            stats_40[col] = "{}:{}".format(round(df_ids_40[col].mean(), 2), round(std(df_ids_40[col].values), 2))
            stats_60[col] = "{}:{}".format(round(df_ids_60[col].mean(), 2), round(std(df_ids_60[col].values), 2))
            stats_80[col] = "{}:{}".format(round(df_ids_80[col].mean(), 2), round(std(df_ids_80[col].values), 2))
            stats_100[col] = "{}:{}".format(round(df_ids_100[col].mean(), 2), round(std(df_ids_100[col].values), 2))

    stats = pd.concat([stats_20, stats_40, stats_60, stats_80, stats_100], axis=1)
    stats = stats.reset_index()
    stats.rename(columns={'index': 'Survival Rate', 0: '0:20', 1: '20:40', 2: '40:60', 3: '60:80', 4: '80:100'},
                 inplace=True)

    write(filename="stats_survidev_rate", data=stats)


def check_close(a, b, x, interval):
    if a < interval[0] or a > interval[1] or b < interval[0] or b > interval[1]:
        raise ValueError("Les nombres doivent être dans l'intervalle spécifié.")

    threshold = (x / 100) * (interval[1] - interval[0])
    if a >= (interval[1] - (threshold / 2)):
        return abs(a - b) <= ((threshold / 2) + (a - (interval[1] - (threshold / 2))))
    elif a <= (interval[0] + (threshold / 2)):
        return abs(a - b) <= ((threshold / 2) + (a + (interval[0] + (threshold / 2))))
    else:
        return abs(a - b) <= (threshold / 2)


def res_interval(x, y, percent, interval):
    res = []
    res_x, res_y = [], []
    for i in range(len(x)):
        if not math.isnan(x[i]) and not math.isnan(y[i]):
            if check_close(x[i], y[i], percent, interval):
                res.append(True)
            else:
                res.append(False)
            res_x.append(x[i])
            res_y.append(y[i])
    print(res_x)
    print(res_y)
    return res


def square3(division):
    data = read("umap_data")
    dev = data.loc[data['Source'] == "proact"]
    val = data.loc[data['Source'] == "exonhit"]
    x_axis, y_axis = part(1, division), part(1, division)
    res_sex, res_onset = dev["Gender"].value_counts(), dev["Onset"].value_counts()
    print(res_sex)
    print(res_onset)
    sex_matrix, age_matrix, weight_matrix, height_matrix, alsfrs_matrix, duration_matrix, fvc_matrix, pulse_matrix, \
    dbp_matrix, sbp_matrix, onset_matrix = init_matrix_nan(division, division), init_matrix_nan(division, division), \
                                           init_matrix_nan(division, division), init_matrix_nan(division, division),\
                                           init_matrix_nan(division, division), init_matrix_nan(division, division), \
                                           init_matrix_nan(division, division), init_matrix_nan(division, division), \
                                           init_matrix_nan(division, division), init_matrix_nan(division, division),\
                                           init_matrix_nan(division, division)
    for i in range(len(x_axis) - 1):
        for j in range(len(y_axis) - 1):
            tmp_dev = dev[(x_axis[i] <= dev['X']) & (dev['X'] <= x_axis[i + 1]) &
                          (y_axis[j] <= dev['Y']) & (dev['Y'] <= y_axis[j + 1])]
            tmp_val = val[(x_axis[i] <= val['X']) & (val['X'] <= x_axis[i + 1]) &
                          (y_axis[j] <= val['Y']) & (val['Y'] <= y_axis[j + 1])]
            if len(tmp_dev) < 5:
                sex_rate, onset_rate, alsfrs_mean, age_mean, weight_mean, height_mean, duration_mean, fvc_mean, \
                pulse_mean, dbp_mean, sbp_mean = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                                                 np.nan, np.nan
            else:
                sex_rate = round(tmp_dev["Gender"].sum() / len(tmp_dev), 2)
                onset_rate = round(tmp_dev["Onset"].sum() / len(tmp_dev), 2)
                alsfrs_mean = round(tmp_dev["ALSFRS"].sum() / len(tmp_dev), 2)
                age_mean = round(tmp_dev["Age"].sum() / len(tmp_dev), 2)
                weight_mean = round(tmp_dev["Weight"].sum() / len(tmp_dev), 2)
                height_mean = round(tmp_dev["Height"].sum() / len(tmp_dev), 2)
                duration_mean = round(tmp_dev["Symptom Duration"].sum() / len(tmp_dev), 2)
                fvc_mean = round(tmp_dev["Forced Vital Capacity"].sum() / len(tmp_dev), 2)
                pulse_mean = round(tmp_dev["Pulse"].sum() / len(tmp_dev), 2)
                dbp_mean = round(tmp_dev["Diastolic Blood Pressure"].sum() / len(tmp_dev), 2)
                sbp_mean = round(tmp_dev["Systolic Blood Pressure"].sum() / len(tmp_dev), 2)
            sex_matrix[i][j], onset_matrix[i][j], alsfrs_matrix[i][j], age_matrix[i][j], weight_matrix[i][j], \
            height_matrix[i][j], duration_matrix[i][j], fvc_matrix[i][j], pulse_matrix[i][j], dbp_matrix[i][j], \
            sbp_matrix[i][j] = sex_rate, onset_rate, alsfrs_mean, age_mean, weight_mean, height_mean, duration_mean, \
                               fvc_mean, pulse_mean, dbp_mean, sbp_mean

    val_ = val[["ID", "X", "Y", "Survived"]].copy()
    div = division
    idxs = val["ID"].values
    fig1 = "umap_stats/A. 1-year Survival.png"
    fig2 = generate_plot_rate(matrix=sex_matrix, title="B. Gender", cmap='RdYlGn', val=val_, div=div, idx=idxs)
    fig3 = generate_plot_rate(matrix=onset_matrix, title="C. Onset", cmap='RdYlGn', val=val_, div=div, idx=idxs)
    fig4 = generate_plot_rate(matrix=age_matrix, title="D. Age", cmap='RdYlGn', val=val_, div=div, idx=idxs)
    fig5 = generate_plot_rate(matrix=weight_matrix, title="E. Weight", cmap='RdYlGn', val=val_, div=div, idx=idxs)
    fig6 = generate_plot_rate(matrix=height_matrix, title="F. Height", cmap='RdYlGn', val=val_, div=div, idx=idxs)
    fig7 = generate_plot_rate(matrix=alsfrs_matrix, title="G. ALSFRS", cmap='RdYlGn', val=val_, div=div, idx=idxs)
    fig8 = generate_plot_rate(matrix=duration_matrix, title="H. Symptom Duration", cmap='RdYlGn', val=val_, div=div,
                              idx=idxs)
    fig9 = generate_plot_rate(matrix=fvc_matrix, title="I. Forced Vital Capacity", cmap='RdYlGn', val=val_, div=div,
                              idx=idxs)
    fig10 = generate_plot_rate(matrix=pulse_matrix, title="J. Pulse", cmap='RdYlGn', val=val_, div=div, idx=idxs)
    fig11 = generate_plot_rate(matrix=dbp_matrix, title="K. Diastolic Blood Pressure", cmap='RdYlGn', val=val_, div=div,
                               idx=idxs)
    fig12 = generate_plot_rate(matrix=sbp_matrix, title="L. Systolic Blood Pressure", cmap='RdYlGn', val=val_, div=div,
                               idx=idxs)

    figures = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12]
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        img = cv2.imread(figures[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.autoscale(False)
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    path = "umap_stats/umap_all.png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == '__main__':
    projection()
    square(division=20)
    square3(division=20)
