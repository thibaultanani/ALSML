import multiprocessing
import os
import math
import traceback
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from feature_selections.filters import Filter
from feature_selections.heuristics import Tide
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def read(filename, separator=','):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


if __name__ == '__main__':
    train, test = read(filename="new_als_train"), None
    scoring = balanced_accuracy_score
    name = "als_result"
    target = "Survived"
    tmax = 3600
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    verbose = True
    suffixes = ['_lr', '_knn', '_tree', '_forest', '_lgbm']
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T3', 'ALSFRS T6',
             'ALSFRS T9', 'ALSFRS T12']
    methods = [LogisticRegression(random_state=42, solver="lbfgs", class_weight="balanced", max_iter=10000),
               KNeighborsClassifier(weights='distance', algorithm='kd_tree', n_neighbors=int(math.sqrt(train.shape[0] / 10))),
               DecisionTreeClassifier(random_state=42, class_weight="balanced"),
               RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50),
               LGBMClassifier(class_weight="balanced", random_state=42, verbosity=-1, n_estimators=50)]
    for (m, suffix) in zip(methods, suffixes):
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', m)])
        name = "results" + suffix
        anov = Filter(name=name, target=target, train=train, test=test, cv=cv,
                      drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                      verbose=verbose, method="Anova")
        tide = Tide(name=name, target=target, train=train, test=test, cv=cv,
                    drops=drops, scoring=scoring, pipeline=pipeline, Tmax=tmax,
                    suffix='_anova', verbose=verbose)
        selection_methods = [anov, tide]
        num_processes = multiprocessing.cpu_count()
        results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(method.start, j + 1): j for j, method in enumerate(selection_methods)}
            for future in as_completed(futures):
                id_method = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"An error has occurred with the method {id_method + 1}: {str(e)}")
                    traceback.print_exc()
        results = sorted(results, key=lambda x: x[0], reverse=True)
        results_str = "\nRankings:\n\n"
        i = 0
        for result in results:
            i = i + 1
            clf_name = result[3].steps[-1][1].__class__.__name__
            results_str += (f"Rang: {i}, method: {result[5]}, pid: {result[4]}, "
                            f"score: {round(result[0], 4)}, model: {clf_name}, "
                            f"n selected: {len(result[2])}, convergence: {result[6]}, "
                            f"n iter: {result[7]}, subset: {result[2]}\n")
        print(results_str)
        res_path = os.path.join(os.getcwd(), os.path.join(os.path.join('out', name), 'res.txt'))
        with open(res_path, "w") as file:
            file.write(results_str)