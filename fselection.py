import multiprocessing
import os
import traceback
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from feature_selections.filters import Filter
from feature_selections.heuristics.population_based import Differential
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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
    k = 10
    std = True
    metric = balanced_accuracy_score
    name = "als_result"
    target = "Survived"
    tmax = 3600
    verbose = True
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T3', 'ALSFRS T6',
             'ALSFRS T9', 'ALSFRS T12']
    m = [LogisticRegression(random_state=42, solver="lbfgs", class_weight="balanced", max_iter=10000),
         RidgeClassifier(random_state=42, class_weight="balanced"),
         KNeighborsClassifier(n_neighbors=10, weights='distance'),
         DecisionTreeClassifier(random_state=42, class_weight='balanced'),
         RandomForestClassifier(random_state=42, class_weight='balanced'),
         LGBMClassifier(class_weight="balanced", verbosity=-1, random_state=42),
         LinearDiscriminantAnalysis(), GaussianNB()]
    corr = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                  metric=metric, model=m, Tmax=tmax, verbose=verbose, method="Correlation")
    anov = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                  metric=metric, model=m, Tmax=tmax, verbose=verbose, method="Anova")
    info = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                  metric=metric, model=m, Tmax=tmax, verbose=verbose, method="Mutual Information")
    mrmr = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                  metric=metric, model=m, Tmax=tmax, verbose=verbose, method="MRMR")
    reli = Filter(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                  metric=metric, model=m, Tmax=tmax, verbose=verbose, method="ReliefF")
    diff = Differential(name=name, target=target, train=train, test=test, k=k, standardisation=std, drops=drops,
                        metric=metric, model=m, Tmax=tmax, verbose=verbose)
    selection_methods = [corr, anov, info, mrmr, reli, diff]
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
        results_str += (f"Rang: {i}, method: {result[5]}, pid: {result[4]}, score: {round(result[0], 4)},"
                        f" classifier: {result[3].__class__.__name__}, n selected: {len(result[2])},"
                        f" convergence: {result[6]}, n iter: {result[7]},"
                        f" subset: {result[2]}\n")
    print(results_str)
    res_path = os.path.join(os.getcwd(), os.path.join(os.path.join('out', name), 'res.txt'))
    with open(res_path, "w") as file:
        file.write(results_str)