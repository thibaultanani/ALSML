import os
import pandas as pd

from multiprocessing import Process
from feature_selections.filters import NoSelection, Correlation, Anova, MutualInformation, Mrmr
from feature_selections.heuristics.population_based import Differential

from sklearn.linear_model import RidgeClassifier


def read(filename, separator=','):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


if __name__ == '__main__':
    train = read(filename="als_train")
    test = read(filename="als_test")
    name = "res"
    target = "Survived"
    metric = "recall"
    tmax = 600

    train, test = train.loc[train['Period'] == 1], test.loc[test['Period'] == 1]
    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival', 'ALSFRS T12']
    m = [RidgeClassifier(random_state=42, class_weight="balanced")]

    all_ = NoSelection(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops)
    corr = Correlation(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops,
                       k=16)
    anov = Anova(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops, k=16)
    info = MutualInformation(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax,
                             drops=drops, k=22)
    mrmr = Mrmr(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops, k=16)
    diff = Differential(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax,
                        drops=drops)

    methods = [all_, corr, anov, info, mrmr, diff]

    processes = []
    for i in range(len(methods)):
        if type(methods[i]) in [type(x) for x in methods[:i]]:
            processes.append(Process(target=methods[i].start, args=(2,)))
        else:
            processes.append(Process(target=methods[i].start, args=(1,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Finish !")
