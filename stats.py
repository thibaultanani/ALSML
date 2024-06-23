import numpy as np
import pandas as pd
from pandas._config import display

from als_test import read, write


def model_stats(df, label):
    results = []
    for feature in df.columns:
        try:
            feature_stats = {'Feature': feature, f'Avg': df[feature].mean().round(2),f'Std': df[feature].std().round(2),
                             f'Min': df[feature].min().round(2), f'Max': df[feature].max().round(2),
                             f'(nb patients)': len(df[feature])}
        except np.core._exceptions._UFuncOutputCastingError as e:
            feature_stats = {'Feature': feature, f'Avg': df[feature].mean().round(2), f'Std': None, f'Min': None,
                             f'Max': None, f'(nb patients)': len(df[feature])}
        results.append(feature_stats)
    df_results = pd.DataFrame(results)
    print(df_results)
    write(data=df_results, filename=f'{label}_feature_stats')
    return df_results


if __name__ == '__main__':
    train = read(filename="new_als_train")
    test = read(filename="new_als_test")
    train, test = train.loc[train['Period'] == 1], test.loc[test['Period'] == 1]

    data = pd.concat([train, test], ignore_index=True, sort=False)
    pro_act = data.loc[data['Source'] == "proact"]
    exonhit = data.loc[data['Source'] == "exonhit"]

    drops = ['ID', 'ExID', 'Period', 'Subject ID', 'Source', 'Death Date', 'Survival']
    data = data.drop(drops, axis=1)
    pro_act = pro_act.drop(drops, axis=1)
    exonhit = exonhit.drop(drops, axis=1)

    proact_df = model_stats(pro_act, 'proact')
    exonhit_df = model_stats(exonhit, 'exonhit')
    overall_df = model_stats(data, 'all')

    latex_lines = []
    for feature in overall_df['Feature'].values:
        proact_values = proact_df[proact_df['Feature'] == feature].iloc[0][['Avg', 'Std', 'Min', 'Max']]
        exonhit_values = exonhit_df[exonhit_df['Feature'] == feature].iloc[0][['Avg', 'Std', 'Min', 'Max']]
        overall_values = overall_df[overall_df['Feature'] == feature].iloc[0][['Avg', 'Std', 'Min', 'Max']]

        line = f"{feature} & {proact_values['Avg']} & {proact_values['Std']} & {proact_values['Min']} & {proact_values['Max']} & {exonhit_values['Avg']} & {exonhit_values['Std']} & {exonhit_values['Min']} & {exonhit_values['Max']} & {overall_values['Avg']} & {overall_values['Std']} & {overall_values['Min']} & {overall_values['Max']} \\\\"
        line = line.replace("nan", "--")
        latex_lines.append(line)

    for line in latex_lines:
        print(line)

