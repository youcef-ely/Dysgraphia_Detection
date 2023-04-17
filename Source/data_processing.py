from sklearn import preprocessing
import pandas as pd
import numpy as np
def PolynomialFeatures_labeled(input_df: pd.DataFrame, power: int):
   
    poly = preprocessing.PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s**%d" % (variable,power)
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + " x " + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df


def eliminate_correlated_features(data: pd.DataFrame, threshold: float = 0.8):
    data_c = data.copy()
    corr_matrix = data_c.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data_c = data_c.drop(data_c[to_drop], axis=1)
    return data_c


import numpy as np
from scipy.stats import norm

def grubbs(data, alpha=0.05):
    # Calculate mean and standard deviation of data
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    # Calculate Grubbs' statistic for each data point
    test_stats = [(np.abs(data[i] - mean)) / std for i in range(n)]
    max_test_stat = max(test_stats)
    # Calculate critical value
    critical_val = norm.ppf(1 - (alpha / (2 * n))) * np.sqrt((n - 1) * (n + 1) / (6 * n))
    # Compare Grubbs' statistic to critical value
    if max_test_stat > critical_val:
        return True
    else:
        return False