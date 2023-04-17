from scipy.stats import kstest
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
import numpy as np
from sklearn.feature_selection import f_classif
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2

def correlation_heat_map(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    res = data.corr()
    for col in res.columns:
        res[col] = res[col].apply(lambda x: np.nan if (x < threshold and x > -threshold) else x)
    plt.figure(figsize = (20, 20))
    sns.heatmap(res, cmap = 'Blues')
    return res



def numerical_columns_historgrams(data: pd.DataFrame):
    i = 1
    size = data.shape[1]
    n = data.shape[1] / 4
    n = int(n + size % 4)
    plt.figure(figsize = (22, 5*n))
    for col in data.columns:
        plt.subplot(n, 4, i)
        sns.kdeplot(data[col], label = col)
        i = i + 1
    plt.show()


def categorical_columns_barplots(data: pd.DataFrame):
    i = 1
    size = data.shape[1]
    n = size / 4
    n = int(n + size % 4)
    plt.figure(figsize = (22, 5*n))
    for col in data.columns:
        plt.subplot(n, 4, i)
        pal = sns.color_palette("Blues_r" ,len(data[col].unique()))
        sns.barplot(x = data[col].value_counts().index, y = data[col].value_counts().values, palette = pal) 
        plt.title(col)
        i = i + 1
    plt.show()


def histograms_of_numerical_columns_and_target(data: pd.DataFrame, numerical_columns):
    i = 1
    numerical_columns.sort()
    size = len(numerical_columns) 
    n = len(numerical_columns) / 4
    n = int(n + size % 4)
    
    plt.figure(figsize = (22, 5*n))
    for col in numerical_columns:
        plt.subplot(n, 4, i)
        for cat in data['Dysgraphia'].unique():
            sns.kdeplot(data[data['Dysgraphia'] == cat][col], label = cat)
        i = i + 1
        plt.legend()
    plt.show()




def data_heatmap(data):
    plt.figure(figsize = (20, 20))
    sns.heatmap(data.isna())
    plt.title('Data heatmap')



def kolmogrov_test(data):
    res = pd.DataFrame(index = data.columns, columns = ['Score', 'p_value'])
    for col in data.columns:
        res.loc[col].Score, res.loc[col].p_value = kstest(data[col], 'norm')[0], kstest(data[col], 'norm')[1]
    return res  



def numerical_columns_scatter_plots(data, numerical_columns):
    combinations = [(a, b) for idx, a in enumerate(numerical_columns) for b in numerical_columns[idx + 1:]]
    i = 1
    n = len(combinations) / 4
    n = int(n + len(combinations) % 4)
    plt.figure(figsize = (22, 5*n))
    for comb in combinations:
        plt.subplot(n, 4, i)
        for cat in data['Dysgraphia'].unique():
            plt.scatter(data[data['Dysgraphia'] == cat][comb[0]], data[data['Dysgraphia'] == cat][comb[1]], label = cat)
            plt.xlabel(comb[0])
            plt.ylabel(comb[1])
            plt.legend()
        i = i + 1
    plt.show()        

def numerical_columns_3Dscatter_plots(data, numerical_columns):
    combs = list(combinations(numerical_columns, 3))
    i = 1
    n = len(combs) / 4
    n = int(n + len(combs) % 4)
    fig = plt.figure(figsize = (22, 5*n))
    for comb in combs:
        ax = fig.add_subplot(n, 4, i, projection = '3d')
        ax.scatter(data[data.Dysgraphia == 'Yes'][comb[0]], data[data.Dysgraphia == 'Yes'][comb[1]], data[data.Dysgraphia == 'Yes'][comb[2]], c = 'blue', label = 'Yes')
        ax.scatter(data[data.Dysgraphia == 'No'][comb[0]], data[data.Dysgraphia == 'No'][comb[1]], data[data.Dysgraphia == 'No'][comb[2]], c = 'orange', label = 'No')
        ax.set_xlabel(comb[0])
        ax.set_ylabel(comb[1])
        ax.set_zlabel(comb[2])
        plt.legend()
        i = i + 1
    plt.show() 



def crosstabs_categorical_columns(data: pd.DataFrame, categorical_columns: list):
    i = 1
    combinations = [(a, b) for idx, a in enumerate(categorical_columns) for b in categorical_columns[idx + 1:]]
    size = len(combinations) 
    n = size / 4
    n = int(n + size % 4)

    plt.figure(figsize = (22, 5*n))
    for comb in combinations:
        plt.subplot(n, 4, i)
        sns.heatmap(pd.crosstab(data[comb[0]], data[comb[1]]), annot = True, cmap='Blues', fmt='g')
        plt.legend()
        i = i + 1
    plt.show()


def chi_squared_test(data: pd.DataFrame):
    scores = pd.DataFrame(index = data.index, columns = data.index)
    p_values = pd.DataFrame(index = data.index, columns = data.index)
    data_c = data.copy()
    for col in data.columns:
        l = LabelEncoder()
        x = l.fit_transform(data.Dysgraphia)
        data_c['Dysgraphia'] = x
    plt.figure(figsize = (15, 10))
    plt.subplot(1, 2, 1)
    sns.heatmap(scores, annot = True)
    sns.heatmap(p_values, annot = True)


def chi_squared_test(data: pd.DataFrame, categorical_columns):
    combinations = [(a, b) for idx, a in enumerate(categorical_columns) for b in categorical_columns[idx + 1:]]
    scores = pd.DataFrame(index = categorical_columns, columns = categorical_columns)
    p_values = pd.DataFrame(index = categorical_columns, columns = categorical_columns)
    data_c = data.copy()
    for col in categorical_columns:
        l = LabelEncoder()
        x = l.fit_transform(data[col])
        data_c[col] = x
    for comb in combinations: 
        score, p_value = chi2(np.array(data_c[comb[0]]).reshape(1, -1), np.array(data_c[comb[1]]).reshape(1, -1))
        scores.loc[comb[0]][comb[1]], p_values.loc[comb[0]][comb[1]] = score[0], np.round(p_value[0], 6)
        scores.loc[comb[1]][comb[0]], p_values.loc[comb[1]][comb[0]] = score[0], np.round(p_value[0], 6)
        
    for col in categorical_columns:
        scores.loc[col][col], p_values.loc[col][col] = 99, 0
    return scores, p_values



def t_test(data, numerical_columns):
    t_test = pd.DataFrame(index = numerical_columns, columns = ['Score', 'p_value'])
    n = data.Dysgraphia.value_counts()['Yes'] if data.Dysgraphia.value_counts()['Yes'] < data.Dysgraphia.value_counts()['No'] else data.Dysgraphia.value_counts()['No']
    for col in numerical_columns:
        score, p_value = ttest_ind(np.array(data[data.Dysgraphia == 'No'][col].sample(n, random_state = 0)).reshape(n, ), 
                                   np.array(data[data.Dysgraphia == 'Yes'][col].sample(n, random_state = 0)).reshape(n, ))
        t_test.loc[col]['Score'], t_test.loc[col]['p_value'] = score, p_value
    t_test = t_test.sort_values(by = 'p_value', ascending = True)
    return t_test



def numerical_columns_boxplots(data: pd.DataFrame, numerical_columns: list):
    i = 1
    size = len(numerical_columns) 
    n = size / 3
    n = int(n + size % 3)
    plt.figure(figsize = (22, 6*n))
    for col in numerical_columns:
        plt.subplot(n, 3, i)
        pal = sns.color_palette("Blues_r" ,len(data.Dysgraphia.unique()))
        sns.boxplot(x = data[col])
        sns.boxplot(x = data[col], y = data.Dysgraphia, palette = pal)
        i = i + 1
    plt.show() 


def anova_test(data: pd.DataFrame, numerical_columns: list, random_state: int = 0):
    anova = pd.DataFrame(index = numerical_columns, columns = ['Score', 'p_value'])
    n = data.Dysgraphia.value_counts()['Yes'] if data.Dysgraphia.value_counts()['Yes'] < data.Dysgraphia.value_counts()['No'] else data.Dysgraphia.value_counts()['No']
    for col in numerical_columns:
        score, p_value = f_classif(np.array(data[data.Dysgraphia == 'No'][col].sample(n, random_state = random_state)).reshape(-1, 1), 
                                   np.array(data[data.Dysgraphia == 'Yes'][col].sample(n, random_state = random_state)).reshape(n, ))
        anova.loc[col]['Score'], anova.loc[col]['p_value'] = score[0], p_value[0]
    anova = anova.sort_values(by = 'p_value', ascending = True)
    return anova


def kolmogrov_test(data):
    res = pd.DataFrame(index = data.columns, columns = ['Score', 'p_value'])
    for col in data.columns:
        res.loc[col].Score, res.loc[col].p_value = kstest(data[col], 'norm')[0], kstest(data[col], 'norm')[1]
    return res  


def outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    outliers = data[((data < (q1 - 1.5 * IQR)) | (data > (q3 + 1.5 * IQR)))]
    return outliers


def mann_whitney(data, numerical_columns):
    mann_whitney = pd.DataFrame(index = numerical_columns, columns = ['Score', 'p_value'])
    n = data.Dysgraphia.value_counts()['Yes'] if data.Dysgraphia.value_counts()['Yes'] < data.Dysgraphia.value_counts()['No'] else data.Dysgraphia.value_counts()['No']
    for col in numerical_columns:
        score, p_value = mannwhitneyu(np.array(data[data.Dysgraphia == 'No'][col].sample(n, random_state = 0)).reshape(n, ), 
                                   np.array(data[data.Dysgraphia == 'Yes'][col].sample(n, random_state = 0)).reshape(n, ))
        mann_whitney.loc[col]['Score'], mann_whitney.loc[col]['p_value'] = score, p_value
    mann_whitney = mann_whitney.sort_values(by = 'p_value', ascending = True)
    return mann_whitney

