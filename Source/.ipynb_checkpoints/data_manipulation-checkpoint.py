import pandas as pd


def read_data(file_name: str = ''):
    return pd.read_excel('C:/Users/User/Desktop/Dyslexia project/Data/'+file_name)


def save_data(data: pd.DataFrame, file_name: str = ''):
    data.to_excel('C:/Users/User/Desktop/Dyslexia project/Data/'+file_name+'.xlsx', index = False)
    
    
def sort_dataFrame(data: pd.DataFrame):
    first_letters = [col[0] for col in data.columns]
    sorted_cols = sorted(data.columns, key = lambda x: first_letters.index(x[0]))
    data = data[sorted_cols]
    return data