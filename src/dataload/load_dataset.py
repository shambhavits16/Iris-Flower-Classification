import pandas as pd

def load_data(url, names):
    # "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    df = pd.read_csv(url , names=names)
    return df
   