import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
def data_visual(df):
    df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    df.hist()
    pyplot.show()

    scatter_matrix(df)
    pyplot.show()