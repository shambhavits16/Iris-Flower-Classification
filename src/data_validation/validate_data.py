from sklearn.model_selection import train_test_split

def validation(dataset):
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    test_split = train_test_split(X, y, test_size=0.20, random_state=1)
    
    return test_split